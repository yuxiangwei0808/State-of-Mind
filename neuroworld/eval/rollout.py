"""Latent rollout errors, persistence reference, fMRI metrics, world-model controls, trajectories.

The direct fMRI forecaster is evaluated with the same keys, measured in fMRI units.
World-model controls (fMRI space; need a readout or the forecaster) give each
window a fixed donor window from the same subject and a different stimulus
(data/splits.py::content_key, so never a repeated viewing of the same clip):
``mismatch`` keeps the window's brain prefix but uses the donor's stimulus rows;
``state_swap`` keeps the window's stimulus rows but starts from the donor's
brain prefix. They separate the stimulus (exogenous) and state (endogenous)
contributions to the rollout. Windows of a subject with no other stimulus in the split
(e.g. an HCP subject with one usable clip) have no donor and are left out of the controls.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from neuroworld.data.splits import content_key
from neuroworld.eval.fmri import FmriAccumulator
from neuroworld.eval.intersubject import intersubject_reference


def move_batch(batch, device):
    return {key: ({name: value.to(device) for name, value in item.items()}
                  if isinstance(item, dict) else item.to(device) if torch.is_tensor(item) else item)
            for key, item in batch.items()}


def summarize_errors(errors, persistence_errors, count):
    mse = (errors / count).tolist()
    persistence = (persistence_errors / count).tolist()
    relative = [error / reference if reference > 1e-12 else None
                for error, reference in zip(mse, persistence)]
    defined = [value for value in relative if value is not None]
    # Scale-free rollout score: raw latent MSE shrinks with the latent scale, not model quality.
    return {"next_state_mse": mse[0], "rollout_mse": sum(mse) / len(mse),
            "mean_relative_mse": sum(defined) / len(defined) if defined else None,
            "mse_by_horizon": mse, "persistence_mse_by_horizon": persistence,
            "relative_mse_by_horizon": relative, "windows": count}


class Donors:
    """Deterministic donor window per evaluation window: same subject, different stimulus content
    (index -1 where the subject saw no other stimulus in the split)."""

    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        subjects = np.array([dataset.records[run]["subject_id"] for run, _ in dataset.windows])
        contents = np.array([content_key(dataset.records[run]["timeline_id"]) for run, _ in dataset.windows])
        rng = np.random.default_rng(seed)
        self.index = np.full(len(dataset), -1, dtype=np.int64)
        for subject in np.unique(subjects):
            rows = np.flatnonzero(subjects == subject)
            for row in rows:
                pool = rows[contents[rows] != contents[row]]
                if len(pool):
                    self.index[row] = pool[rng.integers(len(pool))]
        if (self.index < 0).all():
            raise ValueError("Controls need a subject who saw two stimuli in the split")
        self.missing = int((self.index < 0).sum())

    def rows(self, start, stop):
        """Positions within [start, stop) of the windows that have a donor."""
        return np.flatnonzero(self.index[start:stop] >= 0)

    def batch(self, start, stop):
        """The donors of the windows in [start, stop) that have one, in order."""
        return default_collate([self.dataset[int(j)] for j in self.index[start:stop] if j >= 0])

    def pairs(self):
        def window(i):
            run, start = self.dataset.windows[i]
            return {"run_id": self.dataset.records[run]["run_id"], "start": int(start)}
        return [{"window": window(i), "donor": window(int(j))} for i, j in enumerate(self.index) if j >= 0]


def clear_trajectories(output_dir):
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-running evaluation replaces its generated trajectories.
    for path in output_dir.glob("batch-*.pt"):
        path.unlink()
    return output_dir


def _rollout(model, prefix, stimuli, subject_index, readout, horizon):
    """(latents or None, fMRI or None) for one set of observed prefix frames and stimulus rows."""
    if getattr(model, "forecasts_fmri", False):
        return None, model.rollout_frames(prefix, stimuli, subject_index, horizon_trs=horizon)
    latents = model.rollout(prefix, stimuli, horizon_trs=horizon)
    return latents, None if readout is None else readout(latents, subject_index)


@torch.no_grad()
def evaluate(model, loader, device, *, prefix_trs=1, horizon_trs=20, output_dir=None,
             readout=None, controls=False, identify=False, limits=None, donor_seed=0,
             intersubject=False):
    """Roll out every window from its observed prefix and score the predicted horizon.

    Errors are latent (world model) or fMRI (forecaster), with a persistence reference.
    fMRI metrics need a readout or the forecaster. ``limits`` adds fMRI metrics over the
    first steps of the rollout (e.g. 20, 50, 100) from the same starting windows.
    ``intersubject`` adds the approximate inter-subject correlation references on the same TRs.
    """
    forecaster = getattr(model, "forecasts_fmri", False)
    if forecaster and readout is not None:
        raise ValueError("The fMRI forecaster already contains its readout")
    model.eval()
    if readout is not None:
        readout.eval()
    k, h0, horizon = model.encoder.state_window_trs, prefix_trs, horizon_trs
    limits = sorted(set(limits or [horizon]))
    if limits[-1] != horizon or limits[0] < 1:
        raise ValueError("Metric limits must lie in 1..horizon and include the horizon")
    decodes = forecaster or readout is not None
    names = ["model", "persistence"] + (["encoded"] if readout is not None else [])
    donors, skipped = None, None
    if controls and decodes:
        try:
            donors = Donors(loader.dataset, seed=donor_seed)
            names += ["state_swap"] + (["mismatch"] if model.mode == "stimulus" else [])
        except ValueError as error:
            skipped = str(error)
            print(f"Skipping world-model controls: {skipped}", flush=True)
    contents = {record["run_id"]: content_key(record["timeline_id"]) for record in loader.dataset.records}
    fmri = {limit: {name: FmriAccumulator(limit, identify=identify and name != "persistence",
                                          content_of=lambda run_id: contents.get(run_id, run_id))
                    for name in names} for limit in limits} if decodes else None
    errors = torch.zeros(horizon, dtype=torch.float64)
    persistence_errors = torch.zeros_like(errors)
    subjects, count, offset = {}, 0, 0
    evaluated_frames = {} if intersubject and fmri is not None else None
    latent_count, latent_sum, latent_square_sum = 0, None, None
    output_dir = clear_trajectories(output_dir)
    for index, raw in enumerate(loader):
        batch = move_batch(raw, device)
        brain = batch["brain"]
        if brain.shape[1] != k + h0 + horizon - 1:
            raise ValueError("Evaluation windows must hold prefix_trs + horizon_trs states")
        prefix, observed = brain[:, :k + h0 - 1], brain[:, k + h0 - 1:]
        # Target state h0+j ends at frame k+h0-1+j; the last observed frame is k+h0-2.
        last_frame = brain[:, k + h0 - 2][:, None].expand(-1, horizon, -1)
        head = model.readout if forecaster else readout
        subject_index = None if head is None else head.subject_indices(raw["subject_id"], device)
        latents, decoded = _rollout(model, prefix, batch["stimuli"], subject_index, readout, horizon)
        if forecaster:
            predicted, targets, reference = decoded, observed, last_frame
        else:
            states = model.encode(brain)
            predicted, targets, reference = latents, states[:, h0:], states[:, h0 - 1][:, None]
            values = targets.flatten(0, 1).cpu().double()
            if latent_sum is None:
                latent_sum = torch.zeros(values.shape[-1], dtype=torch.float64)
                latent_square_sum = torch.zeros_like(latent_sum)
            latent_sum += values.sum(dim=0)
            latent_square_sum += values.square().sum(dim=0)
            latent_count += len(values)
        per_sample = (predicted - targets).square().mean(dim=-1).cpu().double()
        # The same last OBSERVED state (or frame) is repeated at every horizon.
        persistence = (reference - targets).square().mean(dim=-1).cpu().double()
        if not torch.isfinite(per_sample).all() or not torch.isfinite(persistence).all():
            raise FloatingPointError("Non-finite rollout evaluation error")
        errors += per_sample.sum(dim=0)
        persistence_errors += persistence.sum(dim=0)
        count += brain.shape[0]
        for row, subject in enumerate(raw["subject_id"]):
            item = subjects.setdefault(str(subject), {"errors": torch.zeros_like(errors),
                                                      "persistence": torch.zeros_like(errors), "count": 0})
            item["errors"] += per_sample[row]
            item["persistence"] += persistence[row]
            item["count"] += 1
        if fmri is not None:
            outputs = {"model": decoded, "persistence": last_frame}
            if "encoded" in names:
                outputs["encoded"] = readout(targets, subject_index)
            controls, kept = {}, None
            if donors is not None:
                kept = donors.rows(offset, offset + brain.shape[0])
                if len(kept):
                    donor = move_batch(donors.batch(offset, offset + brain.shape[0]), device)
                    rows = torch.as_tensor(kept, device=device)
                    own_stimuli = {name: value[rows] for name, value in batch["stimuli"].items()}
                    controls["state_swap"] = _rollout(model, donor["brain"][:, :k + h0 - 1], own_stimuli,
                                                      subject_index[rows], readout, horizon)[1]
                    if "mismatch" in names:
                        controls["mismatch"] = _rollout(model, prefix[rows], donor["stimuli"], subject_index[rows],
                                                        readout, horizon)[1]
            for limit, accumulators in fmri.items():
                for name, value in outputs.items():
                    accumulators[name].add(raw["run_id"], raw["subject_id"], value[:, :limit], observed[:, :limit])
                for name, value in controls.items():
                    accumulators[name].add([raw["run_id"][i] for i in kept], [raw["subject_id"][i] for i in kept],
                                           value[:, :limit], observed[rows][:, :limit])
        if evaluated_frames is not None:
            for run_id, frames in zip(raw["run_id"], raw["state_frame_indices"][:, h0:].numpy()):
                evaluated_frames.setdefault(run_id, []).append(frames)
        offset += brain.shape[0]
        if output_dir is not None:
            payload = {"run_id": raw["run_id"], "subject_id": raw["subject_id"],
                       "timeline_id": raw["timeline_id"], "tr_seconds": raw["tr_seconds"],
                       "target_frame_indices": raw["state_frame_indices"][:, h0:],
                       "target_times_s": raw["physical_time_s"][:, k + h0 - 1:]}
            if not forecaster:
                payload.update(predicted_latents=latents.cpu(), target_latents=targets.cpu(),
                               last_observed_latent=reference[:, 0].cpu())
            if decoded is not None:
                payload["predicted_fmri"] = decoded.cpu()
            torch.save(payload, output_dir / f"batch-{index:06d}.pt")
    if not count:
        raise ValueError("Evaluation loader yielded no windows")
    report = summarize_errors(errors, persistence_errors, count)
    report.update(space="fmri" if forecaster else "latent", prefix_trs=h0, horizon_trs=horizon)
    if forecaster:
        report["latent_spread"] = None
    else:
        # Population spread over all evaluation target vectors, not mean batch std.
        variance = latent_square_sum / latent_count - (latent_sum / latent_count).square()
        report["latent_spread"] = float(variance.clamp_min(0).sqrt().mean())
    report["by_subject"] = {subject: summarize_errors(item["errors"], item["persistence"], item["count"])
                            for subject, item in sorted(subjects.items())}
    if fmri is not None:
        report["fmri"] = {name: accumulator.summary() for name, accumulator in fmri[horizon].items()}
        if len(limits) > 1:
            report["fmri_by_limit"] = {str(limit): {name: accumulator.summary()
                                                    for name, accumulator in accumulators.items()}
                                       for limit, accumulators in fmri.items()}
    if evaluated_frames is not None:
        references = intersubject_reference(loader.dataset, {run: np.stack(frames) for run, frames
                                                             in evaluated_frames.items()}, limits)
        if references[horizon] is not None:
            report["intersubject"] = references[horizon]
            if len(limits) > 1:
                report["intersubject_by_limit"] = {str(limit): value for limit, value in references.items()}
    if donors is not None:
        report["controls"] = {"pairing": "same subject, different stimulus", "seed": donor_seed,
                              "windows_without_donor": donors.missing}
        report["_pairs"] = donors.pairs()
    elif skipped:
        report["controls"] = {"skipped": skipped}
    return report


def output_folder(split="val", horizons=False):
    """evaluation, evaluation-horizons, evaluation-dev, evaluation-dev-horizons, ..."""
    return "evaluation" + ("" if split == "val" else f"-{split}") + ("-horizons" if horizons else "")
