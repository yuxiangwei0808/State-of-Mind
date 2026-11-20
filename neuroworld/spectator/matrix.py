"""Step 4 of Spectator's training (and the JEPA phase's start matrix): the predictor's matrix, in closed form.

The predictor steps the state as s_{t+1} = s_t + [s_t ; m_t ; 1] C, with m_t the movie rows exactly as the model reads them
(standardized per feature when model.stimulus_scaling is standardize; the statistics come from every training row, as
the trainer sets them). C is a ridge regression of one-step state changes on the training windows:

    penalty = lam * mean(diag(Gram)) * I, times `state_factor` on the state block, times `movie_factor` on the movie block,
              and 0 on the constant.

With partial windows (the final fit), the training set also holds windows that start before their run, with those frames
zero, as the benchmark gives them to a model whose queries are as long as its window. Each keeps at least 30 s before its
first state inside the run, as every forecast starting point does. Under data.history_fill (HCP), the oldest frames of a
window (window minus fast frames) may be missing and enter as zero; its recent frames must be valid and inside the split.

Several movie factors can be fitted from one pass over the data; each matrix goes to <out stem>-m<factor>.pt.
With rollmix, the fit also holds transitions from the given matrix's own rollouts (an ablation, not part of Spectator).
"""
import json
from pathlib import Path

import numpy as np
import torch

from neuroworld.benchmark.grid import frames_within
from neuroworld.data.bundle import read_jsonl
from neuroworld.data.schaefer_v2_dataset import SchaeferWindowDataset
from neuroworld.data.splits import frame_ranges
from neuroworld.models import build_model
from neuroworld.train.checkpoint import FORMAT
from neuroworld.train.stage_utils import build_dataset, build_loader, load_encoder_init, window_settings


class PartialWindows(SchaeferWindowDataset):
    """Windows with a negative start read zeros for the frames before the run start."""

    def __getitem__(self, index):
        run_index, start = map(int, self.windows[index])
        if start >= 0:
            return super().__getitem__(index)
        record = self.records[run_index]
        brain = self._open(record["brain"])
        block = np.zeros((self.length, brain.shape[1]), dtype=np.float32)
        block[-start:] = brain[0:start + self.length]
        fill = self.history_fill_frames
        if fill > -start:
            frames = np.arange(0, start + fill)  # run frames inside the oldest `fill` frames
            block[frames - start] *= record["history_ok"][frames][:, None]
        frames = np.arange(start, start + self.length, dtype=np.int64)
        result = {"brain": torch.from_numpy(block), "stimuli": {}, "frame_indices": torch.from_numpy(frames),
                  "state_frame_indices": torch.from_numpy(frames[self.state_window_trs - 1:].copy()),
                  "physical_time_s": torch.from_numpy(record["origin"] + frames * self.tr_seconds),
                  "tr_seconds": self.tr_seconds, "run_id": record["run_id"], "subject_id": record["subject_id"],
                  "timeline_id": record["timeline_id"]}
        if self.include_stimulus:
            first = start + self.state_window_trs - self.lag_trs - (self.stimulus_context_trs - 1)
            stop = first + self.stimulus_length
            assert first >= 0
            result["stimulus_frame_indices"] = torch.arange(first, stop)
            for name, path in record["modalities"].items():
                result["stimuli"][name] = torch.from_numpy(np.array(self._open(path)[first:stop], copy=True))
        return result


def add_partial(ds, cfg):
    """Add negative starts down to the last 30 s inside the run, with valid recent frames and supported movie rows."""
    stride = window_settings(cfg, "train")[1]
    ranges = {r["run_id"]: frame_ranges(cfg, "train", r["run_id"]) for r in ds.records}
    least = frames_within(30.0, ds.tr_seconds)
    fill = ds.history_fill_frames
    masks = {r["run_id"]: r["masks"] for r in read_jsonl(ds.paths.runs_jsonl)}
    extra = []
    for index, record in enumerate(ds.records):
        valid = np.load(record["valid"])
        n = valid.shape[0]
        support = np.load(ds._array_path(masks[record["run_id"]]["stimulus_support"], (n,), np.bool_))
        inside = np.zeros(n, dtype=bool)
        for a, b in ranges.get(record["run_id"]) or [(0, n)]:
            inside[a:b] = True
        first0 = ds.state_window_trs - ds.lag_trs - (ds.stimulus_context_trs - 1)
        for start in range(-stride, least - ds.state_window_trs - 1, -stride):
            first = start + first0
            recent = np.arange(max(0, start + fill), start + ds.length)
            if first < 0 or not support[first:first + ds.stimulus_length].all() or not (valid[recent] & inside[recent]).all() \
                    or not inside[first:first + ds.stimulus_length].all():
                continue
            extra.append((index, start))
    full = len(ds.windows)
    ds.__class__ = PartialWindows
    ds.windows = np.concatenate([ds.windows, np.asarray(extra, dtype=np.int64).reshape(-1, 2)])
    return {"full windows": full, "partial windows": len(extra), "least frames inside the run": least, "history fill frames": fill}


@torch.no_grad()
def fit_matrix(cfg, out, *, penalty=1e-3, movie_factors=(100.0,), state_factor=10.0, partial=True, rollmix=None, device="cuda"):
    """Fit and save the matrix for each movie factor; returns one report per file."""
    from accelerate.utils import set_seed
    device = torch.device(device)
    set_seed(cfg["seed"])
    steps = cfg["train"]["rollout_steps"]
    train_data, val_data = build_dataset(cfg, "train"), build_dataset(cfg, "val")
    report = {"windows": add_partial(train_data, cfg)} if partial else {}
    model = build_model(cfg, train_data.subject_ids, initialize_stimulus=True)
    load_encoder_init(model, cfg)
    if model.stimulus_scale is not None:
        model.stimulus_scale.fit([train_data._open(record["modalities"]["context"]) for record in train_data.records])
    model.to(device).eval()
    p = model.predictor

    def windows(batch):
        """[B, steps + 1, D] states and [B, steps, C] movie rows as the predictor reads them (row t: the step t -> t + 1)."""
        states = model.encode(batch["brain"].to(device))[:, :steps + 1]
        rows = model.encode_stimulus({k: v.to(device) for k, v in batch["stimuli"].items()})[:, :steps]
        return states, p.condition_norm(rows)

    def features(states, rows):
        return torch.cat((states, rows, torch.ones(*states.shape[:-1], 1, device=device)), -1).double()

    start = None
    if rollmix:
        weights = torch.load(rollmix, map_location="cpu", weights_only=False)["model"]
        start = torch.cat((weights["predictor.linear.weight"].T, weights["predictor.condition.weight"].T,
                           weights["predictor.condition.bias"][None])).to(device).double()
    gram, cross, n = 0, 0, 0
    rgram, rcross, rn = 0, 0, 0
    for batch in build_loader(train_data, cfg, shuffle=False, batch_size=512):
        states, rows = windows(batch)
        X = features(states[:, :-1], rows).flatten(0, 1)
        gram, cross, n = gram + X.T @ X, cross + X.T @ (states[:, 1:] - states[:, :-1]).flatten(0, 1).double(), n + len(X)
        if start is not None:
            x = states[:, 0].double()
            for t in range(steps):
                Xt = features(x.float(), rows[:, t])
                if t:  # a predicted state, paired with the real next one
                    rgram, rcross, rn = rgram + Xt.T @ Xt, rcross + Xt.T @ (states[:, t + 1].double() - x), rn + len(Xt)
                x = x + Xt @ start
    D, F = states.shape[-1], gram.shape[0]
    report.update({"transitions": n, "state": D, "features": F, "standardized rows": model.stimulus_scale is not None})
    if start is not None:
        gram, cross = gram + rgram * (n / rn), cross + rcross * (n / rn)
        report["rollout transitions"] = rn
    val = [windows(batch) for batch in build_loader(val_data, cfg, shuffle=False, batch_size=512)]

    def objective(coef):
        """Mean squared latent error on the val split, teacher-forced and over the rollout."""
        tf, roll, total = 0.0, 0.0, 0
        for states, rows in val:
            tf += float(((states[:, :-1].double() + features(states[:, :-1], rows) @ coef - states[:, 1:]) ** 2).sum())
            x, error = states[:, 0].double(), 0.0
            for t in range(steps):
                x = x + features(x.float(), rows[:, t]) @ coef
                error += float(((x - states[:, t + 1]) ** 2).sum())
            roll += error
            total += states.shape[0] * steps * D
        return tf / total, roll / total

    report["identity (teacher-forced, rollout)"] = objective(torch.zeros(F, D, device=device, dtype=torch.float64))
    raw_rows = model.encode_stimulus({k: v.to(device) for k, v in next(iter(build_loader(val_data, cfg, shuffle=False, batch_size=512)))["stimuli"].items()})[:, :steps]
    out = Path(out)
    reports = {}
    for factor in movie_factors:
        path = out if len(movie_factors) == 1 else out.with_name(f"{out.stem}-m{factor:g}{out.suffix}")
        mine = dict(report, **{"movie factor": factor, "state factor": state_factor, "penalty": penalty})
        scaled = gram.diagonal().mean() * torch.eye(F, device=device, dtype=torch.float64)
        scaled[-1, -1] = 0
        scaled[D:-1, D:-1] *= factor
        scaled[:D, :D] *= state_factor
        coef = torch.linalg.solve(gram + penalty * scaled, cross)
        mine["fit (teacher-forced, rollout)"] = [round(x, 5) for x in objective(coef)]
        mine["movie part's norm"], mine["state part's norm"] = float(coef[D:-1].norm()), float(coef[:D].norm())
        coef = coef.float()
        A = coef[:D]
        top = torch.linalg.eigvals((torch.eye(D, device=device) + A).double().cpu()).abs().sort(descending=True).values
        mine.update(largest_eigenvalues=[round(float(x), 4) for x in top[:5]], above_one=int((top > 1).sum()))
        weights = {"predictor.linear.weight": A.T, "predictor.condition.weight": coef[D:-1].T, "predictor.condition.bias": coef[-1]}
        if model.stimulus_scale is not None:
            weights.update({"stimulus_scale.mean": model.stimulus_scale.mean, "stimulus_scale.scale": model.stimulus_scale.scale})
        else:
            weights.update({"predictor.condition_norm.weight": p.condition_norm.weight, "predictor.condition_norm.bias": p.condition_norm.bias})
        torch.save({"format": FORMAT, "model": {k: v.detach().cpu() for k, v in weights.items()}}, path)
        # The file, read back into the model, must take the same steps as the fit.
        missing, extra = model.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=False)
        assert not extra and not [k for k in missing if k.startswith(("predictor.", "stimulus_scale."))], (missing, extra)
        states, rows = val[0]
        mine["largest difference from the fit"] = float((p(states[:, :-1], raw_rows) - (states[:, :-1].double() + features(states[:, :-1], rows) @ coef.double())).abs().max())
        reports[str(path)] = mine
        print(json.dumps({str(path): mine}), flush=True)
    return reports
