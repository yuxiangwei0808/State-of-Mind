"""Shared pieces of the linear references (CPU): run arrays and masks per split, stimulus rows, and scoring on saved windows.

Used by neuroworld.benchmark.references (stimulus rows) and eval/tribev2_reference.py (runs, saved windows, scores). Stimulus
rows are the run's own (per-TR TRIBEv2 projections, or the precomputed contextual rows), or brain-free rows: each modality's
frozen extractor features (V-JEPA 2, Wav2Vec-BERT, Llama 3.2; both layer groups) reduced to principal components fitted on
the training stimuli only.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from neuroworld.data.splits import check_splits, content_key, frame_ranges, split_runs
from neuroworld.eval.fmri import FmriAccumulator
from neuroworld.eval.intersubject import stimulus_key

RAW_COMPONENTS = 384  # per modality, as TRIBEv2's projections


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Runs:
    """Brain arrays, masks and stimulus content of the runs of some splits.

    ``valid[split]`` and ``support[split]`` keep only the split's frames (data.split_frames), so a
    run shared by two splits (HCP) is used only within each split's frames.
    """

    def __init__(self, cfg, splits):
        self.bundle = Path(cfg["data"]["bundle_path"])
        records = check_splits(cfg)
        self.runs = {split: split_runs(cfg, split, records) for split in dict.fromkeys(splits)}
        self.records = {r["run_id"]: r for runs in self.runs.values() for r in runs}
        load = lambda entry: np.load(self.bundle / entry["path"])
        self.brain = {i: load(r["brain_views"]["projected_native"]).astype(np.float64) for i, r in self.records.items()}
        valid = {i: load(r["masks"]["valid_time"]).astype(bool) for i, r in self.records.items()}
        support = {i: load(r["masks"]["stimulus_support"]).astype(bool) for i, r in self.records.items()}
        self.valid, self.support = {}, {}
        for split, runs in self.runs.items():
            self.valid[split], self.support[split] = {}, {}
            for r in runs:
                i = r["run_id"]
                allowed = np.ones(len(valid[i]), dtype=bool)
                ranges = frame_ranges(cfg, split, i)
                if ranges:
                    allowed[:] = False
                    for start, stop in ranges:
                        allowed[start:stop] = True
                self.valid[split][i], self.support[split][i] = valid[i] & allowed, support[i] & allowed
        self.content = {i: content_key(r["timeline_id"]) for i, r in self.records.items()}


def _raw_rows(bundle, records, train_keys, components=RAW_COMPONENTS):
    """Brain-free rows per stimulus: each modality's frozen extractor features (both layer groups) reduced
    to ``components`` principal components fitted on the training stimuli only, side by side."""
    manifest = json.loads((bundle / "bundle.json").read_text())
    features = {f["timeline_id"]: f for f in _read_jsonl(bundle / manifest["artifacts"]["features"]["path"])}
    by_key = {}
    for r in records:
        by_key.setdefault(stimulus_key(r["timeline_id"]), features[r["timeline_id"]]["modalities"])
    load = lambda entry: (lambda a: a.reshape(len(a), -1))(np.load(bundle / entry["path"]).astype(np.float64))
    modalities = sorted({m for entries in by_key.values() for m in entries})
    projections = {}
    for modality in modalities:
        total, square, count = 0.0, 0.0, 0
        for key in sorted(train_keys):
            values = load(by_key[key][modality])
            total, square, count = total + values.sum(0), square + values.T @ values, count + len(values)
        mean = total / count
        eigenvalues, vectors = np.linalg.eigh(square / count - np.outer(mean, mean))
        keep = np.argsort(eigenvalues)[::-1][:min(components, len(mean))]
        projections[modality] = (mean, vectors[:, keep])
    return {key: np.concatenate([(load(entries[m]) - projections[m][0]) @ projections[m][1] for m in modalities], axis=1)
            for key, entries in by_key.items()}


def _stimulus_rows(cfg, bundle, records, *, features="run", train_keys=()):
    """One array of stimulus rows per stimulus (shared by the subjects who watched it): the run's own
    (features "run") or brain-free principal components of the extractor features ("raw")."""
    if features == "raw":
        return _raw_rows(bundle, list(records), train_keys)
    out = {}
    if cfg["model"].get("stimulus_features", "projections") == "tribev2_context":
        root = Path(cfg["data"]["context_features"])
        index = json.loads((root / "index.json").read_text())["timelines"]
        for r in records:
            out.setdefault(stimulus_key(r["timeline_id"]), np.load(root / index[r["timeline_id"]]["path"]).astype(np.float64))
        return out
    from neuroworld.models.stimulus import StimulusEncoder
    manifest = json.loads((bundle / "bundle.json").read_text())
    features = {f["timeline_id"]: f for f in _read_jsonl(bundle / manifest["artifacts"]["features"]["path"])}
    encoder = StimulusEncoder(cfg["model"]["tribev2_checkpoint"]).eval()
    with torch.no_grad():
        for r in records:
            key = stimulus_key(r["timeline_id"])
            if key not in out:
                modalities = features[r["timeline_id"]]["modalities"]
                arrays = {m: torch.from_numpy(np.load(bundle / e["path"]))[None] for m, e in modalities.items()}
                out[key] = encoder(arrays)[0].numpy().astype(np.float64)
    return out


def saved_windows(run_dir, folder):
    """run_id -> [windows, horizon] target frame indices saved by evaluation, or None."""
    paths = sorted((Path(run_dir) / folder / "trajectories").glob("batch-*.pt"))
    if not paths:
        return None
    out = defaultdict(list)
    for path in paths:
        batch = torch.load(path, map_location="cpu", weights_only=False)
        for run_id, frames in zip(batch["run_id"], batch["target_frame_indices"].numpy()):
            out[run_id].append(frames)
    return {r: np.stack(f) for r, f in out.items()}


def score_windows(data, windows, predict, steps, identify=False):
    """Scores of predict(run_id, frames, lo, hi) -> [windows, hi - lo, parcels] on saved windows' steps lo..hi
    (``data``: Runs)."""
    lo, hi = steps
    accumulator = FmriAccumulator(hi - lo, identify=identify, content_of=lambda run_id: data.content.get(run_id, run_id))
    for run_id, frames in sorted(windows.items()):
        subject = data.records[run_id]["subject_id"]
        predicted = predict(run_id, frames, lo, hi)
        observed = data.brain[run_id][frames[:, lo:hi]]
        accumulator.add([run_id] * len(frames), [subject] * len(frames),
                        torch.from_numpy(predicted).float(), torch.from_numpy(observed).float())
    summary = accumulator.summary()
    return {k: summary[k] for k in ("parcel_pearson", "global_r", "pattern_corr", "cls_top10") if summary.get(k) is not None}
