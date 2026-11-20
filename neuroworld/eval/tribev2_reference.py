"""Pretrained TRIBEv2 as a reference stimulus model, scored like our models (CPU).

TRIBEv2 runs as released (models/tribev2.py): bidirectional attention within consecutive 100-s
segments of each stimulus's 2-Hz features, its average-subject brain head and its 5-s offset.
Its fsaverage5 prediction is averaged into Schaefer-1000 parcels (atlas labels sampled within
3 mm of nilearn's fsaverage5 mid-thickness surface, as TRIBEv2 projects volumes; TRIBEv2's own
MNI-registered meshes are not public) and pooled to the bundle's TR grid. Predictions are cached
per stimulus in ``<bundle>-tribev2-predictions/``; frames with no stimulus sample are NaN.

Scores, for each split:

``windows``    parcel Pearson, global r, pattern r and Cls@Top10 on exactly a run's saved
               evaluation windows, as for the linear baselines: comparable with the run's models.
``subjects``   per subject, every usable frame of the split (z-scored per run), Pearson per parcel
               averaged over parcels: the Algonauts 2025 score (TRIBEv2 reports 0.2146 on the
               held-out Friends season 7).
``group``      Pearson per parcel between the prediction and the mean of all subjects' fMRI for
               the same stimulus (TRIBEv2's R_group, "near 0.4" on HCP), and each subject's
               correlation with the mean of the others (its group-predictivity).
``alignment``  the subject score with the prediction delayed by -2..+2 TRs; it should peak at 0.

TRIBEv2 trained on Friends seasons 1-6 and Movie10, so on Algonauts these are scores on its own
training data; HCP is outside its training. ``--modalities audio video`` drops text (as missing
for HCP, whose bundle has no transcript), in a separate cache and result file.

    python -m neuroworld.eval.tribev2_reference checkpoints/<dataset>/<run> --splits val dev
    python -m neuroworld.eval.tribev2_reference neuroworld/configs/<dataset>/base.yaml --splits train val \
        --output-dir <dir>   # the data alone, before any run: no window scores
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from neuroworld.data.splits import with_split
from neuroworld.eval.intersubject import _pearson, stimulus_key
from neuroworld.eval.linear_baselines import Runs, saved_windows, score_windows
from neuroworld.models.tribev2 import MODALITY_DIMS, OFFSET_SECONDS, SEGMENT_SAMPLES, TRIBEv2
from neuroworld.paths import DATA

FORMAT = "neuroworld-tribev2-predictions-v1"
ATLAS = str(DATA / "atlases/neuroworld-schaefer1000-v2/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
SHIFTS = (-2, -1, 0, 1, 2)


def parcel_average(atlas_path, n_parcels=1000, radius=3.0):
    """[n_vertices, n_parcels] matrix averaging fsaverage5 vertices into the atlas's parcels.

    Each vertex takes the most frequent nonzero label among the atlas voxels within ``radius`` mm of
    its mid-thickness position (TRIBEv2 projects volumes with a 3-mm ball). Schaefer numbers
    left-hemisphere parcels first; a vertex with no label, or one from the other hemisphere, is left
    out. Returns (matrix, coverage summary).
    """
    import nibabel
    from nilearn import datasets, surface
    atlas = nibabel.load(atlas_path)
    volume = np.asarray(atlas.dataobj).astype(np.int64)
    to_voxel = np.linalg.inv(atlas.affine)
    step = np.arange(-radius, radius + 1e-9, 1.0)
    ball = np.stack(np.meshgrid(step, step, step, indexing="ij"), axis=-1).reshape(-1, 3)
    ball = ball[(ball ** 2).sum(axis=1) <= radius ** 2]
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    labels = []
    for hemi in ("left", "right"):
        pial, white = (surface.load_surf_mesh(fsaverage[f"{kind}_{hemi}"]).coordinates for kind in ("pial", "white"))
        points = ((pial + white) / 2)[:, None, :] + ball[None]  # [vertices, ball, xyz] in mm
        voxels = np.rint(points @ to_voxel[:3, :3].T + to_voxel[:3, 3]).astype(np.int64)
        inside = np.all((voxels >= 0) & (voxels < volume.shape), axis=-1)
        sampled = np.zeros(inside.shape, dtype=np.int64)
        sampled[inside] = volume[tuple(voxels[inside].T)]
        own = (sampled >= 1) & (sampled <= n_parcels // 2) if hemi == "left" else (sampled > n_parcels // 2)
        for row, values in zip(sampled, own):
            kept = row[values]
            labels.append(np.bincount(kept).argmax() if len(kept) else 0)
    labels = np.asarray(labels)
    assigned = np.flatnonzero(labels > 0)
    matrix = np.zeros((len(labels), n_parcels), dtype=np.float64)
    matrix[assigned, labels[assigned] - 1] = 1.0
    counts = matrix.sum(axis=0)
    matrix[:, counts > 0] /= counts[counts > 0]
    return matrix, {"vertices": len(labels), "assigned_vertices": int(len(assigned)),
                    "parcels_with_vertices": int((counts > 0).sum()), "median_vertices_per_parcel": float(np.median(counts))}


@torch.no_grad()
def segment_rows(model, features, segment=SEGMENT_SAMPLES):
    """TRIBEv2's inference: consecutive non-overlapping segments, each seen whole (bidirectional)."""
    total = len(next(iter(features.values())))
    return np.concatenate([
        model({m: torch.from_numpy(np.ascontiguousarray(v[start:start + segment]))[None] for m, v in features.items()})[0].numpy()
        for start in range(0, total, segment)])


def pool_rows(rows, stimulus_times, *, n_bins, tr_seconds, origin_seconds):
    """Mean of the rows predicting each TR bin (the fMRI OFFSET_SECONDS after the stimulus); NaN where none."""
    index = np.floor((np.asarray(stimulus_times) + OFFSET_SECONDS - origin_seconds) / tr_seconds).astype(np.int64)
    inside = (index >= 0) & (index < n_bins)
    sums = np.zeros((n_bins, rows.shape[1]))
    counts = np.zeros(n_bins)
    np.add.at(sums, index[inside], rows[inside])
    np.add.at(counts, index[inside], 1)
    pooled = np.full_like(sums, np.nan)
    pooled[counts > 0] = sums[counts > 0] / counts[counts > 0, None]
    return pooled


def prepare(cfg, splits, output, *, checkpoint, atlas=ATLAS, raw_index=None, modalities=tuple(MODALITY_DIMS),
            log=print):
    """Cache TRIBEv2's parcel predictions for every stimulus of the splits from the given modalities
    (others are missing, as TRIBEv2 treats them); returns the index path."""
    from neuroworld.data.splits import split_runs
    from neuroworld.nbs_data.stimuli.tribev2_context import raw_chunks, stimulus_groups, stitched_features
    output, bundle = Path(output), Path(cfg["data"]["bundle_path"])
    runs = [r for split in splits for r in split_runs(cfg, split)]
    groups = stimulus_groups(runs)
    index_path = output / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {
        "format": FORMAT, "dataset_id": json.loads((bundle / "bundle.json").read_text())["dataset_id"],
        "checkpoint": str(Path(checkpoint).expanduser().resolve()), "atlas": str(atlas), "head": "average subject",
        "offset_seconds": OFFSET_SECONDS, "segment_samples": SEGMENT_SAMPLES, "modalities": list(modalities),
        "timelines": {}}
    if index.get("modalities", list(MODALITY_DIMS)) != list(modalities):
        raise ValueError(f"{output} holds predictions from {index['modalities']}, not {list(modalities)}")
    missing = {key: members for key, members in groups.items()
               if any(m["timeline_id"] not in index["timelines"] for m in members)}
    if not missing:
        return index_path
    chunks, _ = raw_chunks(bundle, raw_index)
    model = TRIBEv2(brain_head=True).load_tribev2(checkpoint).eval()
    average, coverage = parcel_average(atlas)
    index["parcels"] = coverage
    # The head and the vertex average are both linear: fold them into one map to parcels.
    weight = model.head.weight.detach().double().numpy().T @ average
    bias = model.head.bias.detach().double().numpy() @ average
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    started = time.time()
    for number, (key, members) in enumerate(sorted(missing.items()), start=1):
        first = members[0]
        timing = (first["n_timepoints"], first["tr_seconds"], first["time_origin_seconds"])
        if any((m["n_timepoints"], m["tr_seconds"], m["time_origin_seconds"]) != timing for m in members):
            raise ValueError(f"{key}: timelines of this stimulus differ in timing")
        relative = f"arrays/{re.sub(r'[^A-Za-z0-9._-]+', '_', key)}.npy"
        stitched, grid = stitched_features(chunks[first["timeline_id"]])
        rows = segment_rows(model, {m: v.values for m, v in stitched.items() if m in modalities})
        absent = sorted(set(modalities) - set(stitched))
        if absent:  # e.g. HCP has no transcript: TRIBEv2 treats text as missing
            index["absent_from_data"] = sorted(set(index.get("absent_from_data", [])) | set(absent))
        pooled = pool_rows(rows.astype(np.float64), grid, n_bins=timing[0], tr_seconds=timing[1], origin_seconds=timing[2])
        partial = output / f"{relative}.partial.npy"
        np.save(partial, (pooled @ weight + bias).astype(np.float32))
        os.replace(partial, output / relative)
        for member in members:
            index["timelines"][member["timeline_id"]] = {"path": relative, "n_timepoints": timing[0]}
        log(f"[{number}/{len(missing)}] {key}: {len(grid)} samples -> {timing[0]} TRs "
            f"({(time.time() - started) / 60:.1f} min)")
    partial = output / "index.partial.json"
    partial.write_text(json.dumps(index, indent=1) + "\n")
    os.replace(partial, index_path)
    return index_path


class Predictions:
    """Cached parcel predictions, by timeline."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.index = json.loads((self.root / "index.json").read_text())
        self._arrays = {}

    def __getitem__(self, timeline_id):
        path = self.index["timelines"][timeline_id]["path"]
        if path not in self._arrays:
            self._arrays[path] = np.load(self.root / path).astype(np.float64)
        return self._arrays[path]


def _zscore(values):
    values = values - values.mean(axis=0)
    return values / np.maximum(values.std(axis=0), 1e-12)


def _parcel_r(predicted, observed):
    return float(np.nanmean(_pearson(predicted, observed)))


def _usable(runs, split, predictions, run_id, shift=0):
    """Frames of the split with valid fMRI, stimulus support and a prediction delayed by ``shift`` TRs."""
    predicted = predictions[runs.records[run_id]["timeline_id"]]
    t = np.flatnonzero(runs.valid[split][run_id] & runs.support[split][run_id])
    t = t[(t - shift >= 0) & (t - shift < len(predicted))]
    return t[np.isfinite(predicted[t - shift]).all(axis=1)], predicted


def subject_scores(runs, split, predictions, shift=0):
    """Per subject: Pearson per parcel over all usable frames of the split (z-scored per run), averaged."""
    by_subject = defaultdict(lambda: ([], []))
    for run_id, record in runs.records.items():
        if run_id not in runs.valid[split]:
            continue
        t, predicted = _usable(runs, split, predictions, run_id, shift)
        if len(t) < 3:
            continue
        by_subject[record["subject_id"]][0].append(_zscore(predicted[t - shift]))
        by_subject[record["subject_id"]][1].append(_zscore(runs.brain[run_id][t]))
    scores = {subject: _parcel_r(np.concatenate(p), np.concatenate(o)) for subject, (p, o) in sorted(by_subject.items())}
    values = np.array(list(scores.values()))
    return {"mean": float(values.mean()), "sd": float(values.std()), "subjects": len(values), "by_subject": scores}


def group_scores(runs, split, predictions):
    """R_group: prediction against the mean of all subjects' z-scored fMRI for the same stimulus, and each
    subject's correlation with the mean of the others. Frames need at least two subjects."""
    stimuli = defaultdict(list)
    for run_id in runs.valid[split]:
        stimuli[stimulus_key(runs.records[run_id]["timeline_id"])].append(run_id)
    group_p, group_o, own = [], [], defaultdict(lambda: ([], []))
    for members in stimuli.values():
        timeline = runs.records[members[0]]["timeline_id"]
        predicted = predictions[timeline]
        total, count, frames = np.zeros_like(predicted), np.zeros(len(predicted)), {}
        for run_id in members:
            t, _ = _usable(runs, split, predictions, run_id)
            if len(t) >= 3:
                frames[run_id] = (t, _zscore(runs.brain[run_id][t]))
                total[t] += frames[run_id][1]
                count[t] += 1
        shared = np.flatnonzero(count >= 2)
        if len(shared) < 3:
            continue
        group_p.append(_zscore(predicted[shared]))
        group_o.append(_zscore(total[shared] / count[shared, None]))
        for run_id, (t, values) in frames.items():
            t_shared = count[t] >= 2
            others = (total[t[t_shared]] - values[t_shared]) / (count[t[t_shared]] - 1)[:, None]
            own[runs.records[run_id]["subject_id"]][0].append(_zscore(values[t_shared]))
            own[runs.records[run_id]["subject_id"]][1].append(_zscore(others))
    if not group_p:
        return None
    predictivity = np.array([_parcel_r(np.concatenate(v), np.concatenate(o)) for v, o in own.values()])
    return {"r_group": _parcel_r(np.concatenate(group_p), np.concatenate(group_o)),
            "subject_group_predictivity_median": float(np.median(predictivity)),
            "subject_group_predictivity_mean": float(predictivity.mean()), "subjects": len(predictivity),
            "stimuli": len(group_p)}


def run(target, *, splits=("val",), output_dir=None, predictions_dir=None, checkpoint=None, atlas=ATLAS,
        raw_index=None, modalities=tuple(MODALITY_DIMS), log=print):
    """Prepare (cached) and score TRIBEv2 on each split of a run directory (with its saved windows) or a
    config YAML (the data alone); writes <output>/tribev2[-<modalities>][-<split>].json, by default in
    <run>/baselines."""
    from neuroworld.eval.rollout import output_folder
    from neuroworld.utils.config import load_config
    target = Path(target)
    run_dir = target if target.is_dir() else None
    cfg = load_config(target / "config.yaml" if run_dir else target)
    if output_dir is None and run_dir is None:
        raise ValueError("A config YAML has no run directory: pass output_dir")
    output = Path(output_dir) if output_dir else run_dir / "baselines"
    for split in splits:
        cfg = with_split(cfg, split)[0]
    modalities = tuple(m for m in MODALITY_DIMS if m in modalities)
    tag = "" if modalities == tuple(MODALITY_DIMS) else "-" + "-".join(modalities)
    predictions_dir = Path(predictions_dir or f"{cfg['data']['bundle_path']}-tribev2-predictions{tag}")
    prepare(cfg, splits, predictions_dir, checkpoint=checkpoint or cfg["model"]["tribev2_checkpoint"],
            atlas=atlas, raw_index=raw_index, modalities=modalities, log=log)
    predictions = Predictions(predictions_dir)
    runs = Runs(cfg, splits)
    results = {}
    for split in splits:
        entry = {"source": str(target), "split": split, "predictions": str(predictions_dir), "modalities": list(modalities),
                 "absent_from_data": predictions.index.get("absent_from_data", []),
                 "parcels": predictions.index.get("parcels"),
                 "subjects": subject_scores(runs, split, predictions),
                 "group": group_scores(runs, split, predictions),
                 "alignment": {str(s): subject_scores(runs, split, predictions, shift=s)["mean"] for s in SHIFTS}}
        if run_dir is not None and split != "train":
            for name, horizons in (("windows", False), ("windows_horizons", True)):
                windows = saved_windows(run_dir, output_folder(split, horizons))
                if windows is not None:
                    def predict(run_id, frames, lo, hi):
                        values = predictions[runs.records[run_id]["timeline_id"]][frames[:, lo:hi]]
                        return np.nan_to_num(values, nan=0.0)
                    width = windows[next(iter(windows))].shape[1]
                    entry[name] = {str(limit): score_windows(runs, windows, predict, (0, limit), identify=True)
                                   for limit in (20, 50, 100) if limit <= width} if horizons else \
                        score_windows(runs, windows, predict, (0, width), identify=True)
        results[split] = entry
        log(f"{split}: {json.dumps({k: v for k, v in entry.items() if k not in ('source', 'predictions')})}")
        output.mkdir(parents=True, exist_ok=True)
        (output / (f"tribev2{tag}.json" if split == "val" else f"tribev2{tag}-{split}.json")).write_text(
            json.dumps(entry, indent=1) + "\n")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="A run directory (its config's splits; its saved windows are scored too) "
                                       "or a config YAML (the data alone)")
    parser.add_argument("--splits", nargs="+", default=["val"], choices=("train", "val", "dev", "test"),
                        help="Splits to score (default val); train has no saved windows, only the whole-split scores")
    parser.add_argument("--output-dir", help="Where to write the scores; default <run>/baselines (required for a YAML)")
    parser.add_argument("--predictions", help="Prediction cache; default <bundle>-tribev2-predictions")
    parser.add_argument("--checkpoint", help="TRIBEv2 checkpoint; default the run's model.tribev2_checkpoint")
    parser.add_argument("--atlas", default=ATLAS, help="Volumetric Schaefer-1000 atlas in MNI space")
    parser.add_argument("--raw-index", help="Raw 2-Hz feature cache index; default <bundle>-preparation/raw/chunks.jsonl")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITY_DIMS), choices=list(MODALITY_DIMS),
                        help="Stimulus modalities given to TRIBEv2 (default all; the others count as missing)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    run(args.target, splits=args.splits, output_dir=args.output_dir, predictions_dir=args.predictions, checkpoint=args.checkpoint,
        atlas=args.atlas, raw_index=args.raw_index, modalities=args.modalities, log=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
