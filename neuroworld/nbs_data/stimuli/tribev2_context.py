"""Contextual stimulus features from TRIBEv2's frozen stimulus model (offline preparation).

Runs TRIBEv2's per-modality projections, its 8-block temporal Transformer and its low-rank head
(the 2,048 values per sample that its subject-specific brain head reads) on a bundle's raw 2-Hz
feature cache, with attention masked to past and present samples, then pools the output to the
fMRI TR grid. The runtime reads one row per TR as the stimulus condition when
``model.stimulus_features: tribev2_context``.

The network is models/tribev2.py::TRIBEv2. Adaptations: causal attention instead of the trained
bidirectional attention, so no row sees future stimulus; inference in overlapping 298-sample
windows, where each sample keeps the output of the window that gives it the most past context.

    python -m neuroworld.nbs_data.stimuli.tribev2_context --bundle <bundle> \
        --checkpoint <TRIBEv2 best.ckpt> --output <bundle>-tribev2-context
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from neuroworld.models.tribev2 import CONTEXT_DIM, TRIBEv2
from neuroworld.nbs_data.stimuli.raw_cache import read_chunk_index
from neuroworld.nbs_data.stimuli.timing import pool_modality_to_bold, stitch_modality_chunks

FORMAT = "neuroworld-stimulus-context-v1"
VARIANT = "tribev2_context_causal"
WINDOW_SAMPLES, STRIDE_SAMPLES = 298, 200


@torch.no_grad()
def contextual_rows(model, features, *, window=WINDOW_SAMPLES, stride=STRIDE_SAMPLES):
    """features: modality -> [T, 2, D] numpy at 2 Hz. Returns [T, context_dim] float32.

    With causal attention each sample takes the earliest window that contains it, which gives it the
    most past context; every window after the first contributes its last ``stride`` samples.
    """
    total = next(iter(features.values())).shape[0]
    starts = list(range(0, max(total - window, 0) + 1, stride))
    if starts[-1] + window < total:
        starts.append(total - window)
    out = None
    done = 0
    for start in starts:
        stop = min(start + window, total)
        rows = model({m: torch.from_numpy(np.ascontiguousarray(v[start:stop]))[None] for m, v in features.items()})[0]
        if out is None:
            out = np.zeros((total, rows.shape[-1]), dtype=np.float32)
        out[done:stop] = rows[done - start:].numpy()
        done = stop
    return out


def pool_to_trs(values, timestamps, *, n_bins, tr_seconds, origin_seconds=0.0):
    """Mean of the samples whose centre falls in each half-open TR bin (the bundle's pooling rule)."""
    index = np.floor((np.asarray(timestamps) - origin_seconds) / tr_seconds).astype(np.int64)
    inside = (index >= 0) & (index < n_bins)
    sums = np.zeros((n_bins, values.shape[1]), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sums, index[inside], values[inside])
    np.add.at(counts, index[inside], 1)
    filled = counts > 0
    sums[filled] /= counts[filled, None]
    return sums.astype(np.float32)


def stimulus_key(timeline_id):
    """Timeline shared by all subjects; Algonauts timelines carry a per-subject ``sub-XX:`` prefix."""
    return re.sub(r"^sub-[^:]+:", "", str(timeline_id))


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stimulus_groups(runs):
    """stimulus key -> one run record per distinct timeline that shows it."""
    groups = defaultdict(list)
    for run in runs:
        if run["timeline_id"] not in [r["timeline_id"] for r in groups[stimulus_key(run["timeline_id"])]]:
            groups[stimulus_key(run["timeline_id"])].append(run)
    return groups


def raw_chunks(bundle, raw_index=None):
    """timeline -> modality -> chunk records of the bundle's raw 2-Hz feature cache."""
    raw_index = Path(raw_index) if raw_index else Path(f"{bundle}-preparation") / "raw" / "chunks.jsonl"
    chunks = defaultdict(lambda: defaultdict(list))
    for record in read_chunk_index(raw_index):
        chunks[record.timeline_id][record.modality].append(record)
    return chunks, raw_index


def stitched_features(chunks):
    """modality -> chunk records of one timeline, stitched on the video's 2-Hz grid.

    Returns ({modality: stitched values with .values [T, 2, D] and .timestamps}, grid timestamps).
    """
    stitched = {m: stitch_modality_chunks(m, [{"values": c.values, "timestamps": c.timestamps, "support": c.support,
                                               "source_id": c.source_id}
                                              for c in (record.load() for record in records)])
                for m, records in chunks.items()}
    grid = stitched["video"].timestamps
    for m, values in stitched.items():
        if len(values.timestamps) != len(grid) or not np.allclose(values.timestamps, grid, rtol=0, atol=1e-6):
            raise ValueError(f"{m} is not on the video's 2-Hz grid")
    return stitched, grid


def prepare(bundle, checkpoint, output, *, raw_index=None, limit=0, log=print):
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle / "bundle.json").read_text())
    runs = _read_jsonl(bundle / "runs" / "runs.jsonl")
    features = {r["timeline_id"]: r for r in _read_jsonl(bundle / manifest["artifacts"]["features"]["path"])}
    groups = stimulus_groups(runs)
    chunks, raw_index = raw_chunks(bundle, raw_index)
    model = TRIBEv2(causal=True).load_tribev2(checkpoint).eval()
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    timelines = {}
    names = sorted(groups)[:limit] if limit else sorted(groups)
    started = time.time()
    for number, key in enumerate(names, start=1):
        members = groups[key]
        first = members[0]
        n_bins, tr, origin = first["n_timepoints"], first["tr_seconds"], first["time_origin_seconds"]
        stitched, grid = stitched_features(chunks[first["timeline_id"]])
        # Every subject's bundle features must equal this stitched stimulus pooled to TRs, which checks
        # both the alignment and that the subjects in the group saw the same stimulus.
        for run in members:
            if (run["n_timepoints"], run["tr_seconds"], run["time_origin_seconds"]) != (n_bins, tr, origin):
                raise ValueError(f"{run['timeline_id']}: timing differs from {first['timeline_id']}")
            for m, values in stitched.items():
                pooled = pool_modality_to_bold(m, values, origin_seconds=origin, native_tr_seconds=tr, n_bins=n_bins)
                stored = np.load(bundle / features[run["timeline_id"]]["modalities"][m]["path"], mmap_mode="r")
                gap = float(np.abs(pooled.values - stored).max())
                if gap > 1e-3:
                    raise ValueError(f"{run['timeline_id']} {m}: pooled raw cache differs from the bundle by {gap:.3g}")
        relative = f"arrays/{re.sub(r'[^A-Za-z0-9._-]+', '_', key)}.npy"
        target = output / relative
        if not target.exists():
            rows = contextual_rows(model, {m: v.values for m, v in stitched.items()})
            pooled = pool_to_trs(rows, grid, n_bins=n_bins, tr_seconds=tr, origin_seconds=origin)
            partial = target.with_suffix(".partial.npy")
            np.save(partial, pooled)
            os.replace(partial, target)
        for run in members:
            timelines[run["timeline_id"]] = {"path": relative, "n_timepoints": n_bins}
        log(f"[{number}/{len(names)}] {key}: {len(grid)} samples -> {n_bins} TRs "
            f"({len(members)} timelines); {(time.time() - started) / 60:.1f} min")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    index = {"format": FORMAT, "variant": VARIANT, "dim": CONTEXT_DIM, "dataset_id": manifest["dataset_id"],
             "bundle": str(bundle.resolve()), "raw_index": str(raw_index.resolve()),
             "checkpoint": str(checkpoint_path), "checkpoint_size_bytes": checkpoint_path.stat().st_size,
             "window_samples": WINDOW_SAMPLES, "stride_samples": STRIDE_SAMPLES, "attention": "causal",
             "pooling": "sample_center_half_open_arithmetic_mean", "timelines": timelines}
    partial = output / "index.partial.json"
    partial.write_text(json.dumps(index, indent=1) + "\n")
    os.replace(partial, output / "index.json")
    return output / "index.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Prepared Schaefer-1000 v2 bundle")
    parser.add_argument("--checkpoint", required=True, help="TRIBEv2 checkpoint (best.ckpt)")
    parser.add_argument("--output", required=True, help="Directory for index.json and arrays/")
    parser.add_argument("--raw-index", help="Raw 2-Hz cache index; default <bundle>-preparation/raw/chunks.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N stimuli (for a quick check)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    index = prepare(args.bundle, args.checkpoint, args.output, raw_index=args.raw_index, limit=args.limit,
                    log=lambda text: print(text, flush=True))
    print(json.dumps({"index": str(index)}), flush=True)


if __name__ == "__main__":
    main()
