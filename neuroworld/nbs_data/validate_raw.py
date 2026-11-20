#!/usr/bin/env python3
"""Stream-validate every array referenced by an Algonauts Stage 0 index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np

from neuroworld.nbs_data.stimuli.materialize import discover_source_timeline_ids
from neuroworld.nbs_data.stimuli.raw_cache import ChunkIndexRecord, read_chunk_index
from neuroworld.nbs_data.stimuli.timing import MODALITY_DIMS, RAW_FREQUENCY_HZ


EXPECTED_MODALITIES = ("audio", "text", "video")
EXPECTED_DTYPES = {
    "values": "float32",
    "timestamps": "float64",
    "support": "bool",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-timelines", type=int, default=1_392)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Spawn this many whole-timeline payload validators (default: 4).",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_timeline_payload(
    task: tuple[str, tuple[tuple[int, dict[str, ChunkIndexRecord]], ...]],
) -> tuple[str, dict[str, dict[str, int]], int]:
    """Fully load and validate all segments for one timeline."""

    timeline_id, segments = task
    expected_modality_set = set(EXPECTED_MODALITIES)
    per_modality: dict[str, Counter[str]] = {
        modality: Counter() for modality in EXPECTED_MODALITIES
    }
    previous_stop: dict[str, float] = {}
    total_artifact_bytes = 0

    for chunk_index, segment_records in segments:
        segment_key = (timeline_id, chunk_index)
        _require(
            set(segment_records) == expected_modality_set,
            f"segment {segment_key} does not contain all modalities",
        )
        expected_samples: int | None = None
        reference_timestamps: np.ndarray | None = None
        for modality in EXPECTED_MODALITIES:
            record = segment_records[modality]
            chunk = record.load()
            _require(
                chunk.values.dtype == np.dtype("float32"),
                f"bad values dtype: {record.identity}",
            )
            _require(
                chunk.timestamps.dtype == np.dtype("float64"),
                f"bad timestamps dtype: {record.identity}",
            )
            _require(
                chunk.support.dtype == np.dtype("bool"),
                f"bad support dtype: {record.identity}",
            )
            _require(
                math.isclose(
                    chunk.sample_rate_hz,
                    RAW_FREQUENCY_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
                f"bad sample rate: {record.identity}",
            )

            if expected_samples is None:
                expected_samples = int(chunk.values.shape[0])
                reference_timestamps = chunk.timestamps.copy()
            else:
                _require(
                    int(chunk.values.shape[0]) == expected_samples,
                    f"segment sample counts disagree: {segment_key}",
                )
                _require(
                    np.array_equal(chunk.timestamps, reference_timestamps),
                    f"segment timestamps disagree: {segment_key}",
                )

            prior = previous_stop.get(modality)
            if prior is not None:
                _require(
                    math.isclose(
                        float(chunk.timestamps[0]) - prior,
                        1.0 / RAW_FREQUENCY_HZ,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ),
                    f"non-contiguous timestamps before {record.identity}",
                )
            previous_stop[modality] = float(chunk.timestamps[-1])

            summary = per_modality[modality]
            summary["records"] += 1
            summary["samples"] += int(chunk.values.shape[0])
            summary["supported_samples"] += int(chunk.support.sum())
            for array_name in EXPECTED_DTYPES:
                size = int(getattr(record, array_name)["size_bytes"])
                summary[f"{array_name}_bytes"] += size
                total_artifact_bytes += size
            del chunk
        del reference_timestamps

    return (
        timeline_id,
        {modality: dict(counts) for modality, counts in per_modality.items()},
        total_artifact_bytes,
    )


def _validated_worker_count(value: int, *, timeline_count: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise RuntimeError("workers must be a positive integer")
    return min(int(value), timeline_count)


def validate(
    index_path: Path,
    source_root: Path,
    *,
    expected_timeline_count: int,
    workers: int = 1,
) -> dict[str, Any]:
    expected_timelines = set(
        discover_source_timeline_ids(source_root, dataset="algonauts")
    )
    _require(
        len(expected_timelines) == expected_timeline_count,
        f"source inventory has {len(expected_timelines)} timelines, "
        f"expected {expected_timeline_count}",
    )

    records = read_chunk_index(index_path)
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    modality_indices: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    indexed_timelines: set[str] = set()

    for record in records:
        indexed_timelines.add(record.timeline_id)
        key = (record.timeline_id, record.chunk_index)
        _require(
            record.modality not in grouped[key],
            f"duplicate modality in segment {key}: {record.modality}",
        )
        grouped[key][record.modality] = record
        modality_indices[record.timeline_id][record.modality].add(record.chunk_index)
        _require(
            record.shape[2] == MODALITY_DIMS[record.modality],
            f"wrong feature dimension for {record.identity}: {record.shape}",
        )
        for array_name, expected_dtype in EXPECTED_DTYPES.items():
            descriptor = getattr(record, array_name)
            _require(
                descriptor["dtype"] == expected_dtype,
                f"{record.identity} {array_name} dtype is "
                f"{descriptor['dtype']!r}, expected {expected_dtype!r}",
            )

    _require(
        indexed_timelines == expected_timelines,
        "index/source timeline mismatch: "
        f"missing={sorted(expected_timelines - indexed_timelines)[:10]}, "
        f"extra={sorted(indexed_timelines - expected_timelines)[:10]}",
    )

    expected_modality_set = set(EXPECTED_MODALITIES)
    for timeline_id in sorted(expected_timelines):
        found_modalities = set(modality_indices[timeline_id])
        _require(
            found_modalities == expected_modality_set,
            f"{timeline_id} modalities are {sorted(found_modalities)}, "
            f"expected {list(EXPECTED_MODALITIES)}",
        )
        reference_indices = modality_indices[timeline_id][EXPECTED_MODALITIES[0]]
        _require(
            reference_indices == set(range(len(reference_indices))),
            f"{timeline_id} chunk indices are not contiguous from zero: "
            f"{sorted(reference_indices)}",
        )
        for modality in EXPECTED_MODALITIES[1:]:
            _require(
                modality_indices[timeline_id][modality] == reference_indices,
                f"{timeline_id} chunk indices disagree across modalities",
            )

    per_modality: dict[str, Counter[str]] = {
        modality: Counter() for modality in EXPECTED_MODALITIES
    }
    total_artifact_bytes = 0
    worker_count = _validated_worker_count(
        workers,
        timeline_count=len(expected_timelines),
    )
    tasks = tuple(
        (
            timeline_id,
            tuple(
                (chunk_index, grouped[(timeline_id, chunk_index)])
                for chunk_index in sorted(
                    modality_indices[timeline_id][EXPECTED_MODALITIES[0]]
                )
            ),
        )
        for timeline_id in sorted(expected_timelines)
    )

    def collect(results: Any) -> None:
        nonlocal total_artifact_bytes
        for completed, (_, summaries, artifact_bytes) in enumerate(results, start=1):
            total_artifact_bytes += artifact_bytes
            for modality, counts in summaries.items():
                per_modality[modality].update(counts)
            if completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[validation] loaded {completed}/{len(tasks)} timelines",
                    file=sys.stderr,
                    flush=True,
                )

    # Loading a record performs descriptor/stat checks, disables pickle, reads
    # the complete payload, checks finite numeric values, and applies the raw
    # shape/timing/support contract. Each worker owns whole timelines, so
    # temporal continuity is checked locally while only one large values array
    # per worker is live.
    if worker_count == 1:
        collect(map(_validate_timeline_payload, tasks))
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=worker_count) as pool:
            collect(
                pool.imap_unordered(
                    _validate_timeline_payload,
                    tasks,
                    chunksize=1,
                )
            )


    return {
        "ready": True,
        "index": str(index_path.resolve()),
        "record_count": len(records),
        "timeline_count": len(indexed_timelines),
        "segment_count": len(grouped),
        "payload_workers": worker_count,
        "modalities": {
            modality: dict(sorted(counts.items()))
            for modality, counts in sorted(per_modality.items())
        },
        "total_artifact_bytes": total_artifact_bytes,
        "checks": {
            "source_inventory_exact": True,
            "timeline_modalities_complete": True,
            "chunk_indices_aligned_and_contiguous": True,
            "descriptor_stats_and_dtypes_valid": True,
            "all_payloads_loaded_without_pickle": True,
            "all_numeric_payloads_finite": True,
            "shapes_and_feature_dimensions_valid": True,
            "timestamps_regular_contiguous_and_cross_modal_aligned": True,
            "support_masks_boolean_and_aligned": True,
        },
    }


def main() -> int:
    args = _parse_args()
    report = validate(
        args.index,
        args.source_root,
        expected_timeline_count=args.expected_timelines,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
