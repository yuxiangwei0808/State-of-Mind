"""Regression tests for the full Algonauts Stage 0 array validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import neuroworld.nbs_data.validate_raw as validator
from neuroworld.nbs_data.stimuli.raw_cache import export_raw_chunks
from neuroworld.nbs_data.stimuli.timing import MODALITY_DIMS


def _chunks(timeline_ids: tuple[str, ...]):
    for timeline_offset, timeline_id in enumerate(timeline_ids):
        for chunk_index in range(2):
            timestamps = np.array(
                [chunk_index + 0.25, chunk_index + 0.75],
                dtype=np.float64,
            )
            for modality in validator.EXPECTED_MODALITIES:
                yield {
                    "timeline_id": timeline_id,
                    "modality": modality,
                    "values": np.full(
                        (2, 2, MODALITY_DIMS[modality]),
                        timeline_offset + chunk_index + 1,
                        dtype=np.float32,
                    ),
                    "timestamps": timestamps,
                    "support": np.array([True, False], dtype=np.bool_),
                    "chunk_index": chunk_index,
                }


def test_parallel_validator_matches_serial_full_payload_audit(
    tmp_path: Path, monkeypatch
) -> None:
    timeline_ids = ("timeline-a", "timeline-b", "timeline-c")
    index = export_raw_chunks(_chunks(timeline_ids), tmp_path / "raw")
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(
        validator,
        "discover_source_timeline_ids",
        lambda source_root, *, dataset: timeline_ids,
    )

    serial = validator.validate(
        index,
        source_root,
        expected_timeline_count=len(timeline_ids),
        workers=1,
    )
    parallel = validator.validate(
        index,
        source_root,
        expected_timeline_count=len(timeline_ids),
        workers=2,
    )

    assert serial.pop("payload_workers") == 1
    assert parallel.pop("payload_workers") == 2
    assert serial == parallel
    assert parallel["ready"] is True
    assert parallel["record_count"] == 18
    assert all(parallel["checks"].values())


def test_validator_rejects_nonpositive_workers() -> None:
    with pytest.raises(RuntimeError, match="positive integer"):
        validator._validated_worker_count(0, timeline_count=3)
