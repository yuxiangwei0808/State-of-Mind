from __future__ import annotations

import json

import numpy as np
import pytest

from neuroworld.data.schaefer_v2_dataset import build_schaefer_v2_smoke_loader
from neuroworld.data.bundle import BundleValidationError
from neuroworld.nbs_data.bundle import (
    BundlePaths,
    BundleWriter,
    select_valid_time,
    validate_bundle,
)


def _masks(n_timepoints: int) -> dict[str, np.ndarray]:
    return {
        "valid_time": np.ones(n_timepoints, dtype=bool),
        "nonsteady": np.zeros(n_timepoints, dtype=bool),
        "motion_qc": np.ones(n_timepoints, dtype=bool),
        "stimulus_support": np.ones(n_timepoints, dtype=bool),
    }


def test_v2_bundle_round_trip(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    writer.write_run(
        run_id="sub-01/run-1",
        subject_id="hcp/100610",
        timeline_id="movie-1",
        tr_seconds=1.0,
        brain=np.zeros((3, 1000), dtype=np.float32),
        masks=_masks(3),
        source={"nifti": "source.nii.gz"},
        preprocessing={"normalization": "voxel_temporal_sample_zscore"},
        qc={"n_valid": 3},
    )
    writer.write_features(
        timeline_id="movie-1",
        tr_seconds=1.0,
        arrays={
            "video": np.zeros((3, 2, 1408), dtype=np.float32),
            "audio": np.zeros((3, 2, 1024), dtype=np.float32),
        },
        source={"sample_rate_hz": 2.0},
        pooling={"rule": "sample_center_in_half_open_tr_bin"},
    )
    manifest = writer.finalize(expected_run_count=1, provenance={"test": True})

    assert manifest["schema_version"] == 2
    assert manifest["observation_space"] == "schaefer1000"
    loaded = validate_bundle(tmp_path / "stage")
    assert loaded["run_count"] == 1
    assert loaded["feature_timeline_count"] == 1


def test_v2_bundle_rejects_nested_hashes_computed_marker(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    writer.write_run(
        run_id="run",
        subject_id="hcp/100610",
        timeline_id="movie",
        tr_seconds=1.0,
        brain=np.zeros((1, 1000), dtype=np.float32),
        masks=_masks(1),
        source={},
        preprocessing={},
    )
    writer.write_features(
        timeline_id="movie",
        tr_seconds=1.0,
        arrays={
            "video": np.zeros((1, 2, 1408), dtype=np.float32),
            "audio": np.zeros((1, 2, 1024), dtype=np.float32),
        },
        source={},
        pooling={},
    )
    writer.finalize(expected_run_count=1, provenance={})
    manifest_path = tmp_path / "stage" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["nested"] = {"hashes_computed": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="hashes_computed"):
        validate_bundle(tmp_path / "stage")


def test_v2_bundle_rejects_wrong_brain_shape(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    with pytest.raises(BundleValidationError, match="shape"):
        writer.write_run(
            run_id="run",
            subject_id="hcp/100610",
            tr_seconds=1.0,
            brain=np.zeros((2, 20484), dtype=np.float32),
            masks=_masks(2),
            source={},
            preprocessing={},
        )


def test_v2_bundle_rejects_wrong_dataset_clock(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    with pytest.raises(BundleValidationError, match="TR"):
        writer.write_run(
            run_id="run",
            subject_id="hcp/100610",
            tr_seconds=1.75,
            brain=np.zeros((2, 1000), dtype=np.float32),
            masks=_masks(2),
            source={},
            preprocessing={},
        )


def test_v2_bundle_detects_tampered_array(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "algonauts")
    writer.write_run(
        run_id="run",
        subject_id="algonauts/01",
        tr_seconds=1.49,
        brain=np.zeros((2, 1000), dtype=np.float32),
        masks=_masks(2),
        source={},
        preprocessing={"adapter": "already_parcellated_identity"},
    )
    writer.write_features(
        timeline_id="run",
        tr_seconds=1.49,
        arrays={
            "video": np.zeros((2, 2, 1408), dtype=np.float32),
            "audio": np.zeros((2, 2, 1024), dtype=np.float32),
            "text": np.zeros((2, 2, 3072), dtype=np.float32),
        },
        source={},
        pooling={"rule": "sample_center_in_half_open_tr_bin"},
    )
    writer.finalize(expected_run_count=1, provenance={})
    paths = BundlePaths.from_root(tmp_path / "stage")
    array = np.load(paths.fmri_run("run"), allow_pickle=False)
    array[0, 0] = 1.0
    np.save(paths.fmri_run("run"), array, allow_pickle=False)
    with pytest.raises(BundleValidationError, match="mtime|size"):
        validate_bundle(tmp_path / "stage")


def test_v2_bundle_rejects_invalid_mask_semantics(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    masks = _masks(2)
    masks["nonsteady"][0] = True
    with pytest.raises(BundleValidationError, match="nonsteady"):
        writer.write_run(
            run_id="run",
            subject_id="hcp/100610",
            tr_seconds=1.0,
            brain=np.zeros((2, 1000), dtype=np.float32),
            masks=masks,
            source={},
            preprocessing={},
        )


def test_v2_bundle_requires_aligned_feature_timeline(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    writer.write_run(
        run_id="run",
        subject_id="hcp/100610",
        timeline_id="movie",
        tr_seconds=1.0,
        brain=np.zeros((2, 1000), dtype=np.float32),
        masks=_masks(2),
        source={},
        preprocessing={},
    )
    writer.write_features(
        timeline_id="movie",
        tr_seconds=1.0,
        arrays={
            "video": np.zeros((3, 2, 1408), dtype=np.float32),
            "audio": np.zeros((3, 2, 1024), dtype=np.float32),
        },
        source={},
        pooling={"rule": "sample_center_in_half_open_tr_bin"},
    )
    with pytest.raises(BundleValidationError, match="length/clock"):
        writer.finalize(expected_run_count=1, provenance={})


def test_v2_bundle_rejects_nonboolean_masks_and_valid_time(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    masks = _masks(2)
    masks["valid_time"] = np.array([1, 0], dtype=np.uint8)
    with pytest.raises(BundleValidationError, match="boolean"):
        writer.write_run(
            run_id="run",
            subject_id="hcp/100610",
            tr_seconds=1.0,
            brain=np.zeros((2, 1000), dtype=np.float32),
            masks=masks,
            source={},
            preprocessing={},
        )

    with pytest.raises(BundleValidationError, match="boolean"):
        select_valid_time(
            np.zeros((2, 1), dtype=np.float32),
            np.array([1, 0], dtype=np.uint8),
        )


def test_v2_smoke_loader_always_applies_valid_time(tmp_path) -> None:
    writer = BundleWriter(tmp_path / "stage", "hcp_movies")
    brain = np.zeros((3, 1000), dtype=np.float32)
    brain[0, 0] = 1.0
    brain[1] = 999.0
    brain[2, 0] = 3.0
    masks = _masks(3)
    masks["valid_time"] = np.array([True, False, True], dtype=bool)
    writer.write_run(
        run_id="sub-01/run-1",
        subject_id="hcp/100610",
        timeline_id="movie-1",
        tr_seconds=1.0,
        brain=brain,
        masks=masks,
        source={},
        preprocessing={},
    )
    writer.write_features(
        timeline_id="movie-1",
        tr_seconds=1.0,
        arrays={
            "video": np.zeros((3, 2, 1408), dtype=np.float32),
            "audio": np.zeros((3, 2, 1024), dtype=np.float32),
        },
        source={},
        pooling={"rule": "sample_center_in_half_open_tr_bin"},
    )
    writer.finalize(expected_run_count=1, provenance={})

    batch = next(iter(build_schaefer_v2_smoke_loader(tmp_path / "stage")))

    assert batch["y"].shape == (1, 2, 1000)
    assert not np.any(batch["y"].numpy() == 999.0)
    np.testing.assert_array_equal(
        batch["physical_time_s"].numpy(),
        np.array([[0.0, 2.0]], dtype=np.float64),
    )
    assert batch["run_id"] == ["sub-01/run-1"]
