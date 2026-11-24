from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import nibabel as nib
import numpy as np
import pytest

from state_of_mind.nbs_data.fmri._adapters import (
    AdapterError,
    MissingSourceAssetError,
    SourceShapeError,
    prepare_run,
)
from state_of_mind.nbs_data.fmri._adapters import _result_from_fmri
from state_of_mind.nbs_data.fmri._sources import NativeRunSource


AFFINE = np.eye(4, dtype=np.float64)


def _atlas(shape: tuple[int, int, int] = (1, 1, 1000)) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.arange(1, 1001, dtype=np.int16).reshape(shape), AFFINE)


def _volume(n_time: int = 40) -> nib.Nifti1Image:
    # Smooth, non-flat temporal signals keep standardized DVARS below the
    # default threshold while still exercising spatial smoothing/regression.
    t = np.arange(n_time, dtype=np.float32)[None, None, None, :]
    voxel = np.arange(1000, dtype=np.float32)[None, None, :, None]
    values = 0.1 * t + 0.001 * voxel + 0.02 * np.sin(t / 4.0 + voxel / 100.0)
    return nib.Nifti1Image(values.astype(np.float32), AFFINE)


def _source(
    dataset_id: str,
    fmri: Path,
    *,
    n_time: int,
    root: Path,
    **kwargs: object,
) -> NativeRunSource:
    source_tr = kwargs.pop("source_tr", None)
    return NativeRunSource(
        dataset_id=dataset_id,
        run_id=f"{dataset_id}/synthetic/run-01",
        timeline_id=f"{dataset_id}/synthetic",
        subject_id="synthetic/01",
        tr_seconds=(
            {"algonauts": 1.49, "hcp_movies": 1.0}.get(dataset_id, 1.0)
            if source_tr is None
            else float(source_tr)
        ),
        n_volumes=n_time,
        fmri_path=fmri,
        source_root=root,
        source_space="synthetic",
        **kwargs,
    )


def test_algonauts_adapter_is_exact_float32_identity_and_exposes_masks(tmp_path: Path) -> None:
    path = tmp_path / "official.h5"
    expected = (np.arange(12_000, dtype=np.float32).reshape(12, 1000) / 100.0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("friends_s01e01a", data=expected)
    source = _source(
        "algonauts",
        path,
        n_time=12,
        root=tmp_path,
        hdf5_key="friends_s01e01a",
    )

    prepared = prepare_run(source)

    np.testing.assert_array_equal(prepared.brain, expected)
    assert prepared.brain.dtype == np.float32
    assert prepared.valid_time.all()
    assert prepared.masks["nonsteady"].sum() == 0
    assert prepared.masks["motion_qc"].all()
    assert prepared.provenance["processing"]["re_normalization"] is False
    source_record = prepared.provenance["source"]["files"][0]
    assert source_record["path"] == str(path.resolve())
    assert source_record["size_bytes"] == path.stat().st_size
    assert source_record["mtime_ns"] == path.stat().st_mtime_ns
    assert source_record["shape"] == [12, 1000]
    assert source_record["dtype"] == "float32"
    assert "sha256" not in source_record


def test_algonauts_rejects_nonfinite_or_wrong_time_shape(tmp_path: Path) -> None:
    path = tmp_path / "official.h5"
    with h5py.File(path, "w") as handle:
        bad = np.zeros((3, 999), dtype=np.float32)
        handle.create_dataset("friends_s01e01a", data=bad)
    source = _source("algonauts", path, n_time=3, root=tmp_path, hdf5_key="friends_s01e01a")
    with pytest.raises(SourceShapeError, match="shape"):
        prepare_run(source)


def test_algonauts_adapter_requires_exact_float32_dtype(tmp_path: Path) -> None:
    path = tmp_path / "official.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("friends_s01e01a", data=np.zeros((3, 1000), dtype=np.float64))
    source = _source("algonauts", path, n_time=3, root=tmp_path, hdf5_key="friends_s01e01a")

    with pytest.raises(SourceShapeError, match="dtype float32"):
        prepare_run(source)


def test_algonauts_adapter_binds_hdf5_key_to_canonical_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "official.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ses-001_task-s01e01a", data=np.zeros((3, 1000), dtype=np.float32))
    source = NativeRunSource(
        dataset_id="algonauts",
        run_id="Algonauts2025/01/task-friends/s01e02a/run-00",
        timeline_id="sub-01:friends:s01e02a",
        subject_id="Algonauts2025/01",
        tr_seconds=1.49,
        n_volumes=3,
        fmri_path=path,
        source_root=tmp_path,
        source_space="official_Schaefer18_1000Par7Net_HDF5",
        hdf5_key="ses-001_task-s01e01a",
    )

    with pytest.raises(SourceShapeError, match="identity"):
        prepare_run(source)


def test_hcp_adapter_uses_exact_motion_and_the_minimal_recipe(tmp_path: Path) -> None:
    n_time = 40
    fmri_path = tmp_path / "movie.nii.gz"
    mask_path = tmp_path / "brainmask_fs.1.60.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    motion_path = tmp_path / "Movement_Regressors.txt"
    nib.save(_volume(n_time), fmri_path)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1000), dtype=np.uint8), AFFINE), mask_path)
    nib.save(_atlas(), atlas_path)
    motion = np.zeros((n_time, 12), dtype=np.float64)
    motion[10, 0] = 1.0
    np.savetxt(motion_path, motion, fmt="%.8f")
    source = _source(
        "hcp_movies",
        fmri_path,
        n_time=n_time,
        root=tmp_path,
        brain_mask=mask_path,
        motion_regressors=motion_path,
    )

    prepared = prepare_run(source, atlas_path)

    assert prepared.brain.shape == (n_time, 1000)
    processing = prepared.provenance["processing"]
    assert processing["recipe"] == "minimal_tribev2" and processing["confound_regression"] is False
    assert processing["framewise_displacement"] == "Power_FD_degrees_to_radians_radius_50mm"
    assert processing["nonsteady"] == "flags_unavailable_all_false_storage_mask"
    assert processing["nonsteady_flags_available"] is False
    assert not prepared.masks["motion_qc"][10]
    assert not prepared.valid_time[10]
    # The supplied 12 columns are used directly; the large derivative block
    # is not recomputed by the adapter.
    assert prepared.qc["framewise_displacement"][10] == pytest.approx(1.0)


def test_hcp_adapter_excludes_nonfinite_motion_rows_with_finite_storage(tmp_path: Path) -> None:
    n_time = 40
    fmri_path = tmp_path / "movie.nii.gz"
    mask_path = tmp_path / "brainmask_fs.1.60.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    motion_path = tmp_path / "Movement_Regressors.txt"
    nib.save(_volume(n_time), fmri_path)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1000), dtype=np.uint8), AFFINE), mask_path)
    nib.save(_atlas(), atlas_path)
    motion = np.zeros((n_time, 12), dtype=np.float64)
    motion[10, 0] = np.nan
    np.savetxt(motion_path, motion, fmt="%.8f")
    source = _source(
        "hcp_movies",
        fmri_path,
        n_time=n_time,
        root=tmp_path,
        brain_mask=mask_path,
        motion_regressors=motion_path,
    )

    prepared = prepare_run(source, atlas_path)

    assert not prepared.valid_time[10]
    assert np.isfinite(prepared.brain).all()


def test_volume_adapters_fail_closed_on_missing_assets_and_grid_mismatch(tmp_path: Path) -> None:
    n_time = 40
    fmri_path = tmp_path / "bold.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(_volume(n_time), fmri_path)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1000), dtype=np.uint8), AFFINE), mask_path)
    source = _source(
        "hcp_movies",
        fmri_path,
        n_time=n_time,
        root=tmp_path,
        brain_mask=mask_path,
        motion_regressors=tmp_path / "missing.txt",
    )
    with pytest.raises(MissingSourceAssetError, match="missing"):
        prepare_run(source, _atlas())

    with pytest.raises(AdapterError, match="affine|shape"):
        # A complete atlas with a different affine must not be silently used.
        bad_atlas = nib.Nifti1Image(np.arange(1, 1001, dtype=np.int16).reshape(1, 1, 1000), np.diag([2, 1, 1, 1]))
        # Reuse valid source assets just far enough to reach the grid check.
        motion_path = tmp_path / "Movement_Regressors.txt"
        np.savetxt(motion_path, np.zeros((n_time, 12)), fmt="%.8f")
        complete = _source("hcp_movies", fmri_path, n_time=n_time, root=tmp_path, brain_mask=mask_path, motion_regressors=motion_path)
        prepare_run(complete, bad_atlas)


def test_support_intersects_fmri_validity_and_qc_summary_is_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "official.h5"
    expected = np.ones((8, 1000), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("friends_s01e01a", data=expected)
    source = _source("algonauts", path, n_time=8, root=tmp_path, hdf5_key="friends_s01e01a")
    support = np.asarray([True, True, False, True, False, True, True, True], dtype=np.bool_)

    prepared = prepare_run(source, stimulus_support=support)

    assert set(prepared.masks) == {"valid_time", "nonsteady", "motion_qc", "stimulus_support"}
    np.testing.assert_array_equal(prepared.valid_time, support)
    summary = prepared.qc_summary()
    json.dumps(summary, allow_nan=False, sort_keys=True)
    assert summary["counts"]["valid_time"] == int(support.sum())
    assert summary["counts"]["fMRI_valid"] == 8


@pytest.mark.parametrize(
    ("dataset_id", "wrong_tr"),
    (("algonauts", 1.0), ("hcp_movies", 1.75)),
)
def test_public_adapters_reject_wrong_declared_dataset_clock(
    tmp_path: Path, dataset_id: str, wrong_tr: float
) -> None:
    source = _source(
        dataset_id,
        tmp_path / "missing-source",
        n_time=4,
        root=tmp_path,
        source_tr=wrong_tr,
        hdf5_key="friends_s01e01a" if dataset_id == "algonauts" else None,
    )
    atlas = _atlas() if dataset_id != "algonauts" else None
    with pytest.raises(SourceShapeError, match="TR"):
        prepare_run(source, atlas)


def test_atlas_resampling_preserves_original_path_and_records_decision(tmp_path: Path) -> None:
    from state_of_mind.nbs_data.fmri._adapters import _atlas_resampling_record, _resolve_atlas

    reference = nib.Nifti1Image(np.zeros((1, 1, 1000, 4), dtype=np.float32), AFFINE)
    atlas_path = tmp_path / "atlas_different_grid.nii.gz"
    source_labels = np.repeat(np.arange(1, 1001, dtype=np.int16), 2).reshape(1, 1, 2000)
    nib.save(nib.Nifti1Image(source_labels, np.diag([1.0, 1.0, 0.5, 1.0])), atlas_path)

    resolved, original_path, resampled = _resolve_atlas(
        atlas_path, reference=reference, allow_resample=True
    )

    assert resolved.shape == (1, 1, 1000)
    assert original_path == atlas_path.resolve()
    assert resampled is True
    record = _atlas_resampling_record(
        atlas_path, reference, atlas_path=original_path, performed=resampled
    )
    assert record["performed"] is True
    assert record["order"] == 0
    assert record["interpolation"] == "nearest_neighbor"
    assert record["original_atlas_path"] == str(original_path)
    assert record["source_grid"]["shape"] == [1, 1, 2000]
    assert record["target_fmri_grid"]["shape"] == [1, 1, 1000]
    assert np.unique(resolved.get_fdata()).size == 1000


def test_adapter_preserves_authoritative_kernel_valid_time_native(tmp_path: Path) -> None:
    source = _source(
        "algonauts",
        tmp_path / "unused.h5",
        n_time=3,
        root=tmp_path,
        hdf5_key="friends_s01e01a",
    )
    result = SimpleNamespace(
        parcels=np.zeros((3, 1000), dtype=np.float32),
        valid_time_native=np.array([True, False, True], dtype=np.bool_),
        finite_time_native=np.ones(3, dtype=np.bool_),
        nonsteady_native=np.zeros(3, dtype=np.bool_),
        motion_qc_native=np.ones(3, dtype=np.bool_),
        framewise_displacement=np.zeros(3, dtype=np.float32),
        standardized_dvars=np.zeros(3, dtype=np.float32),
        provenance={},
    )

    prepared = _result_from_fmri(
        source,
        result,
        provenance={},
        stimulus_support=np.ones(3, dtype=np.bool_),
    )

    np.testing.assert_array_equal(
        prepared.valid_time, result.valid_time_native
    )
    assert prepared.provenance["valid_time"]["formula"] == (
        "authoritative kernel valid_time_native"
    )
