from __future__ import annotations

import numpy as np
import pytest

from state_of_mind.nbs_data.fmri.common import (
    AtlasValidationError,
    GridValidationError,
    build_valid_time_mask,
    fill_invalid_rows_linear,
    power_framewise_displacement,
    process_volume_minimal,
    standardized_dvars,
    validate_nifti_grid,
    validate_same_grid,
    validate_schaefer_labels,
)


def test_hcp_power_fd_converts_degrees_and_ignores_derivative_columns() -> None:
    motion = np.zeros((3, 12), dtype=np.float64)
    motion[1, :6] = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    motion[2, :6] = motion[1, :6]
    # A large change in the supplied derivative block must not affect FD.
    motion[1:, 6:] = 1e6
    observed = power_framewise_displacement(motion)
    expected = 0.6 + 50.0 * np.pi / 180.0 * 0.6
    np.testing.assert_allclose(observed, [0.0, expected, 0.0])


def test_valid_mask_is_strict_about_nan_fd_dvars_and_nonsteady() -> None:
    finite = np.array([True, True, False, True, True, True])
    nonsteady = np.array([False, False, False, True, False, False])
    fd = np.array([0.0, np.nan, 0.1, 0.1, 0.6, 0.1])
    dvars = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.6])
    valid, motion_qc = build_valid_time_mask(
        finite_time=finite,
        nonsteady=nonsteady,
        framewise_displacement=fd,
        standardized_dvars=dvars,
    )
    np.testing.assert_array_equal(motion_qc, [True, False, True, True, False, False])
    np.testing.assert_array_equal(valid, [True, False, False, False, False, False])


def test_invalid_rows_are_linearly_filled_with_zero_boundaries() -> None:
    values = np.array([[10.0], [999.0], [30.0], [999.0], [999.0]])
    valid = np.array([True, False, True, False, False])
    filled = fill_invalid_rows_linear(values, valid)
    np.testing.assert_allclose(filled[:, 0], [10.0, 20.0, 30.0, 0.0, 0.0])
    np.testing.assert_array_equal(filled[valid], values[valid])
    all_invalid = fill_invalid_rows_linear(values, np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(all_invalid, np.zeros_like(values))


def test_standardized_dvars_uses_runwise_voxel_zscores_and_zero_first_row() -> None:
    values = np.array(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64
    )
    observed = standardized_dvars(values)
    expected_step = np.sqrt(np.mean(np.square([1.0 / np.std(values[:, 0]), 2.0 / np.std(values[:, 1])])) / 2.0)
    np.testing.assert_allclose(observed, [0.0, expected_step, expected_step, expected_step])


def test_labels_and_grid_validation_fail_closed() -> None:
    with pytest.raises(AtlasValidationError, match="missing"):
        validate_schaefer_labels(np.array([1, 2, 0]), n_parcels=3)
    with pytest.raises(AtlasValidationError, match="non-integer"):
        validate_schaefer_labels(np.array([1.0, 2.5]), n_parcels=2)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape, observed_affine = validate_nifti_grid((2, 3, 4), affine)
    assert shape == (2, 3, 4)
    np.testing.assert_array_equal(observed_affine, affine)
    with pytest.raises(GridValidationError, match="spatial shape"):
        validate_nifti_grid((2, 3, 5), affine, expected_shape=(2, 3, 4))
    with pytest.raises(GridValidationError, match="affine"):
        validate_same_grid((2, 3, 4), (2, 3, 4), reference_affine=affine, candidate_affine=np.diag([2, 2, 3, 1]))


def test_volume_background_is_not_used_for_finite_mask_or_parcels() -> None:
    rng = np.random.default_rng(18)
    n_time = 6
    shape = (11, 10, 10)
    flat_labels = np.zeros(np.prod(shape), dtype=np.int32)
    flat_labels[100:1100] = np.arange(1, 1001, dtype=np.int32)
    labels = flat_labels.reshape(shape)
    volume = np.zeros(shape + (n_time,), dtype=np.float32)
    flat_volume = volume.reshape(-1, n_time)
    time = np.arange(n_time, dtype=np.float32)
    # Background contains a very large drift and NaNs.  Neither may alter the
    # labeled parcels or invalidate otherwise finite rows.
    flat_volume[:100] = 1e6 * time
    flat_volume[0, 2] = np.nan
    flat_volume[100:1100] = rng.normal(size=(1000, n_time)).astype(np.float32)
    kwargs = dict(
        framewise_displacement=np.zeros(n_time),
        standardized_dvars=np.zeros(n_time),
        time_chunk_size=2,
    )
    observed = process_volume_minimal(volume, labels, **kwargs)
    expected = process_volume_minimal(
        flat_volume[100:1100].reshape(1, 1, 1000, n_time),
        np.arange(1, 1001, dtype=np.int32).reshape(1, 1, 1000),
        **kwargs,
    )
    np.testing.assert_array_equal(observed.parcels, expected.parcels)
    np.testing.assert_array_equal(observed.finite_time_native, np.ones(n_time, dtype=bool))


def test_volume_spatial_mask_intersects_atlas_before_validity_and_pooling() -> None:
    """Masked-out labeled NaNs are ignored without diluting parcel means."""

    rng = np.random.default_rng(181)
    n_time = 6
    # Give every parcel one in-mask and one out-of-mask voxel.  The latter is
    # deliberately pathological so a regression to labels-only selection is
    # caught by both finite validity and the output values.
    labels = np.repeat(np.arange(1, 1001, dtype=np.int32), 2).reshape(1, 1, 2000)
    mask = np.zeros((1, 1, 2000), dtype=np.bool_)
    mask[..., ::2] = True
    volume = np.empty((1, 1, 2000, n_time), dtype=np.float32)
    selected = rng.normal(size=(n_time, 1000)).astype(np.float32)
    volume.reshape(-1, n_time)[::2] = selected.T
    volume.reshape(-1, n_time)[1::2] = np.nan
    kwargs = dict(
        framewise_displacement=np.zeros(n_time),
        standardized_dvars=np.zeros(n_time),
        time_chunk_size=2,
    )

    observed = process_volume_minimal(volume, labels, spatial_mask=mask, **kwargs)
    expected = process_volume_minimal(
        selected.T.reshape(1, 1, 1000, n_time),
        np.arange(1, 1001, dtype=np.int32).reshape(1, 1, 1000),
        **kwargs,
    )

    np.testing.assert_array_equal(observed.parcels, expected.parcels)
    np.testing.assert_array_equal(observed.finite_time_native, np.ones(n_time, dtype=bool))


def test_volume_spatial_mask_rejects_missing_in_mask_schaefer_parcel() -> None:
    labels = np.arange(1, 1001, dtype=np.int32).reshape(1, 1, 1000)
    mask = np.ones((1, 1, 1000), dtype=np.bool_)
    mask[..., 0] = False
    volume = np.zeros((1, 1, 1000, 4), dtype=np.float32)
    with pytest.raises(AtlasValidationError, match="in-mask"):
        process_volume_minimal(
            volume,
            labels,
            spatial_mask=mask,
            framewise_displacement=np.zeros(4),
            standardized_dvars=np.zeros(4),
        )


def test_proxy_volume_is_sliced_in_time_chunks_without_full_materialization() -> None:
    class TrackingProxy:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape
            self.requests: list[tuple[int | None, int | None]] = []

        def __getitem__(self, key):
            time_key = key[-1]
            if isinstance(time_key, slice):
                self.requests.append((time_key.start, time_key.stop))
                assert (time_key.stop or self.shape[-1]) - (time_key.start or 0) <= 2
            else:
                self.requests.append((int(time_key), int(time_key) + 1))
            return self.values[key]

    class ImageLike:
        def __init__(self, proxy: TrackingProxy, affine: np.ndarray) -> None:
            self.dataobj = proxy
            self.shape = proxy.shape
            self.affine = affine

    n_time = 6
    values = np.random.default_rng(19).normal(size=(10, 10, 10, n_time)).astype(np.float32)
    proxy = TrackingProxy(values)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image = ImageLike(proxy, affine)
    labels = np.arange(1, 1001, dtype=np.int32).reshape(10, 10, 10)
    process_volume_minimal(
        image,
        labels,
        framewise_displacement=np.zeros(n_time),
        standardized_dvars=np.zeros(n_time),
        time_chunk_size=2,
    )
    assert proxy.requests
    assert all((stop or n_time) - (start or 0) <= 2 for start, stop in proxy.requests)


def test_negative_fd_and_standardized_dvars_fail_motion_qc() -> None:
    valid, motion_qc = build_valid_time_mask(
        finite_time=np.ones(3, dtype=np.bool_),
        framewise_displacement=np.array([0.0, -1.0, 0.0]),
        standardized_dvars=np.array([0.0, 0.0, -1.0]),
    )

    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_array_equal(motion_qc, [True, False, False])


def test_minimal_recipe_is_parcel_mean_then_zscore_and_detrend() -> None:
    import nibabel as nib
    rng = np.random.default_rng(3)
    n_time = 30
    labels = np.repeat(np.arange(1, 1001, dtype=np.int16), 2).reshape(20, 10, 10)
    volume = rng.normal(size=(20, 10, 10, n_time)).astype(np.float32) + 100.0
    volume[0, 0, 0, :] += np.linspace(0, 50, n_time, dtype=np.float32)  # a drift in one voxel of parcel 1
    fd, dvars = np.zeros(n_time), np.zeros(n_time)
    fd[5] = 1.0  # one motion-flagged frame
    result = process_volume_minimal(nib.Nifti1Image(volume, np.eye(4)), nib.Nifti1Image(labels, np.eye(4)),
                                    framewise_displacement=fd, standardized_dvars=dvars)
    valid = np.ones(n_time, dtype=bool)
    valid[5] = False
    np.testing.assert_array_equal(result.valid_time_native, valid)
    # By hand: the mean of each parcel's voxels, z-scored on valid rows, minus its linear trend.
    rows = np.flatnonzero(valid)
    flat, flat_labels = volume.reshape(-1, n_time), labels.reshape(-1)
    means = np.stack([flat[flat_labels == k].mean(axis=0) for k in (1, 2)], axis=1)[rows].astype(np.float64)
    means = (means - means.mean(0)) / means.std(0, ddof=1)
    trend = np.column_stack([np.ones(rows.size), rows - rows.mean()])
    expected = means - trend @ np.linalg.lstsq(trend, means, rcond=None)[0]
    np.testing.assert_allclose(result.parcels[rows][:, :2], expected, rtol=1e-4, atol=1e-5)
    assert abs(np.corrcoef(rows, result.parcels[rows, 0])[0, 1]) < 1e-6  # the drift is removed
    assert result.provenance["confound_regression"] is False and result.provenance["smoothing"]["applied"] is False
