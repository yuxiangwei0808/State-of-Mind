"""Native-TR fMRI cleaning and Schaefer-1000 projection kernels.

The preparation commands use this module as the numerical boundary between a
source derivative and a bundle.  It intentionally contains no dataset
discovery or command-line code.  Array kernels accept ordinary NumPy arrays so
that the numerical contract can be tested without a neuroimaging stack;
optional NIfTI helpers import :mod:`nibabel` only when called.

The canonical orientation for matrix functions is ``[time, voxel]``.  NIfTI
images are conventionally ``[x, y, z, time]`` and are converted to that
orientation by :func:`process_volume_minimal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


N_SCHAEFER_PARCELS = 1000
FD_THRESHOLD_MM = 0.5
STD_DVARS_THRESHOLD = 1.5
POWER_FD_RADIUS_MM = 50.0


class FmriError(ValueError):
    """Base error for an invalid fMRI input or processing contract."""


class GridValidationError(FmriError):
    """Raised when NIfTI spatial grids do not agree."""


class AtlasValidationError(FmriError):
    """Raised when a Schaefer label image is not a complete integer atlas."""


class ConfoundValidationError(FmriError):
    """Raised when confounds cannot be turned into a finite design matrix."""


@dataclass(frozen=True)
class FmriResult:
    """Cleaned native-clock fMRI output.

    ``parcels`` contains valid Schaefer rows and interpolation/zero fillers at
    invalid rows.  ``valid_time_native`` is authoritative; consumers must not
    infer validity from the finite filler values.  ``voxel_z`` is optional to
    keep production memory usage low and is populated only when requested.
    """

    parcels: np.ndarray
    valid_time_native: np.ndarray
    finite_time_native: np.ndarray
    nonsteady_native: np.ndarray
    motion_qc_native: np.ndarray
    framewise_displacement: np.ndarray
    standardized_dvars: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)
    voxel_z: np.ndarray | None = None

    def __post_init__(self) -> None:
        parcels = np.asarray(self.parcels)
        if parcels.ndim != 2 or parcels.shape[1] != N_SCHAEFER_PARCELS:
            raise FmriError(
                "parcels must have shape [time, 1000], got "
                f"{parcels.shape}"
            )
        if parcels.dtype != np.float32 or not parcels.flags.c_contiguous:
            raise FmriError("parcels must be C-contiguous float32")
        if not np.isfinite(parcels).all():
            raise FmriError("parcels contain non-finite values")
        n_time = parcels.shape[0]
        for name in (
            "valid_time_native",
            "finite_time_native",
            "nonsteady_native",
            "motion_qc_native",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (n_time,) or value.dtype != np.bool_:
                raise FmriError(f"{name} must be a boolean vector of length {n_time}")
        for name in ("framewise_displacement", "standardized_dvars"):
            value = np.asarray(getattr(self, name))
            if value.shape != (n_time,):
                raise FmriError(f"{name} must be a vector of length {n_time}")
        if self.voxel_z is not None:
            voxel_z = np.asarray(self.voxel_z)
            if voxel_z.ndim != 2 or voxel_z.shape[0] != n_time:
                raise FmriError("voxel_z must have shape [time, voxel]")
            if voxel_z.dtype != np.float32 or not voxel_z.flags.c_contiguous:
                raise FmriError("voxel_z must be C-contiguous float32")
            # Invalid rows intentionally remain NaN in this diagnostic view;
            # only the stored parcel view receives interpolation fillers.
            if not np.isfinite(voxel_z[np.asarray(self.valid_time_native)]).all():
                raise FmriError("valid voxel_z rows contain non-finite values")

        object.__setattr__(self, "parcels", np.ascontiguousarray(parcels, dtype=np.float32))
        for name in (
            "valid_time_native",
            "finite_time_native",
            "nonsteady_native",
            "motion_qc_native",
        ):
            object.__setattr__(
                self,
                name,
                np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.bool_)),
            )
        for name in ("framewise_displacement", "standardized_dvars"):
            object.__setattr__(
                self,
                name,
                np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float32)),
            )
        if self.voxel_z is not None:
            object.__setattr__(
                self, "voxel_z", np.ascontiguousarray(np.asarray(self.voxel_z, dtype=np.float32))
            )

    # Short aliases make the object convenient in callers while retaining the
    # explicit native-clock names in the on-disk schema.
    @property
    def valid_time(self) -> np.ndarray:
        return self.valid_time_native

    @property
    def projected_native(self) -> np.ndarray:
        """Bundle-facing alias for the Schaefer parcel view."""

        return self.parcels


def _as_shape_affine(
    image_or_shape: Any,
    affine: np.ndarray | None = None,
    *,
    name: str = "image",
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return shape/affine from an image-like object or explicit arguments."""

    if hasattr(image_or_shape, "shape") and not isinstance(
        image_or_shape, (tuple, list, np.ndarray)
    ):
        shape = tuple(int(v) for v in image_or_shape.shape)
        if affine is None:
            affine = getattr(image_or_shape, "affine", None)
    else:
        try:
            shape = tuple(int(v) for v in image_or_shape)
        except (TypeError, ValueError) as exc:
            raise GridValidationError(f"{name} shape is not a sequence") from exc
    if len(shape) not in (3, 4) or any(v <= 0 for v in shape):
        raise GridValidationError(f"{name} shape must be positive 3-D or 4-D, got {shape}")
    if affine is None:
        raise GridValidationError(f"{name} affine is required for grid validation")
    affine_array = np.asarray(affine, dtype=np.float64)
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        raise GridValidationError(f"{name} affine must be a finite 4x4 matrix")
    if not np.isfinite(np.linalg.det(affine_array)) or abs(np.linalg.det(affine_array)) <= 1e-12:
        raise GridValidationError(f"{name} affine is singular")
    return shape, affine_array


def validate_nifti_grid(
    image_or_shape: Any,
    affine: np.ndarray | None = None,
    *,
    expected_shape: Sequence[int] | None = None,
    expected_affine: np.ndarray | None = None,
    name: str = "image",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Validate an image's spatial shape and affine and return both.

    A 4-D input is accepted, but only its first three dimensions participate
    in comparison.  The affine must be finite and nonsingular; silently
    accepting a dimension-only match is unsafe because left/right orientation
    can otherwise be reversed.
    """

    shape, affine_array = _as_shape_affine(image_or_shape, affine, name=name)
    spatial_shape = shape[:3]
    if expected_shape is not None:
        expected = tuple(int(v) for v in expected_shape)
        if len(expected) == 4:
            expected = expected[:3]
        if len(expected) != 3 or any(v <= 0 for v in expected):
            raise GridValidationError(f"expected_shape must be a positive 3-D shape, got {expected}")
        if spatial_shape != expected:
            raise GridValidationError(
                f"{name} spatial shape {spatial_shape} does not match expected {expected}"
            )
    if expected_affine is not None:
        expected_array = np.asarray(expected_affine, dtype=np.float64)
        if expected_array.shape != (4, 4) or not np.isfinite(expected_array).all():
            raise GridValidationError("expected_affine must be a finite 4x4 matrix")
        if not np.allclose(affine_array, expected_array, atol=atol, rtol=rtol):
            raise GridValidationError(f"{name} affine differs from the expected grid")
    return spatial_shape, affine_array


def validate_same_grid(
    reference: Any,
    candidate: Any,
    *,
    reference_affine: np.ndarray | None = None,
    candidate_affine: np.ndarray | None = None,
    names: tuple[str, str] = ("reference", "candidate"),
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Require equal spatial dimensions and affine between two images."""

    ref_shape, ref_affine = validate_nifti_grid(
        reference, reference_affine, name=names[0], atol=atol, rtol=rtol
    )
    cand_shape, cand_affine = validate_nifti_grid(
        candidate, candidate_affine, name=names[1], atol=atol, rtol=rtol
    )
    if ref_shape != cand_shape:
        raise GridValidationError(
            f"{names[0]} spatial shape {ref_shape} differs from {names[1]} {cand_shape}"
        )
    if not np.allclose(ref_affine, cand_affine, atol=atol, rtol=rtol):
        raise GridValidationError(f"{names[0]} and {names[1]} affines differ")
    return ref_shape, ref_affine


def validate_schaefer_labels(
    labels: np.ndarray,
    *,
    n_parcels: int = N_SCHAEFER_PARCELS,
    allow_background: bool = True,
) -> np.ndarray:
    """Validate integer Schaefer labels and return a flattened int32 vector.

    Labels are one-based (``1..n_parcels``).  ``0`` is accepted only as an
    explicit background label and is excluded from parcel means.  Every parcel
    must be represented at least once; missing labels are a hard error rather
    than an implicitly re-numbered atlas.
    """

    if not isinstance(n_parcels, (int, np.integer)) or int(n_parcels) <= 0:
        raise AtlasValidationError("n_parcels must be a positive integer")
    raw = np.asarray(labels)
    if raw.ndim not in (1, 3):
        raise AtlasValidationError(
            f"labels must be a 1-D vector or 3-D image array, got {raw.shape}"
        )
    if raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise AtlasValidationError("labels must be a non-empty numeric array")
    if not np.isfinite(raw).all():
        raise AtlasValidationError("labels contain non-finite values")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise AtlasValidationError("labels contain non-integer values")
    values = np.ascontiguousarray(rounded.reshape(-1), dtype=np.int32)
    minimum = 0 if allow_background else 1
    if int(values.min()) < minimum or int(values.max()) > int(n_parcels):
        raise AtlasValidationError(
            f"labels must lie in {minimum}..{int(n_parcels)}, observed "
            f"{int(values.min())}..{int(values.max())}"
        )
    observed = np.unique(values)
    expected = np.arange(1, int(n_parcels) + 1, dtype=np.int32)
    missing = np.setdiff1d(expected, observed)
    if missing.size:
        preview = missing[:10].tolist()
        suffix = "..." if missing.size > 10 else ""
        raise AtlasValidationError(
            f"Schaefer atlas is missing {missing.size} parcel labels: {preview}{suffix}"
        )
    if not allow_background and np.any(values == 0):
        raise AtlasValidationError("background label 0 is not allowed for this atlas")
    return values


def _coerce_array(value: Any, *, name: str) -> np.ndarray:
    """Load an ndarray or image-like object without importing nibabel."""

    if hasattr(value, "dataobj"):
        try:
            value = np.asanyarray(value.dataobj)
        except Exception as exc:  # pragma: no cover - proxy-specific
            raise FmriError(f"cannot read {name} data proxy: {exc}") from exc
    elif hasattr(value, "get_fdata"):
        try:
            value = np.asanyarray(value.get_fdata(dtype=np.float32))
        except TypeError:  # image-like test doubles may not accept dtype
            value = np.asanyarray(value.get_fdata())
        except Exception as exc:  # pragma: no cover - proxy-specific
            raise FmriError(f"cannot read {name} data: {exc}") from exc
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise FmriError(f"{name} must be numeric")
    return array


def load_nifti(path: str | Path, *, mmap: bool = True) -> Any:
    """Load a NIfTI image with a lazy nibabel import."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FmriError("NIfTI I/O requires nibabel") from exc
    try:
        return nib.load(str(Path(path).expanduser()), mmap=mmap)
    except Exception as exc:
        raise FmriError(f"cannot load NIfTI image {path}: {exc}") from exc


def _check_vector(value: np.ndarray | Sequence[float] | None, n_time: int, *, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n_time, dtype=np.float64)
    array = np.asarray(value)
    if array.shape != (n_time,):
        raise ConfoundValidationError(f"{name} must have shape {(n_time,)}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ConfoundValidationError(f"{name} must be numeric")
    return np.asarray(array, dtype=np.float64)


def _coerce_boolean_mask(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Accept boolean or finite binary image masks and return bool storage."""

    array = _coerce_array(value, name=name) if not isinstance(value, np.ndarray) else np.asarray(value)
    if array.shape != shape:
        raise FmriError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array, dtype=np.bool_)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise FmriError(f"{name} must be a finite binary mask")
    if not np.all((array == 0) | (array == 1)):
        raise FmriError(f"{name} must contain only 0/1 values")
    return np.ascontiguousarray(array.astype(np.bool_), dtype=np.bool_)


def power_framewise_displacement(
    motion: np.ndarray | Sequence[Sequence[float]],
    *,
    rotation_in_degrees: bool = True,
    radius_mm: float = POWER_FD_RADIUS_MM,
) -> np.ndarray:
    """Compute Power FD from six motion parameters.

    The first three parameters are translations in millimetres and the last
    three are rotations.  HCP stores rotations in degrees, so the default
    converts them to radians before multiplying by a 50-mm head radius.  FD's
    first sample is defined as zero; temporal derivative columns, if present,
    are ignored by design.
    """

    array = np.asarray(motion)
    if array.ndim != 2 or array.shape[1] < 6:
        raise ConfoundValidationError(
            f"motion must have at least six columns, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or array.shape[0] <= 0:
        raise ConfoundValidationError("motion must be a non-empty numeric matrix")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ConfoundValidationError("radius_mm must be positive and finite")
    values = np.asarray(array[:, :6], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if rotation_in_degrees:
        values = values.copy()
        values[:, 3:6] = np.deg2rad(values[:, 3:6])
    values[:, 3:6] *= float(radius_mm)
    difference = np.zeros_like(values)
    if values.shape[0] > 1:
        difference[1:] = values[1:] - values[:-1]
    return np.ascontiguousarray(np.abs(difference).sum(axis=1), dtype=np.float64)


# Alias naming used by some callers.
def build_valid_time_mask(
    *,
    finite_time: np.ndarray | Sequence[bool],
    nonsteady: np.ndarray | Sequence[bool] | None = None,
    framewise_displacement: np.ndarray | Sequence[float] | None = None,
    standardized_dvars: np.ndarray | Sequence[float] | None = None,
    fd_threshold_mm: float = FD_THRESHOLD_MM,
    dvars_threshold: float = STD_DVARS_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the authoritative finite/steady/motion-valid time mask."""

    finite = np.asarray(finite_time)
    if finite.ndim != 1 or finite.dtype != np.bool_:
        raise ConfoundValidationError("finite_time must be a boolean vector")
    n_time = finite.size
    steady_flags = np.zeros(n_time, dtype=np.bool_) if nonsteady is None else np.asarray(nonsteady)
    if steady_flags.shape != (n_time,) or steady_flags.dtype != np.bool_:
        raise ConfoundValidationError("nonsteady must be a boolean vector matching finite_time")
    fd = _check_vector(framewise_displacement, n_time, name="framewise_displacement")
    dvars = _check_vector(standardized_dvars, n_time, name="standardized_dvars")
    if not np.isfinite(fd_threshold_mm) or float(fd_threshold_mm) < 0:
        raise ConfoundValidationError("fd_threshold_mm must be finite and non-negative")
    if not np.isfinite(dvars_threshold) or float(dvars_threshold) < 0:
        raise ConfoundValidationError("dvars_threshold must be finite and non-negative")
    motion_qc = (
        np.isfinite(fd)
        & np.isfinite(dvars)
        & (fd >= 0.0)
        & (fd <= float(fd_threshold_mm))
        & (dvars >= 0.0)
        & (dvars <= float(dvars_threshold))
    )
    valid = np.ascontiguousarray(finite & ~steady_flags & motion_qc, dtype=np.bool_)
    return valid, np.ascontiguousarray(motion_qc, dtype=np.bool_)


def standardized_dvars(
    data: np.ndarray,
    *,
    valid_voxel_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Compute the runwise standardized-DVARS scale used by the HCP adapter.

    Every finite, non-flat voxel is standardized across the complete run; the
    RMS temporal derivative is then divided by ``sqrt(2)``.  The first sample
    is defined as zero.  This is deliberately an explicit HCP numerical
    recipe, not a claim to reproduce fMRIPrep's internal ``std_dvars``.
    """

    values = np.asarray(data)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise FmriError("data must be a numeric [time, voxel] matrix")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise FmriError("data must be non-empty")
    if valid_voxel_mask is None:
        supplied = np.ones(values.shape[1], dtype=np.bool_)
    else:
        supplied = np.asarray(valid_voxel_mask)
        if supplied.shape != (values.shape[1],) or supplied.dtype != np.bool_:
            raise FmriError("valid_voxel_mask must be boolean [voxel]")
    finite = np.isfinite(values).all(axis=0)
    scale = np.std(np.asarray(values, dtype=np.float64), axis=0, ddof=0)
    usable = supplied & finite & np.isfinite(scale) & (scale > np.finfo(np.float64).eps)
    if not np.any(usable):
        raise FmriError("cannot compute standardized DVARS: no finite non-flat voxels")
    standardized = (
        np.asarray(values[:, usable], dtype=np.float64)
        - np.asarray(values[:, usable], dtype=np.float64).mean(axis=0)
    ) / scale[usable]
    result = np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[0] > 1:
        differences = np.diff(standardized, axis=0)
        result[1:] = np.sqrt(np.mean(np.square(differences), axis=1)) / math.sqrt(2.0)
    if not np.isfinite(result).all():
        raise FmriError("standardized DVARS is non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def fill_invalid_rows_linear(
    values: np.ndarray,
    valid_time: np.ndarray | Sequence[bool],
) -> np.ndarray:
    """Fill invalid rows linearly, with zero-filled boundary gaps.

    This function is only a finite-storage filler.  It never changes valid
    rows, and callers must retain/use ``valid_time`` as the authoritative mask.
    """

    array = np.asarray(values)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise FmriError("values must be a numeric [time, feature] matrix")
    valid = np.asarray(valid_time)
    if valid.shape != (array.shape[0],) or valid.dtype != np.bool_:
        raise FmriError("valid_time must be a boolean vector matching values rows")
    output = np.zeros(array.shape, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return np.ascontiguousarray(output, dtype=np.float32)
    valid_float = np.asarray(array[valid], dtype=np.float64)
    if not np.isfinite(valid_float).all():
        raise FmriError("valid rows contain non-finite values")
    output[valid] = valid_float
    if valid_indices.size >= 2:
        first = int(valid_indices[0])
        last = int(valid_indices[-1])
        interior = np.arange(first, last + 1)
        invalid_interior = ~valid[first : last + 1]
        if np.any(invalid_interior):
            for column in range(array.shape[1]):
                output[first : last + 1, column] = np.interp(
                    interior.astype(np.float64),
                    valid_indices.astype(np.float64),
                    valid_float[:, column],
                )
            # Reassign valid rows to protect against interpolation roundoff.
            output[valid] = valid_float
    return np.ascontiguousarray(output, dtype=np.float32)


def _volume_input_shape(value: Any) -> tuple[tuple[int, ...], Any]:
    """Get a volume shape and sliceable source without touching a proxy."""

    shape = getattr(value, "shape", None)
    if shape is None:
        array = np.asarray(value)
        return tuple(int(v) for v in array.shape), array
    try:
        normalized_shape = tuple(int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise FmriError("fMRI data shape is invalid") from exc
    # A nibabel ArrayProxy is sliceable and does not materialize the whole run.
    source = getattr(value, "dataobj", value)
    if not hasattr(source, "__getitem__"):
        if hasattr(value, "get_fdata"):
            try:
                source = value.get_fdata(dtype=np.float32)
            except TypeError:
                source = value.get_fdata()
        else:
            source = np.asarray(value)
    return normalized_shape, source


def _describe_volume_input(
    data: Any,
    labels: Any,
    *,
    affine: np.ndarray | None,
    spatial_mask: np.ndarray | Any | None,
) -> dict[str, Any]:
    """Describe a 4-D input and atlas without loading the fMRI proxy."""

    labels_array = _coerce_array(labels, name="labels")
    if labels_array.ndim != 3:
        raise AtlasValidationError("volume processing requires a 3-D label image")
    label_vector = validate_schaefer_labels(labels_array)
    label_shape = tuple(int(v) for v in labels_array.shape)
    data_shape, source = _volume_input_shape(data)
    if len(data_shape) != 4:
        raise FmriError(f"volume processing requires 4-D data, got {data_shape}")
    if tuple(data_shape[:3]) == label_shape:
        layout = "xyzt"
        n_time = int(data_shape[-1])
    elif tuple(data_shape[1:]) == label_shape:
        layout = "txyz"
        n_time = int(data_shape[0])
    else:
        raise FmriError(
            "4-D fMRI spatial dimensions do not match the 3-D Schaefer label image"
        )
    if n_time <= 0:
        raise FmriError("fMRI volume has no timepoints")

    data_affine = affine
    source_affine = getattr(data, "affine", None)
    if data_affine is not None and source_affine is not None:
        if not np.allclose(
            np.asarray(data_affine, dtype=np.float64),
            np.asarray(source_affine, dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise GridValidationError("explicit affine differs from the fMRI image affine")
    if data_affine is None:
        data_affine = source_affine
    if data_affine is not None:
        data_affine = np.asarray(data_affine, dtype=np.float64)
        validate_nifti_grid(
            label_shape + (n_time,),
            data_affine,
            expected_shape=label_shape,
            name="fMRI data",
        )
        labels_affine = getattr(labels, "affine", None)
        if labels_affine is not None:
            validate_same_grid(
                label_shape,
                labels,
                reference_affine=data_affine,
                candidate_affine=labels_affine,
                names=("fMRI data", "Schaefer labels"),
            )

    if spatial_mask is None:
        # Without a supplied brain mask, the atlas support is the closed default.
        mask = label_vector.reshape(label_shape) > 0
    else:
        mask = _coerce_boolean_mask(spatial_mask, label_shape, name="spatial mask")
        mask_affine = getattr(spatial_mask, "affine", None)
        if data_affine is not None and mask_affine is not None:
            if not np.allclose(
                np.asarray(data_affine, dtype=np.float64),
                np.asarray(mask_affine, dtype=np.float64),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise GridValidationError("fMRI data and spatial-mask affines differ")
    # A supplied functional brain mask is authoritative for voxel selection.
    # In particular, an atlas label outside that mask is not a usable parcel
    # voxel: it must not invalidate a time point or contribute to a parcel
    # denominator.  Keep this intersection explicit in the descriptor so every
    # streaming pass consumes exactly the same set.
    mask_vector = np.asarray(mask, dtype=np.bool_).reshape(-1)
    labeled_positions = np.flatnonzero((label_vector > 0) & mask_vector).astype(
        np.int64, copy=False
    )
    if labeled_positions.size == 0:
        raise AtlasValidationError(
            "Schaefer label image has no in-mask labeled voxels"
        )
    labeled_labels = np.ascontiguousarray(label_vector[labeled_positions], dtype=np.int32)
    parcel_counts = np.bincount(
        labeled_labels, minlength=N_SCHAEFER_PARCELS + 1
    )[1:]
    missing = np.flatnonzero(parcel_counts == 0) + 1
    if missing.size:
        preview = missing[:10].tolist()
        suffix = "..." if missing.size > 10 else ""
        raise AtlasValidationError(
            "Schaefer atlas has parcels with no in-mask voxels: "
            f"{preview}{suffix}"
        )
    return {
        "source": source,
        "data_shape": data_shape,
        "layout": layout,
        "spatial_shape": label_shape,
        "n_time": n_time,
        "affine": data_affine,
        "labels": label_vector,
        "labeled_positions": labeled_positions,
        "labeled_labels": labeled_labels,
        "parcel_counts": np.ascontiguousarray(parcel_counts, dtype=np.int64),
        "spatial_mask": mask,
    }


def _volume_chunk(
    descriptor: Mapping[str, Any],
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one chunk and retain only its labeled voxels."""

    source = descriptor["source"]
    layout = str(descriptor["layout"])
    spatial_shape = tuple(descriptor["spatial_shape"])
    if layout == "xyzt":
        raw = np.asarray(source[..., start:stop])
    else:
        raw = np.asarray(source[start:stop, ...])
    if raw.ndim != 4:
        raise FmriError(f"volume source returned a non-4-D chunk: {raw.shape}")
    if layout == "xyzt":
        raw_time = np.moveaxis(raw, -1, 0)
    else:
        raw_time = raw
    if raw_time.shape[1:] != spatial_shape:
        raise FmriError(
            f"volume source chunk spatial shape {raw_time.shape[1:]} does not match "
            f"atlas {spatial_shape}"
        )
    positions = descriptor["labeled_positions"]
    # Background voxels, including pathological NaNs outside the atlas, never
    # participate in finite-time validity.
    raw_matrix = raw_time.reshape(raw_time.shape[0], -1)
    finite = np.isfinite(raw_matrix[:, positions]).all(axis=1)
    selected = np.ascontiguousarray(raw_matrix[:, positions], dtype=np.float32)
    return selected, np.ascontiguousarray(finite, dtype=np.bool_)


def process_volume_minimal(
    data: Any,
    labels: Any,
    *,
    framewise_displacement: np.ndarray | Sequence[float],
    standardized_dvars: np.ndarray | Sequence[float],
    valid_time: np.ndarray | Sequence[bool] | None = None,
    nonsteady: np.ndarray | Sequence[bool] | None = None,
    affine: np.ndarray | None = None,
    spatial_mask: np.ndarray | Any | None = None,
    time_chunk_size: int = 32,
) -> FmriResult:
    """Parcel means of an unsmoothed 4-D volume, z-scored then linearly detrended on valid rows.

    TRIBEv2's recipe (d'Ascoli et al. 2026, sections 5.7-5.8) for HCP's minimally preprocessed
    volumes: no spatial smoothing, no confound or global-signal regression and no high-pass, which
    remove movie-driven signal. Validity combines finite rows, nonsteady flags, FD and
    standardized-DVARS thresholds (:func:`build_valid_time_mask`) and an explicit ``valid_time``;
    invalid rows are flagged and filled for storage only.
    """

    if not isinstance(time_chunk_size, (int, np.integer)) or int(time_chunk_size) <= 0:
        raise FmriError("time_chunk_size must be a positive integer")
    descriptor = _describe_volume_input(data, labels, affine=affine, spatial_mask=spatial_mask)
    n_time = int(descriptor["n_time"])
    parcel_counts = np.asarray(descriptor["parcel_counts"], dtype=np.int64)
    if parcel_counts.shape != (N_SCHAEFER_PARCELS,) or np.any(parcel_counts <= 0):
        raise FmriError("every Schaefer parcel must have at least one in-mask voxel")
    fd = _check_vector(framewise_displacement, n_time, name="framewise_displacement")
    dvars = _check_vector(standardized_dvars, n_time, name="standardized_dvars")
    steady = np.zeros(n_time, dtype=np.bool_) if nonsteady is None else np.asarray(nonsteady)
    explicit_valid = np.ones(n_time, dtype=np.bool_) if valid_time is None else np.asarray(valid_time)
    for name, value in (("nonsteady", steady), ("valid_time", explicit_valid)):
        if value.shape != (n_time,) or value.dtype != np.bool_:
            raise FmriError(f"{name} must be a boolean vector matching data rows")

    # Parcel sums over contiguous label runs of the labeled voxels.
    order = np.argsort(descriptor["labeled_labels"], kind="stable")
    starts = np.searchsorted(descriptor["labeled_labels"][order], np.arange(1, N_SCHAEFER_PARCELS + 1))
    parcels = np.zeros((n_time, N_SCHAEFER_PARCELS), dtype=np.float64)
    finite_time = np.empty(n_time, dtype=np.bool_)
    for start in range(0, n_time, int(time_chunk_size)):
        stop = min(start + int(time_chunk_size), n_time)
        selected, finite = _volume_chunk(descriptor, start, stop)
        finite_time[start:stop] = finite
        sums = np.add.reduceat(np.asarray(selected[:, order], dtype=np.float64), starts, axis=1)
        parcels[start:stop] = sums / parcel_counts[None, :]
    valid, motion_qc = build_valid_time_mask(
        finite_time=finite_time,
        nonsteady=steady,
        framewise_displacement=fd,
        standardized_dvars=dvars,
    )
    valid &= explicit_valid
    rows = np.flatnonzero(valid)
    if rows.size < 3:
        raise FmriError("at least three valid rows are required for z-scoring and detrending")
    values = parcels[rows]
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0, ddof=1), np.finfo(np.float64).eps)
    trend = np.column_stack([np.ones(rows.size), rows - rows.mean()])
    values -= trend @ np.linalg.lstsq(trend, values, rcond=None)[0]
    storage = np.zeros((n_time, N_SCHAEFER_PARCELS), dtype=np.float32)
    storage[rows] = values
    storage = fill_invalid_rows_linear(storage, valid)
    provenance = {
        "space": "schaefer1000",
        "native_clock": True,
        "recipe": "minimal_tribev2",
        "smoothing": {"applied": False},
        "confound_regression": False,
        "parcel_reduction": "mean_of_in_mask_voxels",
        "temporal_normalization": "parcel_zscore_ddof1_then_linear_detrend_on_valid_rows",
        "invalid_storage_fill": "linear_interpolation_interior_zero_boundaries",
        "valid_rows": int(valid.sum()),
        "total_rows": int(n_time),
    }
    return FmriResult(
        parcels=np.ascontiguousarray(storage, dtype=np.float32),
        valid_time_native=valid,
        finite_time_native=np.asarray(finite_time, dtype=np.bool_),
        nonsteady_native=np.asarray(steady, dtype=np.bool_),
        motion_qc_native=np.asarray(motion_qc, dtype=np.bool_),
        framewise_displacement=np.asarray(fd, dtype=np.float32),
        standardized_dvars=np.asarray(dvars, dtype=np.float32),
        provenance=provenance,
    )


__all__ = [
    "AtlasValidationError",
    "ConfoundValidationError",
    "FD_THRESHOLD_MM",
    "FmriError",
    "FmriResult",
    "GridValidationError",
    "N_SCHAEFER_PARCELS",
    "STD_DVARS_THRESHOLD",
    "build_valid_time_mask",
    "fill_invalid_rows_linear",
    "load_nifti",
    "power_framewise_displacement",
    "process_volume_minimal",
    "standardized_dvars",
    "validate_nifti_grid",
    "validate_same_grid",
    "validate_schaefer_labels",
]
