"""Shared adapter contracts for the native-clock Schaefer-1000 fMRI bundle.

Dataset-specific source loading and signal preparation live in the sibling
``algonauts`` and ``hcp_movies`` modules. This module keeps
their common validation, image/atlas IO, provenance, result contracts, and
lazy dispatcher.

The common output is :class:`PreparedFmriRun`.  Its ``brain`` member is a
finite ``float32`` ``[time, 1000]`` Schaefer parcel matrix.  Rows that fail
motion/steady-state/confound validity are finite storage fillers; the ``masks``
mapping is authoritative and must be used by downstream training code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from state_of_mind.nbs_data.fmri.common import (
    FmriError,
    FD_THRESHOLD_MM,
    GridValidationError,
    N_SCHAEFER_PARCELS,
    STD_DVARS_THRESHOLD,
    validate_same_grid,
    validate_schaefer_labels,
)
from state_of_mind.nbs_data.fmri._sources import NativeRunSource, TR_SECONDS


OBSERVATION_SPACE = "schaefer1000"


class AdapterError(ValueError):
    """Base class for a source adapter contract violation."""


class MissingSourceAssetError(AdapterError):
    """Raised when a required source, mask, confound, or tissue asset is absent."""


class SourceShapeError(AdapterError):
    """Raised when a source array does not match its declared run contract."""


def _validate_dataset_tr(source: NativeRunSource, *, dataset: str | None = None) -> None:
    """Require the declared source clock to match the immutable dataset clock."""

    canonical = (dataset or source.dataset_id).strip().lower().replace("-", "_")
    if canonical in {"hcp", "hcp_movie"}:
        canonical = "hcp_movies"
    expected = TR_SECONDS.get(canonical)
    if expected is None:
        return
    if isinstance(source.tr_seconds, (bool, np.bool_)):
        raise SourceShapeError(
            f"{source.dataset_id} run {source.run_id} has a boolean TR"
        )
    try:
        observed = float(source.tr_seconds)
    except (TypeError, ValueError) as exc:
        raise SourceShapeError(
            f"{source.dataset_id} run {source.run_id} has a non-numeric TR"
        ) from exc
    if not np.isfinite(observed) or not np.isclose(
        observed, expected, rtol=0.0, atol=1e-8
    ):
        raise SourceShapeError(
            f"{source.dataset_id} run {source.run_id} declares TR {observed}; "
            f"expected dataset TR {expected} seconds"
        )


def _validate_prepare_source(
    source: NativeRunSource,
    *,
    dataset: str | None = None,
) -> str:
    """Validate the common source boundary and return its canonical dataset."""

    if not isinstance(source, NativeRunSource):
        raise AdapterError("source must be a NativeRunSource")
    canonical = (dataset or source.dataset_id).strip().lower().replace("-", "_")
    _validate_dataset_tr(source, dataset=canonical)
    return canonical


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _load_nifti(value: Any, *, name: str) -> tuple[Any, Path | None]:
    """Load a NIfTI path or validate a NIfTI-like object.

    Keeping path handling here lets synthetic tests and callers with an
    already-open nibabel image use exactly the same processing path.
    """

    if isinstance(value, (str, Path)):
        path = _as_path(value)
        if not path.is_file():
            raise MissingSourceAssetError(f"Missing {name}: {path}")
        try:
            from state_of_mind.nbs_data.fmri.common import load_nifti

            return load_nifti(path), path
        except FmriError as exc:
            raise AdapterError(str(exc)) from exc
    if not hasattr(value, "shape") or not hasattr(value, "affine"):
        raise AdapterError(f"{name} must be a path or NIfTI-like object with shape/affine")
    return value, None


def _nifti_values(image: Any, *, name: str, dtype: np.dtype[Any] = np.dtype(np.float32)) -> np.ndarray:
    try:
        if hasattr(image, "get_fdata"):
            try:
                values = image.get_fdata(dtype=dtype)
            except TypeError:
                values = image.get_fdata()
        elif hasattr(image, "dataobj"):
            values = np.asanyarray(image.dataobj)
        else:
            values = np.asarray(image)
    except Exception as exc:
        raise AdapterError(f"Cannot materialize {name}: {exc}") from exc
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise SourceShapeError(f"{name} must be numeric, got {array.dtype}")
    return array


@dataclass(frozen=True)
class _InMemoryNifti:
    """Minimal NIfTI-like wrapper around one materialized float32 volume."""

    dataobj: np.ndarray
    affine: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.dataobj.shape)


def _materialize_nifti_float32(image: Any, *, name: str) -> _InMemoryNifti:
    """Read a compressed 4-D source once for repeated deterministic passes."""

    values = np.asarray(
        _nifti_values(image, name=name, dtype=np.dtype(np.float32)),
        dtype=np.float32,
    )
    if values.ndim != 4 or values.shape[-1] <= 0:
        raise SourceShapeError(f"{name} must be a non-empty 4-D image, got {values.shape}")
    try:
        affine = np.asarray(getattr(image, "affine"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SourceShapeError(f"{name} affine is invalid") from exc
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise SourceShapeError(f"{name} affine is invalid")
    return _InMemoryNifti(dataobj=values, affine=affine)


def _source_files(
    source: NativeRunSource,
    *,
    extra: Iterable[tuple[str, Path]] = (),
) -> list[dict[str, Any]]:
    """Return exact-path structural provenance records.

    Active Schaefer bundles record the filesystem facts needed to detect a
    changed input, without computing a content digest.  Header/dataset
    metadata is inspected only for shape and dtype; payloads are never
    materialized by this provenance helper.
    """

    values: list[tuple[str, Path]] = [("fmri", source.fmri_path)]
    for attribute, role in (
        ("sidecar_json", "sidecar_json"),
        ("brain_mask", "brain_mask"),
        ("confounds_tsv", "confounds_tsv"),
        ("motion_regressors", "motion_regressors"),
        ("tissue_labels", "tissue_labels"),
    ):
        path = getattr(source, attribute)
        if path is not None:
            values.append((role, path))
    values.extend(extra)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, Path]] = set()
    for role, raw_path in values:
        path = _as_path(raw_path)
        key = (str(role), path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise MissingSourceAssetError(
                f"Missing {role} asset for {source.run_id}: {path}"
            )
        try:
            stat = path.stat()
        except OSError as exc:
            raise MissingSourceAssetError(
                f"Cannot inspect {role} asset {path}: {exc}"
            ) from exc
        try:
            relative = source.source_relative(path)
        except Exception:
            relative = str(path)
        record: dict[str, Any] = {
            "role": str(role),
            "path": str(path),
            "source_relative_path": relative,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        # Structural metadata is deliberately bounded to headers and HDF5
        # dataset descriptors.  It does not imply a checksum or full read.
        try:
            if path.name.lower().endswith((".nii", ".nii.gz")):
                import nibabel as nib

                image = nib.load(str(path))
                record["shape"] = [int(value) for value in image.shape]
                record["dtype"] = np.dtype(image.header.get_data_dtype()).name
            elif path.suffix.lower() in {".h5", ".hdf5"} and source.hdf5_key:
                import h5py

                with h5py.File(path, "r") as handle:
                    dataset = handle[source.hdf5_key]
                    record["shape"] = [int(value) for value in dataset.shape]
                    record["dtype"] = np.dtype(dataset.dtype).name
        except (OSError, ValueError, KeyError, ImportError) as exc:
            raise MissingSourceAssetError(
                f"Cannot inspect structural metadata for {role} asset {path}: {exc}"
            ) from exc
        records.append(record)
    return records


def _base_provenance(
    source: NativeRunSource,
    *,
    adapter: str,
    extra_files: Iterable[tuple[str, Path]] = (),
) -> dict[str, Any]:
    if source.n_volumes <= 0 or not np.isfinite(source.tr_seconds) or source.tr_seconds <= 0:
        raise SourceShapeError(f"Invalid run timing for {source.run_id}: {source.n_volumes}, {source.tr_seconds}")
    return {
        "adapter": adapter,
        "observation_space": OBSERVATION_SPACE,
        "source": {
            "dataset_id": source.dataset_id,
            "run_id": source.run_id,
            "timeline_id": source.timeline_id,
            "subject_id": source.subject_id,
            "source_space": source.source_space,
            "tr_seconds": float(source.tr_seconds),
            "n_volumes": int(source.n_volumes),
            "fmri_path": str(_as_path(source.fmri_path)),
            "files": _source_files(source, extra=extra_files),
        },
    }


def _spatial_mask(image: Any, *, reference: Any, name: str) -> np.ndarray:
    try:
        validate_same_grid(reference, image, names=("fMRI", name))
    except GridValidationError as exc:
        raise AdapterError(str(exc)) from exc
    values = _nifti_values(image, name=name)
    if values.ndim != 3:
        raise SourceShapeError(f"{name} must be 3-D, got {values.shape}")
    if not np.isfinite(values).all():
        raise SourceShapeError(f"{name} contains non-finite values")
    mask = np.asarray(values > 0, dtype=np.bool_)
    if not np.any(mask):
        raise SourceShapeError(f"{name} contains no positive voxels")
    return np.ascontiguousarray(mask)


def _resolve_atlas(
    atlas_image: Any,
    *,
    reference: Any,
    allow_resample: bool,
) -> tuple[Any, Path | None, bool]:
    """Resolve a caller-supplied atlas and require a matching source grid.

    Resampling is opt-in and nearest-neighbor only.  The default is strict
    because a silent MNI grid mismatch can permute parcel assignments.
    """

    image, path = _load_nifti(atlas_image, name="Schaefer atlas")
    resampled = False
    try:
        validate_same_grid(reference, image, names=("fMRI", "Schaefer atlas"))
    except GridValidationError as exc:
        if not allow_resample:
            raise AdapterError(str(exc)) from exc
        try:
            import nibabel as nib
            from nibabel.processing import resample_from_to

            target = (tuple(int(v) for v in reference.shape[:3]), np.asarray(reference.affine))
            image = resample_from_to(image, target, order=0)
            validate_same_grid(reference, image, names=("fMRI", "resampled Schaefer atlas"))
            resampled = True
        except Exception as resample_exc:
            raise AdapterError(
                f"Schaefer atlas grid mismatch and nearest resampling failed: {resample_exc}"
            ) from resample_exc
    values = _nifti_values(image, name="Schaefer atlas")
    if values.ndim != 3:
        raise SourceShapeError(f"Schaefer atlas must be 3-D, got {values.shape}")
    try:
        validate_schaefer_labels(values, n_parcels=N_SCHAEFER_PARCELS)
    except FmriError as exc:
        raise AdapterError(str(exc)) from exc
    return image, path, resampled


def _atlas_resampling_record(
    atlas_image: Any,
    reference: Any,
    *,
    atlas_path: Path | None,
    performed: bool,
) -> dict[str, Any]:
    """Describe atlas grid conversion with JSON-safe source/target geometry."""

    original, _ = _load_nifti(atlas_image, name="Schaefer atlas")
    original_shape = tuple(int(value) for value in getattr(original, "shape", ()))
    if len(original_shape) != 3:
        raise SourceShapeError("Schaefer atlas must be 3-D for provenance, got %s" % (original_shape,))
    raw_target_shape = getattr(reference, "shape", ())
    target_shape = tuple(int(value) for value in raw_target_shape[:3])
    if len(target_shape) != 3:
        raise SourceShapeError("fMRI reference must have three spatial dimensions, got %s" % (raw_target_shape,))
    source_affine = np.asarray(getattr(original, "affine", None), dtype=np.float64)
    target_affine = np.asarray(getattr(reference, "affine", None), dtype=np.float64)
    if source_affine.shape != (4, 4) or target_affine.shape != (4, 4):
        raise SourceShapeError("atlas and fMRI affines must be 4x4 for provenance")
    if not np.isfinite(source_affine).all() or not np.isfinite(target_affine).all():
        raise SourceShapeError("atlas and fMRI affines must be finite for provenance")

    def grid(shape: tuple[int, int, int], affine: np.ndarray) -> dict[str, Any]:
        return dict(shape=list(shape), affine=affine.tolist())

    return dict(
        performed=bool(performed),
        method="nibabel.processing.resample_from_to",
        interpolation="nearest_neighbor",
        order=0,
        original_atlas_path=None if atlas_path is None else str(atlas_path),
        source_grid=grid(original_shape, source_affine),
        target_fmri_grid=grid(target_shape, target_affine),
    )


def _require_source_assets(source: NativeRunSource, attributes: Sequence[str]) -> None:
    """Require both presence and on-disk existence for adapter-specific assets."""

    missing: list[str] = []
    for attribute in attributes:
        value = getattr(source, attribute, None)
        if value is None or not _as_path(value).is_file():
            missing.append(attribute)
    if missing:
        raise MissingSourceAssetError(
            f"{source.dataset_id} run {source.run_id} is missing required assets: {missing}"
        )


def _coerce_stimulus_support(
    source: NativeRunSource,
    value: Sequence[bool] | np.ndarray | None,
) -> np.ndarray:
    """Validate the timeline support mask before numerical processing."""

    support = (
        np.ones(int(source.n_volumes), dtype=np.bool_)
        if value is None
        else np.asarray(value)
    )
    if support.shape != (int(source.n_volumes),) or support.dtype != np.bool_:
        raise SourceShapeError(
            f"stimulus_support must be boolean [{int(source.n_volumes)}]"
        )
    return np.ascontiguousarray(support, dtype=np.bool_)


def _result_from_fmri(
    source: NativeRunSource,
    result: Any,
    *,
    provenance: Mapping[str, Any],
    stimulus_support: Sequence[bool] | np.ndarray | None,
) -> "PreparedFmriRun":
    n_time = int(source.n_volumes)
    support = _coerce_stimulus_support(source, stimulus_support)
    finite_time = np.ascontiguousarray(result.finite_time_native, dtype=np.bool_)
    nonsteady_native = np.ascontiguousarray(result.nonsteady_native, dtype=np.bool_)
    motion_qc_native = np.ascontiguousarray(result.motion_qc_native, dtype=np.bool_)
    authoritative_valid_time = np.asarray(result.valid_time_native)
    if (
        authoritative_valid_time.shape != (n_time,)
        or authoritative_valid_time.dtype != np.bool_
    ):
        raise SourceShapeError(
            f"valid_time_native must be boolean [{n_time}]"
        )
    authoritative_valid_time = np.ascontiguousarray(
        authoritative_valid_time, dtype=np.bool_
    )
    if np.any(authoritative_valid_time & ~support):
        raise SourceShapeError(
            "valid_time_native includes rows outside stimulus_support"
        )
    # Keep the source-only diagnostic count for QC, but never reconstruct the
    # authoritative validity mask: the kernel may apply design-specific rules.
    fmri_valid = np.ascontiguousarray(
        finite_time & ~nonsteady_native & motion_qc_native, dtype=np.bool_
    )
    masks = {
        "valid_time": authoritative_valid_time,
        "nonsteady": nonsteady_native,
        "motion_qc": motion_qc_native,
        "stimulus_support": np.ascontiguousarray(support, dtype=np.bool_),
    }
    qc = {
        "framewise_displacement": np.ascontiguousarray(result.framewise_displacement, dtype=np.float32),
        "standardized_dvars": np.ascontiguousarray(result.standardized_dvars, dtype=np.float32),
        "fd": np.ascontiguousarray(result.framewise_displacement, dtype=np.float32),
        "std_dvars": np.ascontiguousarray(result.standardized_dvars, dtype=np.float32),
        "n_timepoints": n_time,
        "n_valid": int(masks["valid_time"].sum()),
        "n_fmri_valid": int(fmri_valid.sum()),
        "n_finite": int(finite_time.sum()),
        "finite_fraction": float(finite_time.mean()),
        "valid_fraction": float(masks["valid_time"].mean()),
    }
    merged = dict(provenance)
    merged["n_timepoints"] = n_time
    merged["finite_time"] = {
        "n_finite": int(finite_time.sum()),
        "n_nonfinite": int((~finite_time).sum()),
        "finite_fraction": float(finite_time.mean()),
    }
    merged["valid_time"] = {
        "formula": "authoritative kernel valid_time_native",
        "n_fmri_valid": int(fmri_valid.sum()),
        "n_stimulus_supported": int(support.sum()),
        "n_valid": int(masks["valid_time"].sum()),
        "n_invalid": int((~masks["valid_time"]).sum()),
    }
    adapter_processing = dict(merged.get("processing", {}))
    kernel_processing = dict(result.provenance)
    merged["processing"] = {**kernel_processing, **adapter_processing, "kernel": kernel_processing}
    return PreparedFmriRun(
        source=source,
        brain=np.ascontiguousarray(result.parcels, dtype=np.float32),
        masks=masks,
        qc=qc,
        provenance=merged,
    )


@dataclass(frozen=True)
class PreparedFmriRun:
    """Common adapter output for one native-clock run."""

    source: NativeRunSource
    brain: np.ndarray
    masks: Mapping[str, np.ndarray]
    qc: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        brain = np.asarray(self.brain)
        expected = (int(self.source.n_volumes), N_SCHAEFER_PARCELS)
        if brain.shape != expected:
            raise SourceShapeError(f"brain must have shape {expected}, got {brain.shape}")
        if brain.dtype != np.float32 or not brain.flags.c_contiguous:
            raise SourceShapeError("brain must be C-contiguous float32")
        if not np.isfinite(brain).all():
            raise SourceShapeError("brain contains non-finite values")
        required = {"valid_time", "nonsteady", "motion_qc", "stimulus_support"}
        supplied = set(self.masks)
        if supplied != required:
            raise SourceShapeError(f"masks must be exactly {sorted(required)}; got {sorted(supplied)}")
        masks: dict[str, np.ndarray] = {}
        for name, raw in self.masks.items():
            value = np.asarray(raw)
            if value.shape != (expected[0],) or value.dtype != np.bool_:
                raise SourceShapeError(f"mask {name!r} must be boolean [{expected[0]}]")
            masks[name] = np.ascontiguousarray(value, dtype=np.bool_)
        qc = dict(self.qc)
        for key in ("framewise_displacement", "standardized_dvars"):
            if key not in qc:
                raise SourceShapeError(f"qc missing {key!r}")
            value = np.asarray(qc[key])
            if value.shape != (expected[0],) or not np.issubdtype(value.dtype, np.number):
                raise SourceShapeError(f"qc {key!r} must be numeric [{expected[0]}]")
            qc[key] = np.ascontiguousarray(value, dtype=np.float32)
        qc.setdefault("fd", qc["framewise_displacement"])
        qc.setdefault("std_dvars", qc["standardized_dvars"])
        object.__setattr__(self, "brain", brain)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "qc", qc)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def valid_time(self) -> np.ndarray:
        return self.masks["valid_time"]

    @property
    def valid_time_native(self) -> np.ndarray:
        return self.valid_time

    @property
    def framewise_displacement(self) -> np.ndarray:
        return self.qc["framewise_displacement"]

    @property
    def standardized_dvars(self) -> np.ndarray:
        return self.qc["standardized_dvars"]

    def qc_summary(self) -> dict[str, Any]:
        """Return a strict-JSON QC summary for BundleWriter."""

        n_time = int(self.source.n_volumes)
        valid = np.asarray(self.masks["valid_time"], dtype=np.bool_)
        support = np.asarray(self.masks["stimulus_support"], dtype=np.bool_)
        nonsteady = np.asarray(self.masks["nonsteady"], dtype=np.bool_)
        motion_qc = np.asarray(self.masks["motion_qc"], dtype=np.bool_)
        fd = np.asarray(self.qc["framewise_displacement"], dtype=np.float64)
        dvars = np.asarray(self.qc["standardized_dvars"], dtype=np.float64)

        def _finite_max(values: np.ndarray) -> float | None:
            finite = values[np.isfinite(values)]
            return None if finite.size == 0 else float(np.max(finite))

        summary: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": self.source.dataset_id,
            "run_id": self.source.run_id,
            "n_timepoints": n_time,
            "thresholds": {
                "framewise_displacement_mm": float(FD_THRESHOLD_MM),
                "fd_mm": float(FD_THRESHOLD_MM),
                "standardized_dvars": float(STD_DVARS_THRESHOLD),
                "std_dvars": float(STD_DVARS_THRESHOLD),
            },
            "counts": {
                "valid_time": int(valid.sum()),
                "invalid_time": int((~valid).sum()),
                "fMRI_valid": int(self.qc.get("n_fmri_valid", valid.sum())),
                "fmri_valid": int(self.qc.get("n_fmri_valid", valid.sum())),
                "stimulus_supported": int(support.sum()),
                "nonsteady": int(nonsteady.sum()),
                "motion_qc_pass": int(motion_qc.sum()),
                "motion_qc_fail": int((~motion_qc).sum()),
                "finite_time": int(self.qc.get("n_finite", n_time)),
            },
            "framewise_displacement": {
                "finite": int(np.isfinite(fd).sum()),
                "nonfinite": int((~np.isfinite(fd)).sum()),
                "above_threshold": int(np.count_nonzero(np.isfinite(fd) & (fd > FD_THRESHOLD_MM))),
                "max_finite": _finite_max(fd),
            },
            "standardized_dvars": {
                "finite": int(np.isfinite(dvars).sum()),
                "nonfinite": int((~np.isfinite(dvars)).sum()),
                "above_threshold": int(np.count_nonzero(np.isfinite(dvars) & (dvars > STD_DVARS_THRESHOLD))),
                "max_finite": _finite_max(dvars),
            },
            "validity_formula": "authoritative kernel valid_time_native",
        }
        json.dumps(summary, allow_nan=False, sort_keys=True)
        return summary


def prepare_run(
    source: NativeRunSource,
    atlas_image: Any | None = None,
    *,
    stimulus_support: Sequence[bool] | np.ndarray | None = None,
    allow_atlas_resampling: bool = False,
    eager_volume: bool = False,
) -> PreparedFmriRun:
    """Prepare one Algonauts or HCP movie run.

    ``atlas_image`` is mandatory for volume sources and must be a complete
    one-based Schaefer-1000 label image on the source grid.  Algonauts is
    already in the official Schaefer-1000 parcel order and therefore does not
    need an atlas image.  Set ``allow_atlas_resampling=True`` only when the
    caller has explicitly audited the nearest-neighbor conversion.
    """

    dataset = _validate_prepare_source(source)
    if dataset == "algonauts":
        from .algonauts import prepare_algonauts

        return prepare_algonauts(source, stimulus_support=stimulus_support)
    if dataset in {"hcp", "hcp_movie", "hcp_movies"}:
        from .hcp_movies import prepare_hcp_movies

        # HCP uses TRIBEv2's minimal recipe: no smoothing or voxel regression to configure.
        return prepare_hcp_movies(
            source,
            atlas_image,
            stimulus_support=stimulus_support,
            allow_atlas_resampling=allow_atlas_resampling,
            eager_volume=eager_volume,
        )
    raise AdapterError(f"Unsupported dataset for fMRI adapter: {source.dataset_id!r}")


__all__ = [
    "AdapterError",
    "MissingSourceAssetError",
    "OBSERVATION_SPACE",
    "PreparedFmriRun",
    "SourceShapeError",
    "prepare_run",
]
