"""Stable storage for the Schaefer-1000 offline preprocessing pipeline.

A run is committed by writing its arrays first and its JSON record last.
``bundle.json`` is likewise written last, so an interrupted run remains
resumable and cannot look complete.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from state_of_mind.data.bundle import (
    BundleValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npy,
    contained_path,
    read_json_object,
    read_jsonl,
    safe_run_filename,
    utc_now,
)


SCHEMA_VERSION = 2
LAYOUT_ID = "neuroworld-schaefer1000-v2"
OBSERVATION_SPACE = "schaefer1000"
BRAIN_VIEW = "projected_native"
FEATURE_VIEW = "multimodal_native_tr"
N_PARCELS = 1000
N_LAYER_GROUPS = 2
SUPPORTED_DATASETS = frozenset({"algonauts", "hcp_movies"})
SUPPORTED_MODALITIES = frozenset({"video", "audio", "text"})
MODALITY_DIMS = {"video": 1408, "audio": 1024, "text": 3072}
DATASET_MODALITIES = {
    "algonauts": frozenset({"video", "audio", "text"}),
    "hcp_movies": frozenset({"video", "audio"}),
}
# Bundle clocks are fixed by the source acquisition contracts.  Keeping this
# check at storage boundaries prevents a hand-built run or feature record from
# reintroducing a silent resampling/clock mismatch after adapter validation.
DATASET_TR_SECONDS = {
    "algonauts": 1.49,
    "hcp_movies": 1.0,
}
REQUIRED_MASKS = frozenset({"valid_time", "nonsteady", "motion_qc", "stimulus_support"})


def select_valid_time(values: Any, valid_time: Any) -> np.ndarray:
    """Select downstream samples using the authoritative ``valid_time`` mask.

    Prepared arrays retain finite filler rows for invalid timepoints.  Loaders
    must apply this mask before training/evaluation; this tiny helper keeps the
    rule explicit for callers that do not need a larger dataset abstraction.
    """

    array = np.asarray(values)
    mask = np.asarray(valid_time)
    if mask.dtype != np.bool_:
        raise BundleValidationError("valid_time must be a boolean vector")
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    if array.ndim < 1:
        raise BundleValidationError("values must have a time axis")
    if mask.ndim != 1 or mask.shape[0] != array.shape[0]:
        raise BundleValidationError(
            f"valid_time must be one-dimensional with length {array.shape[0]}"
        )
    return np.ascontiguousarray(array[mask])


def _strict_json_copy(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Return a strict-JSON copy and reject NaN, arrays, and opaque objects."""

    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(f"{label} must be strict JSON: {exc}") from exc


def _validate_id(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_dataset_tr(dataset_id: str, tr_seconds: Any, *, label: str) -> float:
    if isinstance(tr_seconds, (bool, np.bool_)):
        raise BundleValidationError(f"{label} TR must be numeric, not boolean")
    try:
        tr = float(tr_seconds)
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(f"{label} TR must be numeric") from exc
    expected = DATASET_TR_SECONDS[dataset_id]
    if not np.isfinite(tr) or not np.isclose(tr, expected, rtol=0.0, atol=1e-8):
        raise BundleValidationError(
            f"{label} TR is {tr}; expected {dataset_id} dataset TR {expected} seconds"
        )
    return tr


@dataclass(frozen=True)
class BundlePaths:
    """Canonical paths in one v2 dataset bundle."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path, *, create: bool = False) -> "BundlePaths":
        paths = cls(Path(root).expanduser().resolve(strict=False))
        if create:
            paths.ensure_layout()
        return paths

    @property
    def bundle_json(self) -> Path:
        return self.root / "bundle.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def run_records(self) -> Path:
        return self.runs / "records"

    @property
    def runs_jsonl(self) -> Path:
        return self.runs / "runs.jsonl"

    @property
    def fmri(self) -> Path:
        return self.root / "fmri" / BRAIN_VIEW

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def features(self) -> Path:
        return self.root / "features" / FEATURE_VIEW

    @property
    def feature_arrays(self) -> Path:
        return self.features / "arrays"

    @property
    def feature_records(self) -> Path:
        return self.features / "records"

    @property
    def features_jsonl(self) -> Path:
        return self.features / "features.jsonl"

    @property
    def qc(self) -> Path:
        return self.root / "qc" / "runs"

    @property
    def preparation(self) -> Path:
        return self.root / "preparation"

    def ensure_layout(self) -> None:
        for directory in (
            self.root,
            self.run_records,
            self.fmri,
            *(self.masks / name for name in sorted(REQUIRED_MASKS)),
            self.feature_arrays,
            self.feature_records,
            self.qc,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def fmri_run(self, run_id: str) -> Path:
        return self.fmri / safe_run_filename(run_id)

    def mask_run(self, kind: str, run_id: str) -> Path:
        if kind not in REQUIRED_MASKS:
            raise ValueError(f"Unknown v2 mask kind {kind!r}")
        return self.masks / kind / safe_run_filename(run_id)

    def run_record(self, run_id: str) -> Path:
        return self.run_records / safe_run_filename(run_id, suffix=".json")

    def run_qc(self, run_id: str) -> Path:
        return self.qc / safe_run_filename(run_id, suffix=".json")

    def feature_array(self, timeline_id: str, modality: str) -> Path:
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality {modality!r}")
        return self.feature_arrays / modality / safe_run_filename(timeline_id)

    def feature_record(self, timeline_id: str) -> Path:
        return self.feature_records / safe_run_filename(timeline_id, suffix=".json")



def _is_retired_hash_key(key: object) -> bool:
    """Recognize explicit legacy digest fields without matching ``shape``."""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    tokens = [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
    return any(
        token in {
            "hash",
            "hashes",
            "sha",
            "sha1",
            "sha224",
            "sha256",
            "sha384",
            "sha512",
            "sha1sum",
            "sha224sum",
            "sha256sum",
            "sha384sum",
            "sha512sum",
            "md5",
            "md5sum",
            "checksum",
            "checksums",
            "digest",
            "digests",
        }
        for token in tokens
    )


def _without_hash_fields(value: Any) -> Any:
    """Copy metadata while dropping legacy content-identity fields.

    The root ``hashes_computed`` capability flag is created explicitly during
    finalization. Any inherited occurrence, including a nested one, is stale
    provenance and must not leak into the active bundle.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_retired_hash_key(key_text):
                continue
            result[key_text] = _without_hash_fields(item)
        return result
    if isinstance(value, list):
        return [_without_hash_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_without_hash_fields(item) for item in value]
    return value


def _assert_no_hash_fields(
    value: Any,
    *,
    label: str,
    allow_root_marker: bool = False,
    _depth: int = 0,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            root_marker = (
                allow_root_marker
                and _depth == 0
                and key_text == "hashes_computed"
            )
            if not root_marker and _is_retired_hash_key(key_text):
                raise BundleValidationError(f"{label} contains retired hash field {key_text!r}")
            _assert_no_hash_fields(
                item,
                label=label,
                allow_root_marker=allow_root_marker,
                _depth=_depth + 1,
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_hash_fields(
                item,
                label=label,
                allow_root_marker=allow_root_marker,
                _depth=_depth + 1,
            )


def _artifact_record(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        file_path = path.expanduser().resolve(strict=True)
        root_path = root.expanduser().resolve(strict=True)
        relative = file_path.relative_to(root_path).as_posix()
        stat = file_path.stat()
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Cannot record artifact {path}: {exc}") from exc
    if not file_path.is_file():
        raise BundleValidationError(f"Artifact is not a regular file: {file_path}")
    return {
        "path": relative,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _npy_record(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Cannot read prepared array {path}: {exc}") from exc
    try:
        record = _artifact_record(path, root=root)
        record.update({"shape": list(array.shape), "dtype": array.dtype.name})
        return record
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _finite_chunks(array: np.ndarray, *, chunk_values: int = 1_000_000) -> bool:
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_values):
        if not np.isfinite(flat[start : start + chunk_values]).all():
            return False
    return True


def _validate_npy_record(
    root: Path,
    record: Any,
    *,
    label: str,
    expected_dtype: str,
    expected_shape: tuple[int | None, ...],
    verify_finite: bool,
) -> tuple[int, ...]:
    if not isinstance(record, Mapping):
        raise BundleValidationError(f"{label} record must be an object")
    required = {"path", "size_bytes", "mtime_ns", "shape", "dtype"}
    missing = required - set(record)
    if missing:
        raise BundleValidationError(f"{label} record is missing {sorted(missing)}")
    path = contained_path(root, record["path"], must_exist=True, kind="file", suffix=".npy")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Cannot load {label} array {path}: {exc}") from exc
    try:
        shape = tuple(int(x) for x in array.shape)
        if list(shape) != record["shape"] or array.dtype.name != record["dtype"]:
            raise BundleValidationError(f"{label} manifest/file array contract differs")
        if array.dtype.name != expected_dtype:
            raise BundleValidationError(
                f"{label} must be {expected_dtype}, found {array.dtype.name}"
            )
        if len(shape) != len(expected_shape) or any(
            wanted is not None and actual != wanted
            for actual, wanted in zip(shape, expected_shape)
        ):
            raise BundleValidationError(
                f"{label} must have shape {expected_shape}, found {shape}"
            )
        if not array.flags.c_contiguous:
            raise BundleValidationError(f"{label} must be C-contiguous")
        if verify_finite and np.issubdtype(array.dtype, np.number) and not _finite_chunks(array):
            raise BundleValidationError(f"{label} contains non-finite values")
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    stat = path.stat()
    if stat.st_size != record["size_bytes"]:
        raise BundleValidationError(f"{label} size differs from its record")
    if int(stat.st_mtime_ns) != record["mtime_ns"]:
        raise BundleValidationError(f"{label} mtime differs from its record")
    _assert_no_hash_fields(record, label=label)
    return shape


class BundleWriter:
    """Atomic, resumable writer for one dataset's v2 offline bundle."""

    def __init__(self, root: str | Path, dataset_id: str) -> None:
        dataset_id = _validate_id(dataset_id, label="dataset_id")
        if dataset_id not in SUPPORTED_DATASETS:
            raise BundleValidationError(
                f"Unsupported dataset {dataset_id!r}; expected one of {sorted(SUPPORTED_DATASETS)}"
            )
        self.dataset_id = dataset_id
        self.paths = BundlePaths.from_root(root, create=True)

    def write_run(
        self,
        *,
        run_id: str,
        subject_id: str,
        tr_seconds: float,
        brain: np.ndarray,
        masks: Mapping[str, np.ndarray],
        source: Mapping[str, Any],
        preprocessing: Mapping[str, Any],
        timeline_id: str | None = None,
        time_origin_seconds: float = 0.0,
        qc: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        run_id = _validate_id(run_id, label="run_id")
        subject_id = _validate_id(subject_id, label="subject_id")
        timeline_value = _validate_id(timeline_id or run_id, label="timeline_id")
        tr = _validate_dataset_tr(self.dataset_id, tr_seconds, label=f"Run {run_id!r}")
        origin = float(time_origin_seconds)
        if not np.isfinite(origin):
            raise BundleValidationError("time_origin_seconds must be finite")
        brain_value = np.asarray(brain, dtype=np.float32, order="C")
        if brain_value.ndim != 2 or brain_value.shape[1] != N_PARCELS:
            raise BundleValidationError(
                f"Prepared brain must have shape [T,{N_PARCELS}], found {brain_value.shape}"
            )
        if brain_value.shape[0] == 0 or not np.isfinite(brain_value).all():
            raise BundleValidationError("Prepared brain must be non-empty and finite")
        unknown_masks = set(masks) - REQUIRED_MASKS
        missing_masks = REQUIRED_MASKS - set(masks)
        if unknown_masks or missing_masks:
            raise BundleValidationError(
                f"Run masks differ from the v2 contract: missing={sorted(missing_masks)}, "
                f"unknown={sorted(unknown_masks)}"
            )
        mask_values: dict[str, np.ndarray] = {}
        for name, value in masks.items():
            try:
                mask = np.asarray(value, order="C")
            except (TypeError, ValueError) as exc:
                raise BundleValidationError(f"Mask {name!r} must be a boolean vector") from exc
            if mask.dtype != np.bool_:
                raise BundleValidationError(f"Mask {name!r} must be a boolean vector")
            if mask.shape != (brain_value.shape[0],):
                raise BundleValidationError(
                    f"Mask {name!r} must have shape {(brain_value.shape[0],)}, found {mask.shape}"
                )
            mask_values[name] = np.ascontiguousarray(mask, dtype=np.bool_)
        if np.any(mask_values["valid_time"] & ~mask_values["stimulus_support"]):
            raise BundleValidationError("valid_time cannot include rows outside stimulus_support")
        if np.any(mask_values["valid_time"] & mask_values["nonsteady"]):
            raise BundleValidationError("valid_time cannot include nonsteady rows")
        if np.any(mask_values["valid_time"] & ~mask_values["motion_qc"]):
            raise BundleValidationError("valid_time cannot include motion-censored rows")
        source_value = _strict_json_copy(_without_hash_fields(source), label="source")
        preprocessing_value = _strict_json_copy(_without_hash_fields(preprocessing), label="preprocessing")
        qc_value = _strict_json_copy(_without_hash_fields(qc or {}), label="qc")

        record_path = self.paths.run_record(run_id)
        with nullcontext():
            if record_path.exists() and not overwrite:
                raise FileExistsError(f"Run is already committed: {run_id}")
            brain_path = self.paths.fmri_run(run_id)
            atomic_write_npy(
                brain_path,
                brain_value,
                dtype=np.float32,
                ndim=2,
                finite=True,
                overwrite=overwrite or brain_path.exists(),
            )
            mask_records: dict[str, dict[str, Any]] = {}
            for name in sorted(REQUIRED_MASKS):
                path = self.paths.mask_run(name, run_id)
                atomic_write_npy(
                    path,
                    mask_values[name],
                    dtype=np.bool_,
                    ndim=1,
                    finite=False,
                    overwrite=overwrite or path.exists(),
                )
                mask_records[name] = _npy_record(path, root=self.paths.root)
            atomic_write_json(
                self.paths.run_qc(run_id),
                qc_value,
                overwrite=overwrite or self.paths.run_qc(run_id).exists(),
            )
            record: dict[str, Any] = {
                "run_id": run_id,
                "subject_id": subject_id,
                "timeline_id": timeline_value,
                "observation_space": OBSERVATION_SPACE,
                "tr_seconds": tr,
                "time_origin_seconds": origin,
                "n_timepoints": int(brain_value.shape[0]),
                "brain_views": {BRAIN_VIEW: _npy_record(brain_path, root=self.paths.root)},
                "masks": mask_records,
                "source": source_value,
                "preprocessing": preprocessing_value,
                "qc": _artifact_record(self.paths.run_qc(run_id), root=self.paths.root),
            }
            atomic_write_json(record_path, record, overwrite=overwrite)
            return record

    def write_features(
        self,
        *,
        timeline_id: str,
        tr_seconds: float,
        arrays: Mapping[str, np.ndarray],
        source: Mapping[str, Any],
        pooling: Mapping[str, Any],
        time_origin_seconds: float = 0.0,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        timeline_id = _validate_id(timeline_id, label="timeline_id")
        modalities = frozenset(arrays)
        expected_modalities = DATASET_MODALITIES[self.dataset_id]
        if modalities != expected_modalities:
            raise BundleValidationError(
                f"{self.dataset_id} feature modalities must be exactly "
                f"{sorted(expected_modalities)}, found {sorted(modalities)}"
            )
        tr = _validate_dataset_tr(
            self.dataset_id, tr_seconds, label=f"Feature timeline {timeline_id!r}"
        )
        origin = float(time_origin_seconds)
        if not np.isfinite(origin):
            raise BundleValidationError("Feature time_origin_seconds must be finite")
        values: dict[str, np.ndarray] = {}
        n_timepoints: int | None = None
        for modality, value in arrays.items():
            array = np.asarray(value, dtype=np.float32, order="C")
            if (
                array.ndim != 3
                or array.shape[1] != N_LAYER_GROUPS
                or array.shape[2] != MODALITY_DIMS[modality]
            ):
                raise BundleValidationError(
                    f"{modality} features must have shape "
                    f"[T,{N_LAYER_GROUPS},{MODALITY_DIMS[modality]}], found {array.shape}"
                )
            if array.shape[0] == 0 or not np.isfinite(array).all():
                raise BundleValidationError(f"{modality} features must be non-empty and finite")
            if n_timepoints is None:
                n_timepoints = int(array.shape[0])
            elif array.shape[0] != n_timepoints:
                raise BundleValidationError("All modalities on a timeline must have equal T")
            values[modality] = array
        record_path = self.paths.feature_record(timeline_id)
        with nullcontext():
            if record_path.exists() and not overwrite:
                raise FileExistsError(f"Feature timeline is already committed: {timeline_id}")
            records: dict[str, dict[str, Any]] = {}
            for modality, value in sorted(values.items()):
                path = self.paths.feature_array(timeline_id, modality)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_npy(
                    path,
                    value,
                    dtype=np.float32,
                    ndim=3,
                    finite=True,
                    overwrite=overwrite or path.exists(),
                )
                records[modality] = _npy_record(path, root=self.paths.root)
            record = {
                "timeline_id": timeline_id,
                "feature_view": FEATURE_VIEW,
                "tr_seconds": tr,
                "time_origin_seconds": origin,
                "n_timepoints": n_timepoints,
                "normalization": "none",
                "modalities": records,
                "source": _strict_json_copy(_without_hash_fields(source), label="feature source"),
                "pooling": _strict_json_copy(_without_hash_fields(pooling), label="feature pooling"),
            }
            atomic_write_json(record_path, record, overwrite=overwrite)
            return record

    def finalize(
        self,
        *,
        expected_run_count: int,
        provenance: Mapping[str, Any],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(expected_run_count, int) or isinstance(expected_run_count, bool):
            raise BundleValidationError("expected_run_count must be an integer")
        if expected_run_count <= 0:
            raise BundleValidationError("expected_run_count must be positive")
        run_records = [read_json_object(path) for path in sorted(self.paths.run_records.glob("*.json"))]
        if len(run_records) != expected_run_count:
            raise BundleValidationError(
                f"Expected {expected_run_count} committed runs, found {len(run_records)}"
            )
        run_ids = [record.get("run_id") for record in run_records]
        if len(set(run_ids)) != len(run_ids):
            raise BundleValidationError("Committed run records contain duplicate run IDs")
        run_records.sort(key=lambda value: str(value["run_id"]))
        feature_records = [
            read_json_object(path) for path in sorted(self.paths.feature_records.glob("*.json"))
        ]
        feature_records.sort(key=lambda value: str(value.get("timeline_id", "")))

        # This is a write-time contract check, not a separate validation pass:
        # every run must have exactly one feature timeline on the same clock.
        timeline_contracts: dict[str, tuple[int, float, float]] = {}
        for record in run_records:
            timeline_id = str(record["timeline_id"])
            contract = (
                int(record["n_timepoints"]),
                float(record["tr_seconds"]),
                float(record.get("time_origin_seconds", 0.0)),
            )
            previous = timeline_contracts.setdefault(timeline_id, contract)
            if previous != contract:
                raise BundleValidationError(
                    f"Runs sharing timeline {timeline_id!r} disagree on length/clock"
                )
        seen_feature_timelines: set[str] = set()
        for record in feature_records:
            timeline_id = str(record["timeline_id"])
            if timeline_id in seen_feature_timelines:
                raise BundleValidationError(f"Duplicate feature timeline {timeline_id!r}")
            seen_feature_timelines.add(timeline_id)
            expected = timeline_contracts.get(timeline_id)
            actual = (
                int(record["n_timepoints"]),
                float(record["tr_seconds"]),
                float(record.get("time_origin_seconds", 0.0)),
            )
            if expected is None or actual != expected:
                raise BundleValidationError(
                    f"Feature timeline {timeline_id!r} does not match its run length/clock"
                )
        if seen_feature_timelines != set(timeline_contracts):
            raise BundleValidationError("Every run timeline must have exactly one feature timeline")

        atomic_write_jsonl(self.paths.runs_jsonl, run_records, overwrite=overwrite or self.paths.runs_jsonl.exists())
        atomic_write_jsonl(
            self.paths.features_jsonl,
            feature_records,
            overwrite=overwrite or self.paths.features_jsonl.exists(),
        )
        subjects = sorted({str(record["subject_id"]) for record in run_records})
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "layout_id": LAYOUT_ID,
            "dataset_id": self.dataset_id,
            "created_utc": utc_now(),
            "observation_space": OBSERVATION_SPACE,
            "brain_view": BRAIN_VIEW,
            "feature_view": FEATURE_VIEW,
            "n_parcels": N_PARCELS,
            "expected_run_count": expected_run_count,
            "run_count": len(run_records),
            "feature_timeline_count": len(feature_records),
            "subjects": subjects,
            "artifacts": {
                "runs": _artifact_record(self.paths.runs_jsonl, root=self.paths.root),
                "features": _artifact_record(self.paths.features_jsonl, root=self.paths.root),
            },
            "hashes_computed": False,
            "training_semantics": "offline_tmi_encoding",
            "prospective_ready": False,
            "provenance": _strict_json_copy(_without_hash_fields(provenance), label="bundle provenance"),
        }
        atomic_write_json(self.paths.bundle_json, manifest, overwrite=overwrite)
        return manifest


def _validate_manifest_and_records(
    paths: BundlePaths,
    manifest: Mapping[str, Any],
    *,
    verify_finite: bool,
) -> None:
    required = {
        "schema_version",
        "layout_id",
        "dataset_id",
        "created_utc",
        "observation_space",
        "brain_view",
        "feature_view",
        "n_parcels",
        "expected_run_count",
        "run_count",
        "feature_timeline_count",
        "subjects",
        "artifacts",
        "hashes_computed",
        "training_semantics",
        "prospective_ready",
        "provenance",
    }
    missing = required - set(manifest)
    if missing:
        raise BundleValidationError(f"Bundle manifest is missing {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION or isinstance(manifest["schema_version"], bool):
        raise BundleValidationError(f"Expected schema_version={SCHEMA_VERSION}")
    if manifest["layout_id"] != LAYOUT_ID:
        raise BundleValidationError(f"Expected layout_id={LAYOUT_ID!r}")
    if manifest["dataset_id"] not in SUPPORTED_DATASETS:
        raise BundleValidationError("Bundle has an unsupported dataset_id")
    if manifest["observation_space"] != OBSERVATION_SPACE:
        raise BundleValidationError("Bundle has the wrong observation space")
    if manifest["brain_view"] != BRAIN_VIEW or manifest["feature_view"] != FEATURE_VIEW:
        raise BundleValidationError("Bundle has the wrong data views")
    if manifest["n_parcels"] != N_PARCELS:
        raise BundleValidationError(f"Bundle must declare exactly {N_PARCELS} parcels")
    if manifest["hashes_computed"] is not False:
        raise BundleValidationError("Active Schaefer bundles must declare hashes_computed=false")
    if manifest["training_semantics"] != "offline_tmi_encoding":
        raise BundleValidationError("Bundle training_semantics must be offline_tmi_encoding")
    if manifest["prospective_ready"] is not False:
        raise BundleValidationError("Active Schaefer bundles are not prospective-ready")
    _assert_no_hash_fields(
        manifest,
        label="bundle manifest",
        allow_root_marker=True,
    )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"runs", "features"}:
        raise BundleValidationError("Bundle artifacts must contain runs and features")
    for name, expected_path in (("runs", paths.runs_jsonl), ("features", paths.features_jsonl)):
        record = artifacts[name]
        if not isinstance(record, Mapping):
            raise BundleValidationError(f"Artifact {name!r} must be an object")
        actual_path = contained_path(paths.root, record.get("path"), must_exist=True, kind="file")
        if actual_path != expected_path.resolve():
            raise BundleValidationError(f"Artifact {name!r} is at the wrong path")
        stat = actual_path.stat()
        if stat.st_size != record.get("size_bytes"):
            raise BundleValidationError(f"Artifact {name!r} has the wrong size")
        if int(stat.st_mtime_ns) != record.get("mtime_ns"):
            raise BundleValidationError(f"Artifact {name!r} has the wrong mtime")
        _assert_no_hash_fields(record, label=f"artifact {name}")

    runs = read_jsonl(paths.runs_jsonl, allow_empty=False)
    if manifest["run_count"] != len(runs) or manifest["expected_run_count"] != len(runs):
        raise BundleValidationError("Bundle run counts do not match runs.jsonl")
    seen_runs: set[str] = set()
    seen_subjects: set[str] = set()
    timelines: set[str] = set()
    timeline_contracts: dict[str, tuple[int, float, float]] = {}
    for record in runs:
        _assert_no_hash_fields(record, label="run record")
        run_id = _validate_id(record.get("run_id"), label="run_id")
        if run_id in seen_runs:
            raise BundleValidationError(f"Duplicate run_id {run_id!r}")
        seen_runs.add(run_id)
        seen_subjects.add(_validate_id(record.get("subject_id"), label="subject_id"))
        timeline_id = _validate_id(record.get("timeline_id"), label="timeline_id")
        timelines.add(timeline_id)
        if record.get("observation_space") != OBSERVATION_SPACE:
            raise BundleValidationError(f"Run {run_id!r} has the wrong observation space")
        tr = record.get("tr_seconds")
        tr = _validate_dataset_tr(str(manifest["dataset_id"]), tr, label=f"Run {run_id!r}")
        brain_views = record.get("brain_views")
        if not isinstance(brain_views, Mapping) or set(brain_views) != {BRAIN_VIEW}:
            raise BundleValidationError(f"Run {run_id!r} must contain only {BRAIN_VIEW}")
        brain_shape = _validate_npy_record(
            paths.root,
            brain_views[BRAIN_VIEW],
            label=f"{run_id}/{BRAIN_VIEW}",
            expected_dtype="float32",
            expected_shape=(None, N_PARCELS),
            verify_finite=verify_finite,
        )
        if brain_shape[0] != record.get("n_timepoints"):
            raise BundleValidationError(f"Run {run_id!r} has inconsistent time length")
        masks = record.get("masks")
        if not isinstance(masks, Mapping) or set(masks) != REQUIRED_MASKS:
            raise BundleValidationError(f"Run {run_id!r} has the wrong mask set")
        mask_values: dict[str, np.ndarray] = {}
        for name in sorted(REQUIRED_MASKS):
            _validate_npy_record(
                paths.root,
                masks[name],
                label=f"{run_id}/{name}",
                expected_dtype="bool",
                expected_shape=(brain_shape[0],),
                verify_finite=False,
            )
            mask_path = contained_path(
                paths.root,
                masks[name]["path"],
                must_exist=True,
                kind="file",
                suffix=".npy",
            )
            mask_values[name] = np.load(mask_path, allow_pickle=False, mmap_mode="r")
        try:
            valid = mask_values["valid_time"]
            if np.any(valid & ~mask_values["stimulus_support"]):
                raise BundleValidationError(
                    f"Run {run_id!r} has valid rows outside stimulus support"
                )
            if np.any(valid & mask_values["nonsteady"]):
                raise BundleValidationError(f"Run {run_id!r} marks nonsteady rows valid")
            if np.any(valid & ~mask_values["motion_qc"]):
                raise BundleValidationError(f"Run {run_id!r} marks motion-censored rows valid")
        finally:
            for mask in mask_values.values():
                mmap = getattr(mask, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        contract = (brain_shape[0], float(tr), float(record.get("time_origin_seconds", 0.0)))
        previous = timeline_contracts.setdefault(timeline_id, contract)
        if previous != contract:
            raise BundleValidationError(
                f"Runs sharing timeline {timeline_id!r} disagree on length or clock"
            )
    if sorted(seen_subjects) != manifest["subjects"]:
        raise BundleValidationError("Manifest subject list differs from runs.jsonl")

    features = read_jsonl(paths.features_jsonl, allow_empty=True)
    if manifest["feature_timeline_count"] != len(features):
        raise BundleValidationError("Feature timeline count differs from features.jsonl")
    seen_timelines: set[str] = set()
    for record in features:
        _assert_no_hash_fields(record, label="feature record")
        timeline_id = _validate_id(record.get("timeline_id"), label="timeline_id")
        if timeline_id in seen_timelines:
            raise BundleValidationError(f"Duplicate feature timeline {timeline_id!r}")
        seen_timelines.add(timeline_id)
        if timeline_id not in timelines:
            raise BundleValidationError(f"Feature timeline {timeline_id!r} is not referenced by a run")
        if record.get("feature_view") != FEATURE_VIEW or record.get("normalization") != "none":
            raise BundleValidationError(f"Feature timeline {timeline_id!r} violates the raw-view contract")
        modalities = record.get("modalities")
        expected_modalities = DATASET_MODALITIES[str(manifest["dataset_id"])]
        if not isinstance(modalities, Mapping) or set(modalities) != expected_modalities:
            raise BundleValidationError(f"Feature timeline {timeline_id!r} has invalid modalities")
        expected_t = record.get("n_timepoints")
        expected_contract = timeline_contracts[timeline_id]
        feature_tr = record.get("tr_seconds")
        feature_origin = record.get("time_origin_seconds")
        try:
            feature_tr_value = _validate_dataset_tr(
                str(manifest["dataset_id"]), feature_tr, label=f"Feature timeline {timeline_id!r}"
            )
        except BundleValidationError:
            raise
        if (
            expected_t != expected_contract[0]
            or not np.isclose(feature_tr_value, expected_contract[1], rtol=0.0, atol=1e-8)
            or not isinstance(feature_origin, (int, float))
            or isinstance(feature_origin, bool)
            or not np.isclose(float(feature_origin), expected_contract[2], rtol=0.0, atol=1e-8)
        ):
            raise BundleValidationError(
                f"Feature timeline {timeline_id!r} does not match its run length/clock"
            )
        for modality, array_record in modalities.items():
            _validate_npy_record(
                paths.root,
                array_record,
                label=f"{timeline_id}/{modality}",
                expected_dtype="float32",
                expected_shape=(expected_t, N_LAYER_GROUPS, MODALITY_DIMS[modality]),
                verify_finite=verify_finite,
            )
    if seen_timelines != timelines:
        missing = sorted(timelines - seen_timelines)
        extra = sorted(seen_timelines - timelines)
        raise BundleValidationError(
            f"Feature timelines must match referenced run timelines: missing={missing}, extra={extra}"
        )


def validate_bundle(
    root: str | Path,
    *,
    verify_finite: bool = True,
    expected_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Validate a finalized v2 bundle and return its manifest."""

    paths = BundlePaths.from_root(root)
    manifest = read_json_object(paths.bundle_json)
    if expected_dataset_id is not None and manifest.get("dataset_id") != expected_dataset_id:
        raise BundleValidationError(
            f"Expected dataset_id={expected_dataset_id!r}, found {manifest.get('dataset_id')!r}"
        )
    _validate_manifest_and_records(
        paths,
        manifest,
        verify_finite=verify_finite,
    )
    return manifest


def committed_run_ids(root: str | Path) -> set[str]:
    """Return run IDs whose commit records exist (useful for safe resume)."""

    paths = BundlePaths.from_root(root)
    result: set[str] = set()
    if not paths.run_records.exists():
        return result
    for path in sorted(paths.run_records.glob("*.json")):
        record = read_json_object(path, required_keys={"run_id"})
        result.add(_validate_id(record["run_id"], label="run_id"))
    return result


__all__ = [
    "BRAIN_VIEW",
    "DATASET_MODALITIES",
    "FEATURE_VIEW",
    "LAYOUT_ID",
    "N_LAYER_GROUPS",
    "N_PARCELS",
    "MODALITY_DIMS",
    "OBSERVATION_SPACE",
    "REQUIRED_MASKS",
    "SCHEMA_VERSION",
    "SUPPORTED_DATASETS",
    "SUPPORTED_MODALITIES",
    "select_valid_time",
    "BundlePaths",
    "BundleWriter",
    "committed_run_ids",
    "validate_bundle",
]
