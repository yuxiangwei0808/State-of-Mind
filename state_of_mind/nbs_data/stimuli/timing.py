"""In-memory raw-stimulus contracts and native-TR pooling.

The raw feature extractor emits regular 2-Hz samples.  This module validates
those samples, removes overlapping context rows deterministically, and pools
sample centres into the native fMRI clock.  It intentionally has no disk
writer/loader: raw chunks are the persisted boundary and native timelines are
assembled in memory by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
STIMULUS_RECIPE = "native_stimulus_v1"
RAW_FREQUENCY_HZ = 2.0
EXPECTED_LAYERS = 2
MODALITY_DIMS: dict[str, int] = {
    "video": 1408,
    "audio": 1024,
    "text": 3072,
}
DATASET_TR_SECONDS = {"algonauts": 1.49, "hcp": 1.0}
ALL_MODALITIES = tuple(MODALITY_DIMS)
REQUIRED_MODALITIES = ("video", "audio")
ALGONAUTS_MODALITIES = ("video", "audio", "text")
NON_ALGONAUTS_MODALITIES = ("video", "audio")

_POST_TRANSFORMER_DIM = 1152
_TIMESTAMP_TOLERANCE = 1e-7
_SAMPLE_SPACING_TOLERANCE = 1e-6
_DIGEST_KEY_TOKENS = frozenset(
    {
        "sha", "sha1", "sha224", "sha256", "sha384", "sha512",
        "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum",
        "md5", "md5sum",
        "hash", "hashes", "checksum", "checksums", "digest", "digests",
    }
)


class StimulusError(ValueError):
    """Base error raised for invalid stimulus inputs."""


class StimulusValidationError(StimulusError):
    """Raised when a chunk or pooled result violates its contract."""


def _normalise_modality(modality: str) -> str:
    if not isinstance(modality, str) or not modality.strip():
        raise StimulusValidationError("modality must be a non-empty string")
    value = modality.strip().lower()
    value = {"visual": "video", "vision": "video", "auditory": "audio", "language": "text"}.get(value, value)
    if value not in MODALITY_DIMS:
        raise StimulusValidationError(f"Unsupported modality {modality!r}; expected {sorted(MODALITY_DIMS)}")
    return value


def _normalise_dataset(dataset: str) -> str:
    if not isinstance(dataset, str) or not dataset.strip():
        raise StimulusValidationError("dataset must be a non-empty string")
    value = dataset.strip().lower().replace("-", "_").replace(" ", "_")
    value = {
        "algonaut": "algonauts", "algonauts_2025": "algonauts",
        "hcp_movie": "hcp", "hcp_movies": "hcp", "hcp_7t": "hcp",
    }.get(value, value)
    if value not in DATASET_TR_SECONDS:
        raise StimulusValidationError(
            "Unsupported dataset {!r}; expected Algonauts or HCP".format(dataset)
        )
    return value


def required_modalities(dataset: str) -> tuple[str, ...]:
    _normalise_dataset(dataset)
    return REQUIRED_MODALITIES


def allowed_modalities(dataset: str) -> tuple[str, ...]:
    return ALGONAUTS_MODALITIES if _normalise_dataset(dataset) == "algonauts" else NON_ALGONAUTS_MODALITIES


def validate_dataset_modalities(
    modalities: Mapping[str, Any] | Iterable[str],
    dataset: str,
    *,
    require_text: bool = False,
) -> tuple[str, ...]:
    canonical_dataset = _normalise_dataset(dataset)
    names: list[str] = []
    for value in modalities.keys() if isinstance(modalities, Mapping) else modalities:
        name = _normalise_modality(str(value))
        if name in names:
            raise StimulusValidationError(f"Duplicate modality {name!r}")
        names.append(name)
    unsupported = sorted(set(names) - set(allowed_modalities(canonical_dataset)))
    if unsupported:
        raise StimulusValidationError(f"Dataset {canonical_dataset!r} does not permit modalities {unsupported}")
    missing = sorted(set(REQUIRED_MODALITIES) - set(names))
    if missing:
        raise StimulusValidationError(f"Dataset {canonical_dataset!r} is missing required modalities {missing}")
    if require_text and canonical_dataset == "algonauts" and "text" not in names:
        raise StimulusValidationError("Strict Algonauts stimulus requires text modality")
    return tuple(sorted(names, key=ALL_MODALITIES.index))


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    candidate = value
    if callable(getattr(candidate, "detach", None)):
        candidate = candidate.detach()
    if callable(getattr(candidate, "cpu", None)):
        candidate = candidate.cpu()
    if callable(getattr(candidate, "numpy", None)):
        candidate = candidate.numpy()
    try:
        array = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise StimulusValidationError(f"{name} is not array-like: {exc}") from exc
    if not np.issubdtype(array.dtype, np.floating):
        raise StimulusValidationError(f"{name} must have a floating dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise StimulusValidationError(f"{name} contains non-finite values")
    return array


def _as_float_timestamps(value: Any, *, name: str, expected: int | None = None) -> np.ndarray:
    try:
        timestamps = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise StimulusValidationError(f"{name} is not array-like: {exc}") from exc
    if timestamps.ndim != 1:
        raise StimulusValidationError(f"{name} must be one-dimensional")
    if expected is not None and timestamps.size != expected:
        raise StimulusValidationError(f"{name} has {timestamps.size} values but expected {expected}")
    if not np.issubdtype(timestamps.dtype, np.number):
        raise StimulusValidationError(f"{name} must be numeric")
    timestamps = timestamps.astype(np.float64, copy=False)
    if not np.all(np.isfinite(timestamps)):
        raise StimulusValidationError(f"{name} contains non-finite values")
    return timestamps


def _as_support(value: Any | None, *, expected: int) -> np.ndarray:
    if value is None:
        return np.ones(expected, dtype=bool)
    try:
        support = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise StimulusValidationError(f"support is not array-like: {exc}") from exc
    if support.ndim != 1 or support.size != expected:
        raise StimulusValidationError(f"support must have shape [{expected}]")
    if support.dtype == np.bool_:
        return support.astype(bool, copy=False)
    if not np.issubdtype(support.dtype, np.number):
        raise StimulusValidationError("support must be boolean or numeric 0/1")
    support_float = support.astype(np.float64, copy=False)
    if not np.all(np.isfinite(support_float)) or not np.all(np.isin(support_float, (0.0, 1.0))):
        raise StimulusValidationError("support must contain only finite 0/1 values")
    return support_float.astype(bool)


def _is_digest_key(key: Any) -> bool:
    """Recognize retired digest fields without treating ``shape`` as one."""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key).strip()).lower()
    parts = [part for part in re.split(r"[^a-z0-9]+", text) if part]
    return any(part in _DIGEST_KEY_TOKENS for part in parts)


def _without_digest_fields(value: Any) -> Any:
    """Strip legacy digest-like metadata without inspecting array contents."""

    if isinstance(value, Mapping):
        return {
            key: _without_digest_fields(child)
            for key, child in value.items()
            if not _is_digest_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_digest_fields(child) for child in value]
    return value


@dataclass
class ModalityChunk:
    modality: str
    values: Any
    timestamps: Any
    support: Any | None = None
    source_id: str = ""
    source_path: str | Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_rate_hz: float = RAW_FREQUENCY_HZ

    def __post_init__(self) -> None:
        self.modality = _normalise_modality(self.modality)
        expected_dim = MODALITY_DIMS[self.modality]
        values = _to_numpy(self.values, name=f"{self.modality} values")
        if values.ndim != 3:
            raise StimulusValidationError(
                f"{self.modality} values must have shape [time, {EXPECTED_LAYERS}, {expected_dim}], got {tuple(values.shape)}"
            )
        if values.shape[1] != EXPECTED_LAYERS:
            raise StimulusValidationError(
                f"{self.modality} layer axis must have length {EXPECTED_LAYERS}, got {values.shape[1]}"
            )
        if values.shape[2] != expected_dim:
            raise StimulusValidationError(
                f"{self.modality} feature dimension must be {expected_dim}, got {values.shape[2]}"
            )
        if values.shape[0] == 0:
            raise StimulusValidationError("A modality chunk cannot be empty")
        timestamps = _as_float_timestamps(self.timestamps, name=f"{self.modality} timestamps", expected=values.shape[0])
        if np.any(np.diff(timestamps) <= 0):
            raise StimulusValidationError(f"{self.modality} timestamps must be strictly increasing within a chunk")
        support = _as_support(self.support, expected=values.shape[0])
        try:
            rate = float(self.sample_rate_hz)
        except (TypeError, ValueError) as exc:
            raise StimulusValidationError("sample_rate_hz must be numeric") from exc
        if not math.isfinite(rate) or rate <= 0 or not math.isclose(rate, RAW_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-6):
            raise StimulusValidationError(f"Raw modality chunks must be at {RAW_FREQUENCY_HZ:g} Hz, got {rate:g}")
        if timestamps.size > 1 and not np.allclose(np.diff(timestamps), 1.0 / rate, rtol=0.0, atol=_SAMPLE_SPACING_TOLERANCE):
            raise StimulusValidationError("Raw modality timestamps must use regular sample centres spaced 0.5 seconds at 2 Hz")
        if not isinstance(self.metadata, Mapping):
            raise StimulusValidationError("metadata must be a mapping")
        try:
            json.dumps(dict(self.metadata), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StimulusValidationError("metadata must be JSON-serialisable") from exc
        self.values = np.ascontiguousarray(values.astype(np.float32, copy=False))
        self.timestamps = np.ascontiguousarray(timestamps)
        self.support = np.ascontiguousarray(support)
        self.sample_rate_hz = rate
        self.source_id = str(self.source_id)
        self.metadata = _without_digest_fields(dict(self.metadata))

    @classmethod
    def from_mapping(cls, modality: str, value: Mapping[str, Any]) -> "ModalityChunk":
        if not isinstance(value, Mapping) or "values" not in value or "timestamps" not in value:
            raise StimulusValidationError("Chunk mapping requires values and timestamps")
        return cls(
            modality=modality, values=value["values"], timestamps=value["timestamps"],
            support=value.get("support"), source_id=str(value.get("source_id", value.get("id", ""))),
            source_path=value.get("source_path"), metadata=value.get("metadata", {}),
            sample_rate_hz=value.get("sample_rate_hz", RAW_FREQUENCY_HZ),
        )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    def provenance(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": None if self.source_path is None else str(self.source_path),
            "n_samples": self.n_samples,
            "supported_samples": int(self.support.sum()),
            "timestamps_start_seconds": float(self.timestamps[0]),
            "timestamps_stop_seconds": float(self.timestamps[-1]),
            "sample_rate_hz": self.sample_rate_hz,
            "metadata": _without_digest_fields(self.metadata),
        }


@dataclass
class StitchedModality:
    modality: str
    values: np.ndarray
    timestamps: np.ndarray
    support: np.ndarray
    source_ids: tuple[str, ...] = ()
    duplicate_tolerance_seconds: float = _TIMESTAMP_TOLERANCE

    def __post_init__(self) -> None:
        self.modality = _normalise_modality(self.modality)
        expected_dim = MODALITY_DIMS[self.modality]
        values = _to_numpy(self.values, name=f"stitched {self.modality} values")
        if values.ndim != 3 or values.shape[1:] != (EXPECTED_LAYERS, expected_dim):
            raise StimulusValidationError(f"stitched values must have shape [time, {EXPECTED_LAYERS}, {expected_dim}]")
        timestamps = _as_float_timestamps(self.timestamps, name="stitched timestamps", expected=values.shape[0])
        support = _as_support(self.support, expected=values.shape[0])
        if values.shape[0] and np.any(np.diff(timestamps) <= 0):
            raise StimulusValidationError("stitched timestamps must be strictly increasing")
        tolerance = float(self.duplicate_tolerance_seconds)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise StimulusValidationError("duplicate timestamp tolerance must be finite/non-negative")
        if self.source_ids and len(self.source_ids) != values.shape[0]:
            raise StimulusValidationError("source_ids must align with stitched samples")
        self.values = np.ascontiguousarray(values.astype(np.float32, copy=False))
        self.timestamps = np.ascontiguousarray(timestamps)
        self.support = np.ascontiguousarray(support)
        self.source_ids = tuple(str(value) for value in self.source_ids)
        self.duplicate_tolerance_seconds = tolerance

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])


@dataclass
class PooledModality:
    modality: str
    values: np.ndarray
    support: np.ndarray
    source_counts: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = _normalise_modality(self.modality)
        values = _to_numpy(self.values, name=f"pooled {self.modality} values")
        if values.ndim != 3:
            raise StimulusValidationError(
                f"pooled {self.modality} values must be three-dimensional"
            )
        expected = (int(values.shape[0]), EXPECTED_LAYERS, MODALITY_DIMS[self.modality])
        if values.shape != expected:
            raise StimulusValidationError(f"pooled {self.modality} values must have shape {expected}, got {values.shape}")
        support = _as_support(self.support, expected=expected[0])
        source_counts = np.asarray(self.source_counts)
        if source_counts.ndim != 1 or source_counts.size != expected[0] or not np.issubdtype(source_counts.dtype, np.integer):
            raise StimulusValidationError("source_counts must be an integer vector aligned with values")
        source_counts = source_counts.astype(np.int64, copy=False)
        if np.any(source_counts < 0) or np.any(support != (source_counts > 0)):
            raise StimulusValidationError("support must equal source_counts > 0 and counts cannot be negative")
        if not isinstance(self.provenance, Mapping):
            raise StimulusValidationError("provenance must be a mapping")
        try:
            json.dumps(dict(self.provenance), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StimulusValidationError(
                "pooled provenance must be JSON-serialisable"
            ) from exc
        self.values = np.ascontiguousarray(values.astype(np.float32, copy=False))
        self.support = np.ascontiguousarray(support)
        self.source_counts = np.ascontiguousarray(source_counts)
        self.provenance = _without_digest_fields(dict(self.provenance))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.values.shape)


@dataclass
class NativeStimulusTimeline:
    dataset: str
    timeline_id: str
    origin_seconds: float
    native_tr_seconds: float
    modalities: Mapping[str, PooledModality]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = _normalise_dataset(self.dataset)
        if not isinstance(self.timeline_id, str) or not self.timeline_id.strip():
            raise StimulusValidationError("timeline_id must be a non-empty string")
        self.timeline_id = self.timeline_id.strip()
        self.origin_seconds = float(self.origin_seconds)
        self.native_tr_seconds = float(self.native_tr_seconds)
        if not math.isfinite(self.origin_seconds) or not math.isfinite(self.native_tr_seconds) or self.native_tr_seconds <= 0:
            raise StimulusValidationError("origin_seconds/native_tr_seconds must be finite; TR must be positive")
        expected_tr = DATASET_TR_SECONDS[self.dataset]
        if not math.isclose(self.native_tr_seconds, expected_tr, rel_tol=0.0, abs_tol=1e-8):
            raise StimulusValidationError(f"{self.dataset} native TR is fixed at {expected_tr} seconds; received {self.native_tr_seconds}")
        if not isinstance(self.modalities, Mapping) or not self.modalities:
            raise StimulusValidationError("modalities must be a non-empty mapping")
        canonical: dict[str, PooledModality] = {}
        for name, value in self.modalities.items():
            modality = _normalise_modality(name)
            if modality in canonical or not isinstance(value, PooledModality) or value.modality != modality:
                raise StimulusValidationError("modalities must contain one matching PooledModality per key")
            canonical[modality] = value
        validate_dataset_modalities(canonical, self.dataset)
        if len({value.values.shape[0] for value in canonical.values()}) != 1:
            raise StimulusValidationError("All modality timelines must have the same number of bins")
        if not isinstance(self.provenance, Mapping):
            raise StimulusValidationError("timeline provenance must be a mapping")
        try:
            json.dumps(dict(self.provenance), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StimulusValidationError(
                "timeline provenance must be JSON-serialisable"
            ) from exc
        self.modalities = {key: canonical[key] for key in sorted(canonical, key=ALL_MODALITIES.index)}
        self.provenance = _without_digest_fields(dict(self.provenance))

    @property
    def n_bins(self) -> int:
        return int(next(iter(self.modalities.values())).values.shape[0])


def extract_batch_data(batch: Any, modality: str | None = None, *, batch_index: int = 0,
                       expected_layers: int = EXPECTED_LAYERS, expected_dim: int | None = None) -> np.ndarray:
    """Convert NeuralSet ``[B,L,D,T]`` or ``[L,D,T]`` data to ``[T,L,D]``."""
    data = getattr(batch, "data", batch)
    if isinstance(data, Mapping):
        if modality is None:
            if len(data) != 1:
                raise StimulusValidationError("modality is required when batch.data contains multiple keys")
            key = next(iter(data))
        else:
            key = modality
            if key not in data:
                canonical = _normalise_modality(key)
                matches = [candidate for candidate in data if _normalise_modality(str(candidate)) == canonical]
                if not matches:
                    raise KeyError(f"Batch data has no modality {modality!r}")
                key = matches[0]
        values = data[key]
        modality_name = _normalise_modality(str(key))
    else:
        values = data
        modality_name = None if modality is None else _normalise_modality(modality)
    array = _to_numpy(values, name="batch modality")
    if array.ndim == 4:
        if not isinstance(batch_index, (int, np.integer)) or not 0 <= int(batch_index) < array.shape[0]:
            raise StimulusValidationError(f"batch_index must be in [0, {array.shape[0]})")
        if array.shape[1] != expected_layers:
            raise StimulusValidationError(f"Raw modality layer axis must have length {expected_layers}, got {array.shape[1]}")
        if expected_dim is None and modality_name is not None:
            expected_dim = MODALITY_DIMS[modality_name]
        if expected_dim is not None and array.shape[2] != expected_dim:
            raise StimulusValidationError(f"{modality_name or 'modality'} raw feature dimension must be {expected_dim}, got {array.shape[2]}")
        output = np.transpose(array[int(batch_index)], (2, 0, 1))
    elif array.ndim == 3:
        if array.shape[-1] == _POST_TRANSFORMER_DIM:
            raise StimulusValidationError("Post-Transformer hidden features [B,T,1152] are not raw modality data")
        if array.shape[0] != expected_layers:
            raise StimulusValidationError("Unbatched raw modality data must have shape [2,D,T]")
        if expected_dim is None and modality_name is not None:
            expected_dim = MODALITY_DIMS[modality_name]
        if expected_dim is not None and array.shape[1] != expected_dim:
            raise StimulusValidationError(f"{modality_name or 'modality'} raw feature dimension must be {expected_dim}, got {array.shape[1]}")
        output = np.transpose(array, (2, 0, 1))
    else:
        raise StimulusValidationError("Raw modality data must have shape [B,L,D,T] or [L,D,T]")
    return np.ascontiguousarray(output.astype(np.float32, copy=False))


def _coerce_chunk(modality: str, value: Any) -> ModalityChunk:
    if isinstance(value, ModalityChunk):
        if value.modality != _normalise_modality(modality):
            raise StimulusValidationError("Chunk modality does not match mapping key")
        return value
    if isinstance(value, Mapping):
        return ModalityChunk.from_mapping(modality, value)
    if isinstance(value, (tuple, list)) and len(value) in {2, 3}:
        return ModalityChunk(modality, value[0], value[1], None if len(value) == 2 else value[2])
    raise StimulusValidationError("Each chunk must be ModalityChunk, mapping, or (values, timestamps[, support])")


def stitch_modality_chunks(modality: str, chunks: Iterable[ModalityChunk | Mapping[str, Any] | Sequence[Any]], *,
                           duplicate_tolerance_seconds: float = _TIMESTAMP_TOLERANCE) -> StitchedModality:
    canonical = _normalise_modality(modality)
    tolerance = float(duplicate_tolerance_seconds)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise StimulusValidationError("duplicate timestamp tolerance must be finite/non-negative")
    materialized = [_coerce_chunk(canonical, value) for value in chunks]
    if not materialized:
        raise StimulusValidationError(f"No chunks supplied for modality {canonical!r}")
    indexed = list(enumerate(materialized))
    indexed.sort(key=lambda item: (float(item[1].timestamps[0]), item[1].source_id,
                                   "" if item[1].source_path is None else str(item[1].source_path),
                                   item[1].n_samples, float(item[1].sample_rate_hz), item[0]))
    candidates: list[tuple[float, int, int, ModalityChunk]] = []
    for rank, (_, chunk) in enumerate(indexed):
        for row, timestamp in enumerate(chunk.timestamps):
            candidates.append((float(timestamp), rank, row, chunk))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    duplicate_groups: list[list[tuple[float, int, int, ModalityChunk]]] = []
    for candidate in candidates:
        if (
            not duplicate_groups
            or candidate[0] - duplicate_groups[-1][0][0] > tolerance
        ):
            duplicate_groups.append([candidate])
        else:
            duplicate_groups[-1].append(candidate)
    rows: list[tuple[float, np.ndarray, bool, str]] = []
    for copies in duplicate_groups:
        copies.sort(key=lambda item: (not bool(item[3].support[item[2]]), item[1], item[2]))
        timestamp, _, row, chunk = copies[0]
        supported = bool(chunk.support[row])
        value = chunk.values[row].copy()
        if not supported:
            value.fill(0.0)
        rows.append((timestamp, value, supported, chunk.source_id))
    rows.sort(key=lambda value: value[0])
    return StitchedModality(
        modality=canonical,
        values=np.stack([value[1] for value in rows]).astype(np.float32, copy=False),
        timestamps=np.asarray([value[0] for value in rows], dtype=np.float64),
        support=np.asarray([value[2] for value in rows], dtype=bool),
        source_ids=tuple(value[3] for value in rows),
        duplicate_tolerance_seconds=tolerance,
    )


def pool_modality_to_bold(modality: str, stitched: StitchedModality, *, origin_seconds: float,
                          native_tr_seconds: float, n_bins: int) -> PooledModality:
    canonical = _normalise_modality(modality)
    if not isinstance(stitched, StitchedModality) or stitched.modality != canonical:
        raise StimulusValidationError("stitched must be a matching StitchedModality")
    if isinstance(n_bins, (bool, np.bool_)):
        raise StimulusValidationError("n_bins must be a positive integer")
    try:
        bins = int(n_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StimulusValidationError("n_bins must be a positive integer") from exc
    if bins <= 0 or bins != n_bins:
        raise StimulusValidationError("n_bins must be a positive integer")
    origin, tr = float(origin_seconds), float(native_tr_seconds)
    if not math.isfinite(origin) or not math.isfinite(tr) or tr <= 0:
        raise StimulusValidationError("origin_seconds/native_tr_seconds must be finite; n_bins must be positive")
    values = np.zeros((bins, EXPECTED_LAYERS, MODALITY_DIMS[canonical]), dtype=np.float32)
    source_counts = np.zeros(bins, dtype=np.int64)
    indices = np.floor((stitched.timestamps - origin) / tr).astype(np.int64)
    for row, index in enumerate(indices):
        if 0 <= index < bins and stitched.support[row]:
            values[index] += stitched.values[row]
            source_counts[index] += 1
    nonempty = source_counts > 0
    values[nonempty] /= source_counts[nonempty, None, None].astype(np.float32)
    provenance = {
        "recipe": STIMULUS_RECIPE,
        "pooling": "sample_center_half_open_arithmetic_mean",
        "raw_frequency_hz": RAW_FREQUENCY_HZ,
        "origin_seconds": origin,
        "native_tr_seconds": tr,
        "n_bins": bins,
        "stitched_samples": stitched.n_samples,
        "stitched_supported_samples": int(stitched.support.sum()),
        "interpolation": False,
        "offline_normalization": False,
        "hemodynamic_lag_seconds": None,
        "source_ids": sorted(set(stitched.source_ids)),
    }
    return PooledModality(canonical, values, nonempty, source_counts, provenance)


def build_native_timeline(*, dataset: str, timeline_id: str,
                          chunks: Mapping[str, Iterable[ModalityChunk | Mapping[str, Any] | Sequence[Any]]],
                          origin_seconds: float, native_tr_seconds: float, n_bins: int,
                          duplicate_tolerance_seconds: float = _TIMESTAMP_TOLERANCE,
                          require_text: bool = False) -> NativeStimulusTimeline:
    canonical_dataset = _normalise_dataset(dataset)
    if not isinstance(chunks, Mapping):
        raise StimulusValidationError("chunks must be a mapping from modality to chunk iterables")
    names = validate_dataset_modalities(chunks, canonical_dataset, require_text=require_text)
    pooled: dict[str, PooledModality] = {}
    provenance: dict[str, Any] = {}
    for modality in names:
        raw_chunks = chunks.get(modality)
        if raw_chunks is None:
            raw_chunks = next((value for key, value in chunks.items() if _normalise_modality(str(key)) == modality), None)
        if isinstance(raw_chunks, (ModalityChunk, Mapping)):
            iterable: Iterable[Any] = (raw_chunks,)
        elif isinstance(raw_chunks, tuple) and len(raw_chunks) in {2, 3} and not isinstance(raw_chunks[0], (ModalityChunk, Mapping, list, tuple)):
            iterable = (raw_chunks,)
        else:
            iterable = raw_chunks
        materialized = [_coerce_chunk(modality, value) for value in iterable]
        stitched = stitch_modality_chunks(modality, materialized, duplicate_tolerance_seconds=duplicate_tolerance_seconds)
        pooled[modality] = pool_modality_to_bold(modality, stitched, origin_seconds=origin_seconds,
                                                  native_tr_seconds=native_tr_seconds, n_bins=n_bins)
        provenance[modality] = [value.provenance() for value in materialized]
    timeline_provenance = {
        "recipe": STIMULUS_RECIPE,
        "source_clock_hz": RAW_FREQUENCY_HZ,
        "axis_order": ["time", "layer", "feature"],
        "dataset": canonical_dataset,
        "timeline_id": timeline_id,
        "chunks": provenance,
        "duplicate_context_policy": "deterministic_supported_first_no_averaging",
        "pooling": "sample_center_half_open_arithmetic_mean",
        "offline_normalization": False,
        "hemodynamic_lag_seconds": None,
    }
    return NativeStimulusTimeline(canonical_dataset, timeline_id, origin_seconds, native_tr_seconds, pooled, timeline_provenance)


__all__ = [
    "ALGONAUTS_MODALITIES", "ALL_MODALITIES", "DATASET_TR_SECONDS", "EXPECTED_LAYERS",
    "MODALITY_DIMS", "NON_ALGONAUTS_MODALITIES", "PooledModality", "RAW_FREQUENCY_HZ",
    "REQUIRED_MODALITIES", "SCHEMA_VERSION", "STIMULUS_RECIPE", "StimulusError",
    "StimulusValidationError", "ModalityChunk", "StitchedModality", "NativeStimulusTimeline",
    "allowed_modalities", "build_native_timeline", "extract_batch_data",
    "pool_modality_to_bold", "required_modalities", "stitch_modality_chunks",
    "validate_dataset_modalities",
]
