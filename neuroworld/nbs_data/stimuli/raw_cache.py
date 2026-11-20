"""Read and export raw NeuralSet stimulus-cache chunks.

The offline stimulus contract is intentionally small: a cache adapter emits
raw, pre-projector modality samples at 2 Hz, together with physical sample
centres and a source-support mask.  The canonical in-memory representation is
``float32 [time, 2, feature]`` and is defined by
``neuroworld.nbs_data.stimuli.timing``. No projector, temporal Transformer,
post-Transformer representation, normalisation, or haemodynamic lag belongs
in this module.

The existing TRIBEv2/NeuralSet cache is an Exca directory containing
``*-info.jsonl`` files and binary ``data/*.data`` files.  Those files are
treated as read-only evidence.  Their index records are useful for locating
the values, but a direct import is accepted only when the indexed tensor is
already ``[time, 2, feature]``.  The released caches commonly store all 20
backbone layers; those must be loaded through an explicit NeuralSet extractor
that returns a ``[B, 2, D, T]`` batch.  We deliberately do not guess how an
upstream installation should construct its loader.

The JSONL format written here is self-contained relative to its index file and
contains one record per chunk. Every values/timestamps/support artifact is a
separate ``.npy`` file described by path, byte size, modification time, shape,
and dtype. Hashing is deliberately outside this one-off preparation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote
import uuid

import numpy as np

from neuroworld.nbs_data.stimuli.timing import (
    EXPECTED_LAYERS,
    MODALITY_DIMS,
    RAW_FREQUENCY_HZ,
    ModalityChunk,
    NativeStimulusTimeline,
    build_native_timeline,
    extract_batch_data,
)


RAW_CACHE_SCHEMA_VERSION = 1
RAW_CACHE_INDEX_NAME = "chunks.jsonl"
# Native clocks are properties of the source datasets, not free parameters of
# a cache request.  Keep the map here so a raw-cache caller constructing a
# NativeTimelineSpec directly cannot accidentally pool a timeline on a
# different clock.  ``hcp`` is the canonical name returned by
# ``_canonical_dataset`` for all HCP aliases.
NATIVE_TR_SECONDS = {
    "algonauts": 1.49,
    "hcp": 1.0,
}
_RETIRED_DIGEST_TOKENS = frozenset({"sha", "sha1", "sha224", "sha256", "sha384", "sha512", "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum", "md5", "md5sum", "hash", "hashes", "checksum", "checksums", "digest", "digests"})


class RawCacheError(ValueError):
    """Base error for malformed or unsupported raw-cache inputs."""


class RawCacheIntegrityError(RawCacheError):
    """Raised when a JSONL record or referenced artifact is tampered with."""


class RawCacheTransactionError(RuntimeError):
    """Raised when an atomic raw-cache export cannot be committed."""


def _canonical_modality(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RawCacheError("modality must be a non-empty string")
    aliases = {"visual": "video", "vision": "video", "auditory": "audio", "language": "text"}
    modality = aliases.get(value.strip().lower(), value.strip().lower())
    if modality not in MODALITY_DIMS:
        raise RawCacheError(
            f"Unsupported modality {value!r}; expected {sorted(MODALITY_DIMS)}"
        )
    return modality


def _canonical_dataset(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RawCacheError("dataset must be a non-empty string")
    dataset = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "algonaut": "algonauts",
        "hcp_movie": "hcp",
        "hcp_movies": "hcp",
        "hcp_7t": "hcp",
    }
    dataset = aliases.get(dataset, dataset)
    if dataset not in {"algonauts", "hcp"}:
        raise RawCacheError(
            f"Unsupported dataset {value!r}; expected algonauts or hcp"
        )
    return dataset


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RawCacheError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RawCacheError(f"{name} must be finite")
    return result


def sample_center_timestamps(
    start_seconds: float,
    n_samples: int,
    *,
    frequency_hz: float = RAW_FREQUENCY_HZ,
) -> np.ndarray:
    """Return physical sample-centre times for a regular raw chunk.

    A chunk beginning at ``start_seconds`` with frequency ``f`` has centres
    ``start_seconds + (i + 1/2) / f``.  The explicit half-sample convention is
    important when pooling into half-open native-TR bins.
    """

    start = _finite(start_seconds, name="start_seconds")
    try:
        count = int(n_samples)
    except (TypeError, ValueError) as exc:
        raise RawCacheError("n_samples must be an integer") from exc
    if isinstance(n_samples, bool) or count <= 0 or count != n_samples:
        raise RawCacheError("n_samples must be a positive integer")
    frequency = _finite(frequency_hz, name="frequency_hz")
    if frequency <= 0:
        raise RawCacheError("frequency_hz must be positive")
    return np.ascontiguousarray(
        start + (np.arange(count, dtype=np.float64) + 0.5) / frequency
    )


def _safe_json(value: Any, *, label: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RawCacheError(f"{label} must be JSON-serialisable without NaN/Inf") from exc
    return value


def _safe_relative_path(root: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RawCacheIntegrityError(f"{label} path must be a non-empty string")
    candidate = Path(raw_path)
    if candidate.is_absolute() or "\\" in raw_path or ".." in candidate.parts:
        raise RawCacheIntegrityError(f"{label} path must be relative and contained: {raw_path!r}")
    try:
        resolved = (root / candidate).resolve(strict=True)
    except OSError as exc:
        raise RawCacheIntegrityError(f"{label} artifact is missing: {raw_path!r}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RawCacheIntegrityError(f"{label} path escapes index root: {raw_path!r}") from exc
    if not resolved.is_file():
        raise RawCacheIntegrityError(f"{label} artifact is not a regular file: {resolved}")
    return resolved


def _source_path_provenance(source_path: str | Path | None, root: Path) -> str | None:
    """Encode original source provenance without confusing it with artifacts.

    Paths inside the exported root are kept relative so an exported cache can
    be relocated.  External paths remain absolute provenance strings; they are
    never used as load paths for the exported arrays.
    """

    if source_path is None:
        return None
    text = str(source_path).strip()
    if not text:
        return None
    candidate = Path(text).expanduser().resolve(strict=False)
    destination_root = root.expanduser().resolve(strict=False)
    try:
        return candidate.relative_to(destination_root).as_posix()
    except ValueError:
        return str(candidate)


def _raw_timeline_slug(timeline_id: str) -> str:
    """Return a reversible, contained directory name for a timeline ID."""

    normalized = str(timeline_id).strip()
    if not normalized:
        raise RawCacheError("timeline_id must be a non-empty string")
    slug = quote(normalized, safe="-_.~")
    # A suffix-less ``..``/``.`` would be interpreted as a path component by
    # ``Path``.  Encode dots for these two IDs while preserving reversibility.
    if slug in {".", ".."}:
        slug = quote(normalized, safe="-_~")
    path = Path(slug)
    if not slug or path.is_absolute() or path.name != slug or ".." in path.parts:
        raise RawCacheError(
            f"timeline_id cannot be represented as a contained path: {timeline_id!r}"
        )
    if len(slug.encode("utf-8")) > 240:
        raise RawCacheError(
            f"timeline_id is too long for a safe raw-cache path: {timeline_id!r}"
        )
    return slug


def _descriptor_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RawCacheIntegrityError(f"{label} must be a non-negative integer")
    return value


def _check_file_stats(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Validate the cheap structural identity recorded for one array."""

    try:
        stat = path.stat()
    except OSError as exc:
        raise RawCacheIntegrityError(f"Cannot stat {label} artifact {path}: {exc}") from exc
    expected_size = _descriptor_integer(
        descriptor.get("size_bytes"), label=f"{label}.size_bytes"
    )
    if stat.st_size != expected_size:
        raise RawCacheIntegrityError(
            f"{label} byte-size mismatch: index says {expected_size}, "
            f"file is {stat.st_size}: {path}"
        )
    expected_mtime = _descriptor_integer(
        descriptor.get("mtime_ns"), label=f"{label}.mtime_ns"
    )
    if stat.st_mtime_ns != expected_mtime:
        raise RawCacheIntegrityError(
            f"{label} mtime mismatch: index says {expected_mtime}, "
            f"file is {stat.st_mtime_ns}: {path}"
        )


def _read_array(path: Path, descriptor: Mapping[str, Any], *, label: str) -> np.ndarray:
    _check_file_stats(path, descriptor, label=label)
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RawCacheIntegrityError(f"Cannot load {label} artifact {path}: {exc}") from exc
    if "dtype" in descriptor and str(array.dtype) != str(descriptor["dtype"]):
        raise RawCacheIntegrityError(
            f"{label} dtype mismatch: index says {descriptor['dtype']!r}, file is {array.dtype}"
        )
    if "shape" in descriptor and list(array.shape) != list(descriptor["shape"]):
        raise RawCacheIntegrityError(
            f"{label} shape mismatch: index says {descriptor['shape']!r}, file is {list(array.shape)!r}"
        )
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise RawCacheIntegrityError(f"{label} contains non-finite values: {path}")
    return np.asarray(array)


@dataclass(frozen=True)
class RawCacheChunk:
    """One timeline-labelled chunk suitable for JSONL export."""

    timeline_id: str
    modality: str
    values: Any
    timestamps: Any
    support: Any | None = None
    source_id: str = ""
    source_path: str | Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_rate_hz: float = RAW_FREQUENCY_HZ
    chunk_index: int | None = None

    def __post_init__(self) -> None:
        timeline = str(self.timeline_id).strip()
        if not timeline:
            raise RawCacheError("timeline_id must be a non-empty string")
        modality = _canonical_modality(self.modality)
        source_path = None if self.source_path is None else str(self.source_path).strip() or None
        chunk = ModalityChunk(
            modality=modality,
            values=self.values,
            timestamps=self.timestamps,
            support=self.support,
            source_id=str(self.source_id),
            source_path=source_path,
            metadata=dict(self.metadata),
            sample_rate_hz=self.sample_rate_hz,
        )
        if self.chunk_index is not None:
            if isinstance(self.chunk_index, bool) or int(self.chunk_index) < 0 or int(self.chunk_index) != self.chunk_index:
                raise RawCacheError("chunk_index must be a non-negative integer")
        _safe_json(dict(self.metadata), label="metadata")
        object.__setattr__(self, "timeline_id", timeline)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "values", chunk.values)
        object.__setattr__(self, "timestamps", chunk.timestamps)
        object.__setattr__(self, "support", chunk.support)
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "metadata", dict(chunk.metadata))
        object.__setattr__(self, "sample_rate_hz", float(chunk.sample_rate_hz))

    @classmethod
    def from_chunk(
        cls,
        timeline_id: str,
        chunk: ModalityChunk,
        *,
        chunk_index: int | None = None,
    ) -> "RawCacheChunk":
        if not isinstance(chunk, ModalityChunk):
            raise RawCacheError("chunk must be a ModalityChunk")
        return cls(
            timeline_id=timeline_id,
            modality=chunk.modality,
            values=chunk.values,
            timestamps=chunk.timestamps,
            support=chunk.support,
            source_id=chunk.source_id,
            source_path=chunk.source_path,
            metadata=chunk.metadata,
            sample_rate_hz=chunk.sample_rate_hz,
            chunk_index=chunk_index,
        )

    def modality_chunk(self) -> ModalityChunk:
        return ModalityChunk(
            modality=self.modality,
            values=self.values,
            timestamps=self.timestamps,
            support=self.support,
            source_id=self.source_id,
            source_path=self.source_path,
            metadata=self.metadata,
            sample_rate_hz=self.sample_rate_hz,
        )


@dataclass(frozen=True)
class ChunkIndexRecord:
    """Validated metadata for one line in a raw-cache JSONL index."""

    timeline_id: str
    modality: str
    values: Mapping[str, Any]
    timestamps: Mapping[str, Any]
    support: Mapping[str, Any]
    n_samples: int
    shape: tuple[int, int, int]
    sample_rate_hz: float = RAW_FREQUENCY_HZ
    source_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    index_path: Path | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        timeline = str(self.timeline_id).strip()
        if not timeline:
            raise RawCacheIntegrityError("timeline_id must be non-empty")
        modality = _canonical_modality(self.modality)
        if not isinstance(self.values, Mapping) or not isinstance(self.timestamps, Mapping) or not isinstance(self.support, Mapping):
            raise RawCacheIntegrityError("values/timestamps/support must be descriptor objects")
        expected_dim = MODALITY_DIMS[modality]
        shape = tuple(int(v) for v in self.shape)
        if shape != (int(self.n_samples), EXPECTED_LAYERS, expected_dim):
            raise RawCacheIntegrityError(
                f"{timeline}/{modality}: shape must be [n,2,{expected_dim}], got {shape}"
            )
        if int(self.n_samples) <= 0 or int(self.n_samples) != self.n_samples:
            raise RawCacheIntegrityError("n_samples must be a positive integer")
        if int(self.chunk_index) < 0 or int(self.chunk_index) != self.chunk_index:
            raise RawCacheIntegrityError("chunk_index must be a non-negative integer")
        _finite(self.sample_rate_hz, name="sample_rate_hz")
        if float(self.sample_rate_hz) <= 0 or not math.isclose(float(self.sample_rate_hz), RAW_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-6):
            raise RawCacheIntegrityError("raw cache sample_rate_hz must be exactly 2 Hz")
        if self.source_path is not None and (
            not isinstance(self.source_path, str) or not self.source_path.strip()
        ):
            raise RawCacheIntegrityError("source_path must be a non-empty provenance string")
        _safe_json(dict(self.metadata), label="metadata")
        object.__setattr__(self, "timeline_id", timeline)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_path", self.source_path)
        object.__setattr__(self, "metadata", _without_checksum_fields(dict(self.metadata)))

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.timeline_id, self.modality, self.chunk_index

    def as_dict(self) -> dict[str, Any]:
        """Return the hash-free JSON representation of this index row."""

        return {
            "schema_version": RAW_CACHE_SCHEMA_VERSION,
            "record_type": "chunk",
            "timeline_id": self.timeline_id,
            "modality": self.modality,
            "chunk_index": self.chunk_index,
            "n_samples": self.n_samples,
            "shape": list(self.shape),
            "sample_rate_hz": self.sample_rate_hz,
            "values": dict(self.values),
            "timestamps": dict(self.timestamps),
            "support": dict(self.support),
            "source_id": self.source_id,
            "source_path": self.source_path,
            "metadata": _without_checksum_fields(dict(self.metadata)),
        }

    def load(self) -> RawCacheChunk:
        """Load and structurally validate the three referenced artifacts."""

        if self.index_path is None:
            raise RawCacheIntegrityError("Cannot load a detached ChunkIndexRecord without index_path")
        root = self.index_path.parent.resolve(strict=True)
        values_path = _safe_relative_path(root, self.values.get("path"), label="values")
        timestamps_path = _safe_relative_path(root, self.timestamps.get("path"), label="timestamps")
        support_path = _safe_relative_path(root, self.support.get("path"), label="support")
        values = _read_array(values_path, self.values, label="values")
        timestamps = _read_array(timestamps_path, self.timestamps, label="timestamps")
        support = _read_array(support_path, self.support, label="support")
        if values.shape != self.shape:
            raise RawCacheIntegrityError(f"{self.identity}: values shape does not match record")
        if timestamps.shape != (self.n_samples,) or support.shape != (self.n_samples,):
            raise RawCacheIntegrityError(f"{self.identity}: timestamps/support must align with values")
        try:
            return RawCacheChunk(
                timeline_id=self.timeline_id,
                modality=self.modality,
                values=values,
                timestamps=timestamps,
                support=support,
                source_id=self.source_id,
                # This is provenance for the original upstream cache, not the
                # exported values artifact.  Keep it untouched on import.
                source_path=self.source_path,
                metadata=self.metadata,
                sample_rate_hz=self.sample_rate_hz,
                chunk_index=self.chunk_index,
            )
        except RawCacheError:
            raise
        except Exception as exc:
            raise RawCacheIntegrityError(f"Invalid chunk {self.identity}: {exc}") from exc


def _coerce_export_chunk(value: Any, *, ordinal: int) -> RawCacheChunk:
    if isinstance(value, RawCacheChunk):
        if value.chunk_index is None:
            # Copy directly so an ordinal assignment cannot discard source
            # provenance that is not part of the older ModalityChunk class.
            return RawCacheChunk(
                timeline_id=value.timeline_id,
                modality=value.modality,
                values=value.values,
                timestamps=value.timestamps,
                support=value.support,
                source_id=value.source_id,
                source_path=value.source_path,
                metadata=value.metadata,
                sample_rate_hz=value.sample_rate_hz,
                chunk_index=ordinal,
            )
        return value
    if isinstance(value, ModalityChunk):
        raise RawCacheError(
            "A bare ModalityChunk has no timeline_id; provide RawCacheChunk or "
            "a mapping with timeline_id and modality"
        )
    if isinstance(value, Mapping):
        if "timeline_id" not in value or "modality" not in value:
            raise RawCacheError("Chunk mapping requires timeline_id and modality")
        timeline_id = str(value["timeline_id"])
        modality = str(value["modality"])
        if "batch" in value:
            return neuralset_batch_to_chunk(
                timeline_id,
                modality,
                value["batch"],
                start_seconds=value.get("start_seconds"),
                timestamps=value.get("timestamps"),
                support=value.get("support"),
                support_intervals=value.get("support_intervals"),
                events=value.get("events"),
                batch_index=int(value.get("batch_index", 0)),
                source_id=str(value.get("source_id", "")),
                source_path=value.get("source_path"),
                metadata=value.get("metadata", {}),
            )
        if "values" not in value or "timestamps" not in value:
            raise RawCacheError("Chunk mapping requires values/timestamps, or batch/start_seconds")
        return RawCacheChunk(
            timeline_id=timeline_id,
            modality=modality,
            values=value["values"],
            timestamps=value["timestamps"],
            support=value.get("support"),
            source_id=str(value.get("source_id", "")),
            source_path=value.get("source_path"),
            metadata=value.get("metadata", {}),
            sample_rate_hz=value.get("sample_rate_hz", RAW_FREQUENCY_HZ),
            chunk_index=value.get("chunk_index", ordinal),
        )
    if isinstance(value, (tuple, list)) and len(value) == 3:
        timeline_id, modality, chunk = value
        if isinstance(chunk, ModalityChunk):
            return RawCacheChunk.from_chunk(str(timeline_id), chunk, chunk_index=ordinal)
        if isinstance(chunk, Mapping):
            payload = dict(chunk)
            payload.update({"timeline_id": timeline_id, "modality": modality})
            return _coerce_export_chunk(payload, ordinal=ordinal)
    raise RawCacheError(
        "Extractor chunks must be RawCacheChunk, (timeline_id, modality, ModalityChunk), "
        "or an explicit mapping"
    )


def _without_checksum_fields(value: Any) -> Any:
    """Drop retired digest-like fields from imported source metadata."""

    def _is_checksum_key(key: Any) -> bool:
        text = str(key).strip()
        text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text).lower()
        parts = [part for part in re.split(r"[^a-z0-9]+", text) if part]
        if any(part in _RETIRED_DIGEST_TOKENS for part in parts):
            return True
        compact = "".join(parts)
        return compact in _RETIRED_DIGEST_TOKENS

    if isinstance(value, Mapping):
        return {
            key: _without_checksum_fields(child)
            for key, child in value.items()
            if not _is_checksum_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_checksum_fields(child) for child in value]
    return value


def _descriptor(
    path: Path,
    array: np.ndarray,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _reuse_or_write_array(path: Path, array: np.ndarray) -> None:
    """Reuse an exact staged array, otherwise replace it atomically."""

    value = np.asarray(array)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise RawCacheTransactionError(
                f"Raw-cache staging artifact is not a regular file: {path}"
            )
        stored = None
        try:
            stored = np.load(path, allow_pickle=False, mmap_mode="r")
            matches = (
                stored.dtype == value.dtype
                and stored.shape == value.shape
                and np.array_equal(stored, value)
            )
        except Exception:
            matches = False
        finally:
            del stored
        if matches:
            return
    _write_array(path, value)


def _discard_staging_temporaries(staging: Path) -> None:
    """Remove private atomic-write leftovers before publishing a resumed stage."""

    for candidate in staging.rglob(".*.tmp"):
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> Path:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RawCacheTransactionError(f"Raw-cache index already exists: {path}")
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException as exc:
        temp.unlink(missing_ok=True)
        raise RawCacheTransactionError(f"Atomic JSONL write failed for {path}: {exc}") from exc
    return path


def _prepare_raw_export(
    output_root: str | Path,
    *,
    index_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Resolve and create the private staging directory for one export."""

    root = Path(output_root).expanduser().resolve(strict=False)
    if not index_name or Path(index_name).name != index_name:
        raise RawCacheError("index_name must be a simple filename")
    if root.exists() and not overwrite:
        raise RawCacheTransactionError(f"Raw-cache output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging"
    if os.path.lexists(staging):
        if staging.is_symlink() or not staging.is_dir():
            raise RawCacheTransactionError(
                f"Raw-cache staging path is not a directory: {staging}"
            )
    else:
        staging.mkdir()
    return root, staging


def _stage_raw_chunks(
    chunks: Iterable[Any],
    *,
    root: Path,
    staging: Path,
) -> list[ChunkIndexRecord]:
    """Atomically stage one disjoint chunk stream and return compact records.

    Multiple processes may call this helper against the same staging directory
    only when their timeline IDs are disjoint. The parallel materializer
    establishes that partitioning before workers start; this helper still
    rejects duplicate identities within its own stream.
    """

    records: list[ChunkIndexRecord] = []
    seen: set[tuple[str, str, int]] = set()
    for ordinal, value in enumerate(chunks):
        chunk = _coerce_export_chunk(value, ordinal=ordinal)
        chunk_index = ordinal if chunk.chunk_index is None else int(chunk.chunk_index)
        identity = (chunk.timeline_id, chunk.modality, chunk_index)
        if identity in seen:
            raise RawCacheError(f"Duplicate raw-cache chunk identity: {identity}")
        seen.add(identity)
        slug = _raw_timeline_slug(chunk.timeline_id)
        base = Path("chunks") / slug / chunk.modality / f"chunk-{chunk_index:06d}"
        values_path = staging / f"{base}.values.npy"
        timestamps_path = staging / f"{base}.timestamps.npy"
        support_path = staging / f"{base}.support.npy"
        _reuse_or_write_array(values_path, chunk.values)
        _reuse_or_write_array(timestamps_path, chunk.timestamps)
        _reuse_or_write_array(support_path, chunk.support)
        records.append(
            ChunkIndexRecord(
                timeline_id=chunk.timeline_id,
                modality=chunk.modality,
                values=_descriptor(values_path, chunk.values),
                timestamps=_descriptor(timestamps_path, chunk.timestamps),
                support=_descriptor(support_path, chunk.support),
                n_samples=chunk.values.shape[0],
                shape=tuple(int(v) for v in chunk.values.shape),
                sample_rate_hz=chunk.sample_rate_hz,
                source_id=chunk.source_id,
                metadata=chunk.metadata,
                chunk_index=chunk_index,
                source_path=_source_path_provenance(chunk.source_path, root),
            )
        )
        del chunk
    return records


def _publish_raw_export(
    records: Sequence[ChunkIndexRecord],
    *,
    root: Path,
    staging: Path,
    index_name: str,
) -> Path:
    """Validate staged records and atomically expose a completed export."""

    if not records:
        raise RawCacheError("Cannot export an empty raw-cache chunk collection")
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        if record.identity in seen:
            raise RawCacheError(
                f"Duplicate raw-cache chunk identity: {record.identity}"
            )
        seen.add(record.identity)
    rows = [
        record.as_dict()
        for record in sorted(records, key=lambda item: item.identity)
    ]
    for row in rows:
        for name in ("values", "timestamps", "support"):
            descriptor = row[name]
            try:
                descriptor["path"] = str(
                    Path(descriptor["path"]).relative_to(staging)
                ).replace(os.sep, "/")
            except (KeyError, TypeError, ValueError) as exc:
                raise RawCacheTransactionError(
                    "Staged descriptor is outside the export transaction: "
                    f"{descriptor!r}"
                ) from exc
    staged_index = staging / index_name
    _write_jsonl_atomic(staged_index, rows, overwrite=staged_index.exists())
    _discard_staging_temporaries(staging)
    if root.exists():
        backup = root.parent / f".{root.name}.{uuid.uuid4().hex}.backup"
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except BaseException:
            if not root.exists() and backup.exists():
                os.replace(backup, root)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(staging, root)
    return root / index_name


def export_raw_chunks(
    chunks: Iterable[Any],
    output_root: str | Path,
    *,
    index_name: str = RAW_CACHE_INDEX_NAME,
    overwrite: bool = False,
) -> Path:
    """Write raw arrays with automatic resume and atomic final publication.

    A deterministic private sibling staging directory retains completed arrays
    after interruption. A single-writer rerun reuses arrays that exactly match
    the incoming chunks and atomically replaces missing or incomplete arrays.
    ``output_root`` is exposed only after every array and JSONL line is ready.
    Existing output directories are never modified unless ``overwrite=True``.
    """

    root, staging = _prepare_raw_export(
        output_root,
        index_name=index_name,
        overwrite=overwrite,
    )
    try:
        records = _stage_raw_chunks(chunks, root=root, staging=staging)
        return _publish_raw_export(
            records,
            root=root,
            staging=staging,
            index_name=index_name,
        )
    except BaseException as exc:
        if isinstance(exc, (RawCacheError, RawCacheTransactionError)):
            raise
        raise RawCacheTransactionError(f"Atomic raw-cache export failed: {exc}") from exc

def _strict_json_loads(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(raw, parse_constant=reject_constant)


def _record_from_mapping(value: Mapping[str, Any], *, index_path: Path) -> ChunkIndexRecord:
    if value.get("record_type", "chunk") != "chunk":
        raise RawCacheIntegrityError("raw-cache JSONL record_type must be 'chunk'")
    if value.get("schema_version") != RAW_CACHE_SCHEMA_VERSION:
        raise RawCacheIntegrityError(
            f"Unsupported raw-cache schema_version {value.get('schema_version')!r}"
        )
    try:
        shape = tuple(int(v) for v in value["shape"])
        record = ChunkIndexRecord(
            timeline_id=str(value["timeline_id"]),
            modality=str(value["modality"]),
            values=value["values"],
            timestamps=value["timestamps"],
            support=value["support"],
            n_samples=int(value["n_samples"]),
            shape=shape,
            sample_rate_hz=value.get("sample_rate_hz", RAW_FREQUENCY_HZ),
            source_id=str(value.get("source_id", "")),
            metadata=value.get("metadata", {}),
            chunk_index=int(value.get("chunk_index", 0)),
            index_path=index_path,
            source_path=value.get("source_path"),
        )
    except KeyError as exc:
        raise RawCacheIntegrityError(f"raw-cache record is missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise RawCacheIntegrityError(f"Malformed raw-cache record: {exc}") from exc
    for name in ("values", "timestamps", "support"):
        descriptor = getattr(record, name)
        if not isinstance(descriptor.get("path"), str) or not descriptor.get("path"):
            raise RawCacheIntegrityError(f"{record.identity}: {name}.path is missing")
        for field in ("size_bytes", "mtime_ns", "shape", "dtype"):
            if field not in descriptor:
                raise RawCacheIntegrityError(
                    f"{record.identity}: {name} descriptor is missing {field}"
                )
        if not isinstance(descriptor["dtype"], str) or not descriptor["dtype"].strip():
            raise RawCacheIntegrityError(
                f"{record.identity}: {name}.dtype must be a non-empty string"
            )
        shape_value = descriptor["shape"]
        if not isinstance(shape_value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape_value
        ):
            raise RawCacheIntegrityError(
                f"{record.identity}: {name}.shape must be a list of non-negative integers"
            )
        expected_shape = record.shape if name == "values" else (record.n_samples,)
        if tuple(shape_value) != expected_shape:
            raise RawCacheIntegrityError(
                f"{record.identity}: {name}.shape does not match the record"
            )
        for field in ("size_bytes", "mtime_ns"):
            _descriptor_integer(
                descriptor[field],
                label=f"{record.identity}.{name}.{field}",
            )
    return record


def read_chunk_index(index_path: str | Path) -> tuple[ChunkIndexRecord, ...]:
    """Read an index and validate artifact containment and structural stats."""

    index = Path(index_path).expanduser().resolve(strict=True)
    if not index.is_file():
        raise RawCacheIntegrityError(f"Raw-cache index is not a regular file: {index}")
    records: list[ChunkIndexRecord] = []
    seen: set[tuple[str, str, int]] = set()
    try:
        with index.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = _strict_json_loads(text)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    raise RawCacheIntegrityError(f"Malformed raw-cache JSONL at {index}:{line_number}: {exc}") from exc
                if not isinstance(payload, Mapping):
                    raise RawCacheIntegrityError(f"Raw-cache record at {index}:{line_number} is not an object")
                record = _record_from_mapping(payload, index_path=index)
                if record.identity in seen:
                    raise RawCacheIntegrityError(f"Duplicate raw-cache chunk identity at {index}:{line_number}: {record.identity}")
                seen.add(record.identity)
                root = index.parent.resolve(strict=True)
                for name in ("values", "timestamps", "support"):
                    descriptor = getattr(record, name)
                    path = _safe_relative_path(root, descriptor["path"], label=name)
                    _check_file_stats(path, descriptor, label=name)
                records.append(record)
    except OSError as exc:
        raise RawCacheIntegrityError(f"Cannot read raw-cache index {index}: {exc}") from exc
    if not records:
        raise RawCacheIntegrityError(f"Raw-cache index is empty: {index}")
    return tuple(records)


def support_mask_from_intervals(
    timestamps: Any,
    intervals: Iterable[Sequence[float]],
) -> np.ndarray:
    """Mark timestamp centres inside half-open physical support intervals."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RawCacheError("timestamps must be a finite one-dimensional array")
    mask = np.zeros(values.shape, dtype=bool)
    count = 0
    for interval in intervals:
        if len(interval) != 2:
            raise RawCacheError("support intervals must be [start, stop] pairs")
        start = _finite(interval[0], name="support start")
        stop = _finite(interval[1], name="support stop")
        if stop <= start:
            raise RawCacheError("support interval stop must be greater than start")
        mask |= (values >= start) & (values < stop)
        count += 1
    if count == 0:
        raise RawCacheError("support intervals cannot be empty")
    return np.ascontiguousarray(mask)


def _event_rows(events: Any) -> Iterator[Mapping[str, Any]]:
    if events is None:
        return
    if hasattr(events, "to_dict"):
        try:
            rows = events.to_dict(orient="records")
        except TypeError:
            rows = events.to_dict("records")
        yield from rows
        return
    if isinstance(events, Mapping):
        yield events
        return
    for row in events:
        if isinstance(row, Mapping):
            yield row
        elif hasattr(row, "_asdict"):
            yield row._asdict()
        else:
            raise RawCacheError("events must be a mapping, DataFrame, or iterable of mappings")


def support_mask_from_events(
    timestamps: Any,
    events: Any,
    *,
    timeline_id: str | None = None,
    event_types: Iterable[str] | None = None,
) -> np.ndarray:
    """Build a source-support mask from transformed-event rows.

    Event rows must provide ``start`` and either ``stop`` or ``duration``.  A
    supplied ``timeline_id`` filters rows before interval construction.  The
    interval convention is half-open, matching the native pooling contract.
    """

    accepted = None if event_types is None else {str(value) for value in event_types}
    intervals: list[tuple[float, float]] = []
    for row in _event_rows(events):
        if timeline_id is not None and str(row.get("timeline", "")) != str(timeline_id):
            continue
        if accepted is not None and str(row.get("type", "")) not in accepted:
            continue
        try:
            start = _finite(row.get("start"), name="event start")
        except RawCacheError:
            continue
        stop_value = row.get("stop")
        if stop_value in (None, ""):
            duration_value = row.get("duration")
            try:
                duration = _finite(duration_value, name="event duration")
            except RawCacheError:
                continue
            stop = start + duration
        else:
            try:
                stop = _finite(stop_value, name="event stop")
            except RawCacheError:
                continue
        if stop > start:
            intervals.append((start, stop))
    if not intervals:
        raise RawCacheError("No finite positive event support intervals were found")
    return support_mask_from_intervals(timestamps, intervals)


def neuralset_batch_to_chunk(
    timeline_id: str,
    modality: str,
    batch: Any,
    *,
    start_seconds: float | None = None,
    timestamps: Any | None = None,
    support: Any | None = None,
    support_intervals: Iterable[Sequence[float]] | None = None,
    events: Any | None = None,
    event_types: Iterable[str] | None = None,
    batch_index: int = 0,
    source_id: str = "",
    source_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RawCacheChunk:
    """Convert an explicit NeuralSet-style batch to a physical raw chunk.

    The function is intentionally agnostic to NeuralSet imports: the caller
    owns loader construction and passes its batch here.  ``timestamps`` may be
    supplied when an upstream loader exposes a nonzero physical origin;
    otherwise ``start_seconds`` is required and sample centres are generated.
    Support is similarly explicit: a boolean mask, intervals, or transformed
    event rows may be provided.  If no support information is supplied, every
    returned row is considered source-backed because the caller has asserted
    that the batch is complete.
    """

    canonical = _canonical_modality(modality)
    values = extract_batch_data(
        batch,
        canonical,
        batch_index=batch_index,
        expected_layers=EXPECTED_LAYERS,
        expected_dim=MODALITY_DIMS[canonical],
    )
    if timestamps is None:
        if start_seconds is None:
            raise RawCacheError(
                "A NeuralSet batch requires physical start_seconds or explicit timestamps; "
                "do not infer timing from an array index"
            )
        timestamp_values = sample_center_timestamps(start_seconds, values.shape[0])
    else:
        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        if timestamp_values.ndim != 1 or timestamp_values.size != values.shape[0]:
            raise RawCacheError("Explicit timestamps must be one-dimensional and align with the batch")
        if not np.all(np.isfinite(timestamp_values)) or np.any(np.diff(timestamp_values) <= 0):
            raise RawCacheError("Explicit timestamps must be finite and strictly increasing")
    if support is not None and (support_intervals is not None or events is not None):
        raise RawCacheError("Provide exactly one of support, support_intervals, or events")
    if support is None and support_intervals is not None:
        support = support_mask_from_intervals(timestamp_values, support_intervals)
    elif support is None and events is not None:
        support = support_mask_from_events(
            timestamp_values,
            events,
            timeline_id=timeline_id,
            event_types=event_types,
        )
    return RawCacheChunk(
        timeline_id=timeline_id,
        modality=canonical,
        values=values,
        timestamps=timestamp_values,
        support=support,
        source_id=source_id,
        source_path=source_path,
        metadata={} if metadata is None else metadata,
        sample_rate_hz=RAW_FREQUENCY_HZ,
    )


def load_explicit_extractor(source: str) -> Callable[..., Any]:
    """Load ``module:function`` without importing NeuralSet at module import."""

    if not isinstance(source, str) or ":" not in source:
        raise RawCacheError(
            "Extractor source must be an explicit module:function path; "
            "automatic NeuralSet loader discovery is disabled"
        )
    module_name, function_name = source.rsplit(":", 1)
    if not module_name or not function_name or not function_name.isidentifier():
        raise RawCacheError(f"Invalid extractor source {source!r}; use module:function")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise RawCacheError(
            f"Cannot import explicit extractor {source!r}. Put NeuralSet/TRIBEv2 in the "
            "runtime and expose a function returning raw chunks: module:function"
        ) from exc
    if not callable(function):
        raise RawCacheError(f"Explicit extractor {source!r} is not callable")
    return function


def chunks_from_explicit_extractor(
    source: str | Callable[..., Any],
    *,
    source_kwargs: Mapping[str, Any] | None = None,
    strict_timeline_ids: bool = False,
) -> Iterator[RawCacheChunk]:
    """Return a lazy, validated iterator from an explicit extractor.

    The callable is invoked only with the exact keyword arguments supplied in
    ``source_kwargs``.  This prevents accidental version-specific guesses about
    NeuralSet's loader API.  Validation and conversion happen as each item is
    requested, so an upstream batch is not retained while later chunks are
    extracted.  A callback can return any input accepted by
    :func:`export_raw_chunks`, including mappings containing ``batch`` and
    physical ``start_seconds``.
    """

    function = load_explicit_extractor(source) if isinstance(source, str) else source
    if not callable(function):
        raise RawCacheError("source must be module:function or a callable")
    kwargs = {} if source_kwargs is None else dict(source_kwargs)
    _safe_json(kwargs, label="source_kwargs")

    def _validated() -> Iterator[RawCacheChunk]:
        try:
            try:
                result = function(**kwargs)
            except TypeError as exc:
                try:
                    signature = str(inspect.signature(function))
                except (TypeError, ValueError):
                    signature = "<unavailable>"
                raise RawCacheError(
                    f"Explicit extractor rejected source_kwargs ({exc}); its signature is {signature}. "
                    "Supply the upstream-specific arguments explicitly; no NeuralSet API is inferred."
                ) from exc
            if result is None:
                raise RawCacheError(
                    "Explicit extractor returned None; return an iterable of raw chunks"
                )
            try:
                iterable = iter(result)
            except TypeError as exc:
                raise RawCacheError(
                    "Explicit extractor must return an iterable of raw chunks"
                ) from exc
            yielded = False
            for ordinal, value in enumerate(iterable):
                yielded = True
                if strict_timeline_ids:
                    if isinstance(value, Mapping):
                        raw_timeline_id = value.get("timeline_id")
                    elif isinstance(value, (tuple, list)) and len(value) == 3:
                        raw_timeline_id = value[0]
                    else:
                        raw_timeline_id = getattr(value, "timeline_id", None)
                    if not isinstance(raw_timeline_id, str) or not raw_timeline_id.strip():
                        raise RawCacheError(
                            "Raw-only extractor yielded a chunk with a non-empty string timeline_id"
                        )
                yield _coerce_export_chunk(value, ordinal=ordinal)
            if not yielded:
                raise RawCacheError("Explicit extractor returned no raw chunks")
        finally:
            # Close generator-backed upstream loaders promptly on a failed
            # conversion or when the consumer stops early.
            close = locals().get("iterable")
            close_method = getattr(close, "close", None)
            if callable(close_method):
                close_method()

    return _validated()


_CONFIG_MISSING = object()


def _config_get(container: Any, key: str, default: Any = _CONFIG_MISSING) -> Any:
    if isinstance(container, Mapping):
        if key in container:
            return container[key]
    else:
        try:
            return getattr(container, key)
        except AttributeError:
            pass
    if default is not _CONFIG_MISSING:
        return default
    raise RawCacheError(f"TRIBEv2 config is missing {key!r}")


def _config_set(container: Any, key: str, value: Any) -> None:
    try:
        if isinstance(container, Mapping):
            container[key] = value
        else:
            setattr(container, key, value)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RawCacheError(f"Cannot set TRIBEv2 config field {key!r}: {exc}") from exc


def _config_pop(container: Any, key: str) -> None:
    try:
        if isinstance(container, Mapping):
            container.pop(key, None)
        elif hasattr(container, key):
            delattr(container, key)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RawCacheError(f"Cannot remove TRIBEv2 config field {key!r}: {exc}") from exc


def _iter_config_nodes(value: Any) -> Iterator[Any]:
    """Walk plain/OmegaConf config nodes without importing either package."""

    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_config_nodes(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_config_nodes(child)


def _config_values(config: Any, keys: Iterable[str]) -> list[Any]:
    wanted = {str(key) for key in keys}
    values: list[Any] = []
    for node in _iter_config_nodes(config):
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in wanted:
                    values.append(value)
    return values


def _normalise_layer_selection(value: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        # A YAML string is accepted only when it is an explicit comma-separated
        # list; this avoids silently treating a model name as a layer list.
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pieces = list(value)
    else:
        raise RawCacheError(f"{label} must be a sequence of layer fractions")
    try:
        result = tuple(float(piece) for piece in pieces)
    except (TypeError, ValueError) as exc:
        raise RawCacheError(f"{label} must contain numeric layer fractions") from exc
    if result != (0.5, 0.75, 1.0):
        raise RawCacheError(
            f"{label} must be exactly [0.5, 0.75, 1.0] for the accepted raw cache; got {result}"
        )
    return result


def _load_tribev2_config(config_path: Path) -> Any:
    """Load the pinned YAML into the object form expected by NeuralSet Data.

    OmegaConf is convenient when available, but the pinned TRIBEv2 release
    serialises pathlib values that some OmegaConf versions reject.  Its own
    read-only route uses ``exca.ConfDict(yaml.load(..., UnsafeLoader))`` for
    this trusted local checkpoint YAML, so retain that as the first fallback.
    The restricted repository loader remains a dependency-free last resort for
    tests/minimal installations.
    """

    omega_error: Exception | None = None
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None:
        try:
            return OmegaConf.load(str(config_path))
        except Exception as exc:
            omega_error = exc

    confdict_error: Exception | None = None
    try:
        import yaml
        from exca import ConfDict
    except ImportError as exc:
        confdict_error = exc
    else:
        try:
            raw = yaml.load(
                config_path.read_text(encoding="utf-8"),
                Loader=yaml.UnsafeLoader,
            )
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"expected a YAML mapping, got {type(raw).__name__}"
                )
            return ConfDict(raw)
        except Exception as exc:
            confdict_error = exc

    try:
        # This known-tag-only loader is useful in minimal environments where
        # neither OmegaConf nor Exca is installed; it still rejects arbitrary
        # Python object construction.
        from neuroworld.nbs_data.stimuli.config import load_checkpoint_yaml_mapping

        return load_checkpoint_yaml_mapping(config_path)
    except Exception as restricted_error:
        details = [f"restricted loader: {restricted_error}"]
        if omega_error is not None:
            details.insert(0, f"OmegaConf: {omega_error}")
        if confdict_error is not None:
            details.insert(0, f"ConfDict: {confdict_error}")
        raise RawCacheError(
            f"Cannot load TRIBEv2 config {config_path}; use the pinned TRIBE environment: "
            + "; ".join(details)
        ) from restricted_error


def _prepare_tribev2_data_config(
    config: Any,
    *,
    cache_root: Path,
    dataset: str,
) -> Any:
    """Rewrite only the read-only Data cache bindings used by current caches."""

    data_config = _config_get(config, "data")
    requested_modalities = ("video", "audio", "text") if dataset == "algonauts" else ("video", "audio")
    # Restrict Data construction to the cache modalities that this dataset is
    # allowed to consume.  In particular, HCP must not accidentally ask
    # NeuralSet to prepare a checkpoint's contextual-text cache.
    _config_set(data_config, "features_to_use", list(requested_modalities))
    # Data validates every configured feature object, even when a feature is
    # absent from features_to_use.  Bind all current cache extractors to the
    # supplied read-only root, while the explicit features_to_use list below
    # controls which modalities are actually prepared/emitted for this dataset.
    for modality in ("video", "audio", "text"):
        feature = _config_get(data_config, f"{modality}_feature", None)
        if feature is None:
            if modality in requested_modalities:
                raise RawCacheError(
                    f"TRIBEv2 config is missing data.{modality}_feature required for {dataset}"
                )
            continue
        infra = _config_get(feature, "infra", None)
        if infra is None:
            if modality in requested_modalities:
                raise RawCacheError(
                    f"TRIBEv2 config is missing data.{modality}_feature.infra for the current cache"
                )
            continue
        _config_set(infra, "folder", str(cache_root))
        _config_set(infra, "cluster", None)


    study = _config_get(data_config, "study", None)
    if study is not None:
        _config_set(study, "path", ".")
        _config_pop(study, "infra_timelines")
    neuro = _config_get(data_config, "neuro", None)
    if neuro is not None:
        _config_pop(neuro, "infra")
    image_feature = _config_get(data_config, "image_feature", None)
    if image_feature is not None:
        _config_pop(image_feature, "infra")
    subject_id = _config_get(data_config, "subject_id", None)
    if subject_id is not None:
        _config_set(subject_id, "predefined_mapping", None)

    _config_set(data_config, "batch_size", 1)
    _config_set(data_config, "num_workers", 0)
    _config_set(data_config, "shuffle_train", False)
    _config_set(data_config, "shuffle_val", False)

    # Refuse a checkpoint that would read a different layer reduction or cache
    # width.  These are the values used to build the accepted 20-layer caches;
    # no averaging/projector/Transformer is performed here.  Keep the checks
    # at data-level/requested-extractor scope: an unrelated image-feature
    # setting such as ``layer_aggregation: mean`` is not evidence about these
    # raw modality caches.
    layer_value = _config_get(data_config, "layers_to_use", None)
    if layer_value is None:
        raise RawCacheError(
            "TRIBEv2 config data.layers_to_use is missing; cannot prove the accepted "
            "[0.5, 0.75, 1.0] group-mean raw-cache contract"
        )
    _normalise_layer_selection(layer_value, label="data.layers_to_use")
    aggregation_value = _config_get(data_config, "layer_aggregation", None)
    if aggregation_value is None:
        raise RawCacheError(
            "TRIBEv2 config data.layer_aggregation is missing; cannot prove group_mean cache semantics"
        )
    if str(aggregation_value).strip().lower() != "group_mean":
        raise RawCacheError(
            "TRIBEv2 config data.layer_aggregation must be exactly 'group_mean'"
        )
    for modality in requested_modalities:
        feature = _config_get(data_config, f"{modality}_feature")
        cache_layer_values = _config_values(feature, ("cache_n_layers",))
        if not cache_layer_values:
            raise RawCacheError(
                f"TRIBEv2 config data.{modality}_feature does not expose cache_n_layers; "
                "cannot prove the accepted 20-layer raw cache"
            )
        for value in cache_layer_values:
            try:
                cache_layers = int(value)
            except (TypeError, ValueError) as exc:
                raise RawCacheError(
                    f"TRIBEv2 data.{modality}_feature cache_n_layers must be an integer"
                ) from exc
            if cache_layers != 20:
                raise RawCacheError(
                    f"TRIBEv2 data.{modality}_feature cache_n_layers must be 20 for the accepted raw cache, got {cache_layers}"
                )
    return config


def _validate_tribev2_data_runtime(data: Any) -> float:
    frequency = _config_get(data, "frequency", None)
    if frequency is None:
        raise RawCacheError("TRIBEv2 Data has no frequency; cannot establish a physical 2 Hz clock")
    try:
        frequency_value = float(frequency)
    except (TypeError, ValueError) as exc:
        raise RawCacheError(f"TRIBEv2 Data frequency is not numeric: {frequency!r}") from exc
    if not math.isfinite(frequency_value) or not math.isclose(
        frequency_value, RAW_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RawCacheError(
            f"TRIBEv2 Data frequency must be exactly {RAW_FREQUENCY_HZ} Hz, got {frequency!r}"
        )
    for name, expected in (
        ("batch_size", 1),
        ("num_workers", 0),
        ("shuffle_train", False),
        ("shuffle_val", False),
    ):
        current = _config_get(data, name, None)
        if current is not None and current != expected:
            raise RawCacheError(
                f"TRIBEv2 Data {name} must be {expected!r}, got {current!r}"
            )
    get_loaders = _config_get(data, "get_loaders", None)
    if not callable(get_loaders):
        raise RawCacheError("TRIBEv2 Data does not expose get_loaders(events=..., split_to_build='all')")
    return frequency_value


def _load_tribev2_events(events_path: Any) -> Any:
    if hasattr(events_path, "columns") and hasattr(events_path, "iterrows"):
        return events_path
    try:
        import pandas as pd
        from neuroworld.nbs_data.stimuli.common import (
            _read_transformed_events,
            _resolve_events_paths,
        )
        paths = _resolve_events_paths(events_path)
        return _read_transformed_events(paths, pd)
    except ImportError as exc:
        raise RawCacheError(
            "TRIBEv2 raw extraction from TSV requires pandas in the TRIBE environment"
        ) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise RawCacheError(f"Cannot load transformed TRIBEv2 events: {exc}") from exc


def _batch_modality_payload(batch: Any) -> dict[str, Any]:
    payload = getattr(batch, "data", batch)
    if not isinstance(payload, Mapping):
        raise RawCacheError(
            "TRIBEv2 loader batch.data must be a mapping containing video/audio (and text for Algonauts)"
        )
    result: dict[str, Any] = {}
    for key, value in payload.items():
        try:
            modality = _canonical_modality(str(key))
        except RawCacheError:
            # subject_id and other bookkeeping fields are not stimulus modalities.
            continue
        if modality in result:
            raise RawCacheError(f"TRIBEv2 batch.data contains duplicate {modality!r} modalities")
        result[modality] = value
    return result


def _validate_batch_size_one(payload: Mapping[str, Any], *, timeline_id: str) -> None:
    for modality, value in payload.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            try:
                shape = np.asarray(value).shape
            except (TypeError, ValueError) as exc:
                raise RawCacheError(f"{timeline_id}: {modality} batch is not array-like") from exc
        if len(shape) != 4 or int(shape[0]) != 1:
            raise RawCacheError(
                f"{timeline_id}: {modality} must be NeuralSet [B,L,D,T] with B=1, got {tuple(shape)}"
            )


def _iter_loader_with_close(loader: Any) -> Iterator[Any]:
    """Yield loader batches and close the loader on errors or early stop."""

    try:
        yield from loader
    finally:
        close = getattr(loader, "close", None)
        if callable(close):
            close()


def _normalise_selection_ids(value: Iterable[str] | None, *, label: str) -> set[str] | None:
    """Normalize JSON-friendly timeline/stimulus selections to non-empty sets."""

    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise RawCacheError(f"{label} must be a JSON/list-like collection of IDs, not a string")
    try:
        selected = {str(item).strip() for item in value}
    except TypeError as exc:
        raise RawCacheError(f"{label} must be an iterable of IDs") from exc
    if not selected or "" in selected:
        raise RawCacheError(f"{label} must contain at least one non-empty ID")
    return selected


def iter_tribev2_raw_chunks(
    *,
    events_path: Any,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    dataset: str = "algonauts",
    selected_stimulus_ids: set[str] | None = None,
    selected_timeline_ids: set[str] | None = None,
    limit: int | None = None,
    data: Any | None = None,
    data_factory: Callable[[Any], Any] | None = None,
) -> Iterator[RawCacheChunk]:
    """Stream current TRIBEv2/NeuralSet raw cache batches as ``RawCacheChunk``.

    This is the concrete callback for the current caches.  With no injection,
    it loads ``tribev2.main.Data`` from the local checkpoint ``config.yaml``;
    it rewrites only the read-only cache bindings and calls
    ``data.get_loaders(events=group.events, split_to_build='all')``.  The
    returned batch is consumed *before* any projector, Transformer, aggregate
    feature call, or post-transformer representation.  NeuralSet's configured
    ``layers_to_use=[0.5,0.75,1.0]`` and ``layer_aggregation='group_mean'``
    therefore produce the accepted ``[B,2,D,T]`` arrays, which this adapter
    converts to ``[T,2,D]``.

    ``data`` or ``data_factory`` are explicit dependency-injection hooks for
    tests and pinned environment wrappers.  ``data_factory`` receives the
    rewritten config object.  They do not change the production Data-only path.
    Every yielded segment carries physical sample-centre timestamps and a
    modality-specific support mask from the transformed event table.  Dataset
    policy is strict: Algonauts requires video/audio/text batch arrays, while
    transcript-free timelines emit finite all-zero text arrays with an all-false
    support mask. HCP requires video/audio and explicitly
    ignore text only after validating it is present as an unrequested batch
    modality.
    """

    canonical_dataset = _canonical_dataset(dataset)
    selected_stimulus_ids = _normalise_selection_ids(
        selected_stimulus_ids,
        label="selected_stimulus_ids",
    )
    selected_timeline_ids = _normalise_selection_ids(
        selected_timeline_ids,
        label="selected_timeline_ids",
    )
    if selected_stimulus_ids is not None and selected_timeline_ids is not None:
        raise RawCacheError("Select transformed events by timeline or stimulus, not both")
    if data is not None and data_factory is not None:
        raise RawCacheError("Provide at most one of data or data_factory")
    if limit is not None and (
        isinstance(limit, bool) or int(limit) <= 0 or int(limit) != limit
    ):
        raise RawCacheError("limit must be a positive integer")
    def _iter() -> Iterator[RawCacheChunk]:
        cache_root: Path | None = None
        if cache_dir is not None:
            try:
                cache_root = Path(cache_dir).expanduser().resolve(strict=True)
            except OSError as exc:
                raise RawCacheError(f"TRIBEv2 cache_dir is not readable: {cache_dir!r}") from exc
            if not cache_root.is_dir():
                raise RawCacheError(f"TRIBEv2 cache_dir is not a directory: {cache_root}")
        if data is None and cache_root is None:
            raise RawCacheError("cache_dir is required for the concrete TRIBEv2 cache route")

        resolved_config: Path | None = None
        config: Any | None = None
        if config_path is not None:
            try:
                resolved_config = Path(config_path).expanduser().resolve(strict=True)
            except OSError as exc:
                raise RawCacheError(f"TRIBEv2 config.yaml is not readable: {config_path!r}") from exc
            if not resolved_config.is_file():
                raise RawCacheError(f"TRIBEv2 config is not a regular file: {resolved_config}")
            config = _load_tribev2_config(resolved_config)
            if cache_root is None:
                raise RawCacheError("cache_dir is required when loading Data from config.yaml")
            _prepare_tribev2_data_config(
                config,
                cache_root=cache_root,
                dataset=canonical_dataset,
            )

        if data is not None:
            data_runtime = data
        elif data_factory is not None:
            try:
                data_runtime = data_factory(config)
            except Exception as exc:
                raise RawCacheError(f"Explicit TRIBEv2 Data factory failed: {exc}") from exc
        else:
            try:
                from tribev2.main import Data
            except ImportError as exc:
                raise RawCacheError(
                    "The concrete TRIBEv2 route requires tribev2.main.Data; run it in the pinned TRIBE environment"
                ) from exc
            data_config = _config_get(config, "data")
            if not isinstance(data_config, Mapping):
                raise RawCacheError(
                    "TRIBEv2 config data must be a mapping suitable for Data(**config['data'])"
                )
            try:
                # The pinned Data class is a pydantic-style constructor whose
                # public API is Data(**cfg["data"]); passing the full root
                # config would silently skip/rename fields in some versions.
                data_runtime = Data(**data_config)
            except Exception as exc:
                raise RawCacheError(
                    f"Cannot instantiate tribev2.main.Data from config['data'] keyword fields: {exc}. "
                    "Use data_factory only for an explicitly reviewed environment wrapper."
                ) from exc
        frequency = _validate_tribev2_data_runtime(data_runtime)

        # A directly injected fake/runtime has no YAML to validate.  Require
        # the fields that establish the accepted layer semantics explicitly.
        if config is None:
            runtime_layers = _config_get(data_runtime, "layers_to_use", None)
            if runtime_layers is None:
                raise RawCacheError(
                    "Data.layers_to_use is missing; cannot prove the accepted [0.5, 0.75, 1.0] layer contract"
                )
            _normalise_layer_selection(runtime_layers, label="Data.layers_to_use")
            runtime_aggregation = _config_get(data_runtime, "layer_aggregation", None)
            if runtime_aggregation is None:
                raise RawCacheError(
                    "Data.layer_aggregation is missing; cannot prove group_mean cache semantics"
                )
            if str(runtime_aggregation).strip().lower() != "group_mean":
                raise RawCacheError("Data.layer_aggregation must be exactly 'group_mean'")

        events = _load_tribev2_events(events_path)
        try:
            from neuroworld.nbs_data.stimuli.common import (
                iter_timeline_event_groups,
                neuralset_segment_boundary_shift_seconds,
                normalize_neuralset_segment_start,
            )
        except ImportError as exc:
            raise RawCacheError(
                "Cannot import the audited TRIBEv2 event/boundary helpers"
            ) from exc
        groups: Iterable[Any] = iter_timeline_event_groups(
            events,
            selected_stimulus_ids=selected_stimulus_ids,
            selected_timeline_ids=selected_timeline_ids,
        )
        if limit is not None:
            from itertools import islice

            groups = islice(groups, int(limit))
        yielded_any = False
        for group in groups:
            requested_modalities = (
                ("video", "audio", "text")
                if canonical_dataset == "algonauts"
                else ("video", "audio")
            )
            event_type_values = {
                str(value) for value in group.events["type"]
            }
            support_types = {
                "video": ("Video",),
                "audio": ("Audio",),
                "text": ("Word", "Text"),
            }
            has_support_rows = {
                modality: bool(event_type_values.intersection(event_types))
                for modality, event_types in support_types.items()
            }
            # Video and audio define the physical movie timeline and must
            # always be source-backed. Some audited Algonauts clips have no
            # transcript rows; their cached text tensor is still required but
            # none of its samples may be presented as linguistically supported.
            for modality in ("video", "audio"):
                if not has_support_rows[modality]:
                    raise RawCacheError(
                        f"{group.timeline_id}: transformed events have no {modality} support rows"
                    )
            ignored_text = canonical_dataset != "algonauts" and has_support_rows["text"]
            boundary_shift = neuralset_segment_boundary_shift_seconds(
                group.events,
                physical_support_start_seconds=group.start_seconds,
                label=group.timeline_id,
            )
            group_stop = float(group.start_seconds + group.duration_seconds)
            loaders = data_runtime.get_loaders(
                events=group.events,
                split_to_build="all",
            )
            if not isinstance(loaders, Mapping) or "all" not in loaders:
                raise RawCacheError(
                    f"{group.timeline_id}: Data.get_loaders did not return an 'all' loader"
                )
            loader = loaders["all"]
            yielded_group = False
            previous_segment_stop: float | None = None
            for segment_index, batch in enumerate(_iter_loader_with_close(loader)):
                payload = _batch_modality_payload(batch)
                _validate_batch_size_one(payload, timeline_id=group.timeline_id)
                missing = set(requested_modalities) - set(payload)
                if canonical_dataset == "algonauts" and not has_support_rows["text"]:
                    missing.discard("text")
                missing = sorted(missing)
                if missing:
                    raise RawCacheError(
                        f"{group.timeline_id}: loader batch is missing requested modalities {missing}"
                    )
                # Text in non-Algonauts is deliberately checked above, then
                # excluded from the output by iterating requested_modalities.
                raw_segment = getattr(batch, "segments", None)
                if raw_segment is None or len(raw_segment) != 1:
                    raise RawCacheError(
                        f"{group.timeline_id}: batch_size=1 requires exactly one NeuralSet segment"
                    )
                ns_segment = raw_segment[0]
                try:
                    raw_start = float(ns_segment.start)
                    segment_duration = float(ns_segment.duration)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RawCacheError(
                        f"{group.timeline_id}: NeuralSet segment lacks numeric start/duration"
                    ) from exc
                normalized_start = normalize_neuralset_segment_start(
                    raw_start,
                    boundary_shift_seconds=boundary_shift,
                    label=group.timeline_id,
                )
                if not math.isfinite(segment_duration) or segment_duration <= 0:
                    raise RawCacheError(
                        f"{group.timeline_id}: NeuralSet segment duration must be finite and positive"
                    )
                segment_stop = normalized_start + segment_duration
                if not math.isfinite(segment_stop):
                    raise RawCacheError(
                        f"{group.timeline_id}: NeuralSet segment end must be finite"
                    )
                timeline_start = float(group.start_seconds)
                if previous_segment_stop is None:
                    if not math.isclose(
                        normalized_start,
                        timeline_start,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise RawCacheError(
                            f"{group.timeline_id}: first segment must start at timeline start "
                            f"{timeline_start:g}, got {normalized_start:g}"
                        )
                elif not math.isclose(
                    normalized_start,
                    previous_segment_stop,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise RawCacheError(
                        f"{group.timeline_id}: segments must be contiguous; expected start "
                        f"{previous_segment_stop:g}, got {normalized_start:g}"
                    )
                if not (normalized_start < group_stop and segment_stop > timeline_start):
                    raise RawCacheError(
                        f"{group.timeline_id}: segment [{normalized_start:g}, {segment_stop:g}) "
                        f"does not intersect timeline [{timeline_start:g}, {group_stop:g})"
                    )
                segment_samples: int | None = None
                for modality in requested_modalities:
                    synthesized_values = (
                        modality == "text"
                        and modality not in payload
                        and not has_support_rows["text"]
                    )
                    if synthesized_values:
                        if segment_samples is None:
                            raise RawCacheError(
                                f"{group.timeline_id}: cannot establish text segment length"
                            )
                        values = np.zeros(
                            (
                                segment_samples,
                                EXPECTED_LAYERS,
                                MODALITY_DIMS[modality],
                            ),
                            dtype=np.float32,
                        )
                    else:
                        values = extract_batch_data(
                            batch,
                            modality,
                            batch_index=0,
                            expected_layers=EXPECTED_LAYERS,
                            expected_dim=MODALITY_DIMS[modality],
                        )
                    n_samples = int(values.shape[0])
                    if segment_samples is None:
                        segment_samples = n_samples
                        expected_duration = n_samples / frequency
                        if not math.isclose(
                            segment_duration,
                            expected_duration,
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        ):
                            raise RawCacheError(
                                f"{group.timeline_id}: segment duration {segment_duration} does not match "
                                f"{n_samples} samples at {frequency:g} Hz"
                            )
                    elif n_samples != segment_samples:
                        raise RawCacheError(
                            f"{group.timeline_id}: modality segment lengths disagree"
                        )
                    timestamps = sample_center_timestamps(
                        normalized_start,
                        n_samples,
                        frequency_hz=frequency,
                    )
                    if modality == "text" and not has_support_rows["text"]:
                        support = np.zeros(n_samples, dtype=np.bool_)
                    else:
                        try:
                            support = support_mask_from_events(
                                timestamps,
                                group.events,
                                timeline_id=group.timeline_id,
                                event_types=support_types[modality],
                            )
                        except RawCacheError as exc:
                            raise RawCacheError(
                                f"{group.timeline_id}: cannot build {modality} support mask from transformed events: {exc}"
                            ) from exc
                    source_id = (
                        f"tribev2:{group.timeline_id}:segment-{segment_index:06d}:{modality}"
                    )
                    metadata = {
                        "extractor": "tribev2.main.Data.get_loaders",
                        "raw_segment_start_seconds": raw_start,
                        "physical_segment_start_seconds": normalized_start,
                        "segment_duration_seconds": segment_duration,
                        "timeline_support_start_seconds": float(group.start_seconds),
                        "timeline_support_stop_seconds": group_stop,
                        "boundary_shift_seconds": float(boundary_shift),
                        "support_event_types": list(support_types[modality]),
                        "support_rows_present": has_support_rows[modality],
                        "values_synthesized": synthesized_values,
                        "layers_to_use": [0.5, 0.75, 1.0],
                        "layer_aggregation": "group_mean",
                        "cache_n_layers": 20,
                        "ignored_unrequested_text": ignored_text,
                    }
                    chunk = RawCacheChunk(
                        timeline_id=group.timeline_id,
                        modality=modality,
                        values=values,
                        timestamps=timestamps,
                        support=support,
                        source_id=source_id,
                        source_path=cache_root,
                        metadata=metadata,
                        sample_rate_hz=frequency,
                        chunk_index=segment_index,
                    )
                    yielded_any = yielded_group = True
                    yield chunk
                    del chunk, values, timestamps, support
                previous_segment_stop = segment_stop
                del batch
            if not yielded_group:
                raise RawCacheError(
                    f"{group.timeline_id}: Data.get_loaders returned no usable raw segments"
                )
            if previous_segment_stop is None or previous_segment_stop < group_stop - 1e-9:
                raise RawCacheError(
                    f"{group.timeline_id}: segments do not cover timeline through "
                    f"{group_stop:g}; final segment ends at {previous_segment_stop:g}"
                )
        if not yielded_any:
            raise RawCacheError("TRIBEv2 event selection produced no raw segments")

    return _iter()


@dataclass(frozen=True)
class NativeTimelineSpec:
    """Timing contract needed to pool one raw timeline to native BOLD bins."""

    timeline_id: str
    dataset: str
    origin_seconds: float
    native_tr_seconds: float
    n_bins: int
    require_text: bool = False

    def __post_init__(self) -> None:
        if not str(self.timeline_id).strip():
            raise RawCacheError("timeline_id must be non-empty")
        object.__setattr__(self, "timeline_id", str(self.timeline_id).strip())
        object.__setattr__(self, "dataset", _canonical_dataset(self.dataset))
        object.__setattr__(self, "origin_seconds", _finite(self.origin_seconds, name="origin_seconds"))
        if isinstance(self.native_tr_seconds, (bool, np.bool_)):
            raise RawCacheError("native_tr_seconds must be numeric, not boolean")
        tr = _finite(self.native_tr_seconds, name="native_tr_seconds")
        if tr <= 0:
            raise RawCacheError("native_tr_seconds must be positive")
        expected_tr = NATIVE_TR_SECONDS[self.dataset]
        if not math.isclose(tr, expected_tr, rel_tol=0.0, abs_tol=1e-8):
            raise RawCacheError(
                f"{self.dataset} native TR is fixed at {expected_tr} seconds; "
                f"received {tr}"
            )
        object.__setattr__(self, "native_tr_seconds", tr)
        if isinstance(self.n_bins, bool) or int(self.n_bins) <= 0 or int(self.n_bins) != self.n_bins:
            raise RawCacheError("n_bins must be a positive integer")
        object.__setattr__(self, "n_bins", int(self.n_bins))
        object.__setattr__(self, "require_text", bool(self.require_text))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, dataset: str | None = None) -> "NativeTimelineSpec":
        if not isinstance(value, Mapping):
            raise RawCacheError("timeline specification must be an object")
        timeline_id = value.get("timeline_id", value.get("run_id"))
        if timeline_id is None:
            raise RawCacheError("timeline specification requires timeline_id (or run_id)")
        return cls(
            timeline_id=str(timeline_id),
            dataset=str(value.get("dataset", dataset or "")),
            origin_seconds=value.get("origin_seconds", value.get("time_origin_seconds", 0.0)),
            native_tr_seconds=value.get("native_tr_seconds", value.get("tr_seconds")),
            n_bins=value.get("n_bins", value.get("n_timepoints")),
            require_text=bool(value.get("require_text", False)),
        )


def _coerce_specs(specs: Mapping[str, Any] | Iterable[NativeTimelineSpec | Mapping[str, Any]], *, dataset: str | None = None) -> tuple[NativeTimelineSpec, ...]:
    if isinstance(specs, Mapping):
        values: list[Any] = []
        for key, value in specs.items():
            if isinstance(value, NativeTimelineSpec):
                values.append(value)
            elif isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("timeline_id", str(key))
                values.append(payload)
            else:
                raise RawCacheError(f"Timeline spec {key!r} must be an object")
    else:
        values = list(specs)
    output: list[NativeTimelineSpec] = []
    seen: set[str] = set()
    for value in values:
        spec = value if isinstance(value, NativeTimelineSpec) else NativeTimelineSpec.from_mapping(value, dataset=dataset)
        if spec.timeline_id in seen:
            raise RawCacheError(f"Duplicate timeline specification: {spec.timeline_id}")
        seen.add(spec.timeline_id)
        output.append(spec)
    if not output:
        raise RawCacheError("At least one native timeline specification is required")
    return tuple(sorted(output, key=lambda item: item.timeline_id))


def iter_indexed_timelines(
    index_path: str | Path,
    timeline_specs: Mapping[str, Any] | Iterable[NativeTimelineSpec | Mapping[str, Any]],
    *,
    allow_unrequested: bool = False,
) -> Iterator[tuple[NativeTimelineSpec, NativeStimulusTimeline]]:
    """Yield one pooled native timeline at a time from an indexed cache.

    ``timeline_specs`` is validated before the index is opened.  The index is
    first reduced to compact per-timeline record metadata; array artifacts are
    loaded only while building the currently yielded timeline.  In exact mode
    (the default), requested and indexed timeline IDs must match. A limited run
    may explicitly set ``allow_unrequested=True`` to select a subset from a
    larger index; requested timelines may never be missing.
    """

    # This is intentionally outside the nested iterator.  It gives callers a
    # deterministic timing/configuration error before any index filesystem I/O.
    specs = _coerce_specs(timeline_specs)

    def _iter() -> Iterator[tuple[NativeTimelineSpec, NativeStimulusTimeline]]:
        # The index read validates contained paths and structural stats without
        # loading all arrays. Selected arrays are loaded one timeline at a time.
        records = read_chunk_index(index_path)
        by_timeline: dict[str, dict[str, list[ChunkIndexRecord]]] = {}
        for record in records:
            by_timeline.setdefault(record.timeline_id, {}).setdefault(
                record.modality, []
            ).append(record)
        for modality_groups in by_timeline.values():
            for modality, modality_records in modality_groups.items():
                modality_groups[modality] = sorted(
                    modality_records, key=lambda item: item.identity
                )

        known = set(by_timeline)
        requested = {spec.timeline_id for spec in specs}
        missing = sorted(requested - known)
        if missing:
            raise RawCacheError(
                f"Native timeline specs have no indexed chunks: {missing}"
            )
        extra = sorted(known - requested)
        if extra and not allow_unrequested:
            raise RawCacheError(
                "Raw-cache index contains timelines without explicit native timing "
                f"specs: {extra[:10]}"
            )

        for spec in specs:
            # Pass generators into build_native_timeline.  It materializes only
            # the current modality/timeline, and no previous timeline remains
            # reachable after this generator resumes past the yield.
            def _loaded_chunks(
                modality_records: Sequence[ChunkIndexRecord],
            ) -> Iterator[ModalityChunk]:
                for record in modality_records:
                    yield record.load().modality_chunk()

            chunk_groups = {
                modality: _loaded_chunks(modality_records)
                for modality, modality_records in by_timeline[spec.timeline_id].items()
            }
            timeline: NativeStimulusTimeline | None = None
            try:
                timeline = build_native_timeline(
                    dataset=spec.dataset,
                    timeline_id=spec.timeline_id,
                    chunks=chunk_groups,
                    origin_seconds=spec.origin_seconds,
                    native_tr_seconds=spec.native_tr_seconds,
                    n_bins=spec.n_bins,
                    require_text=spec.require_text,
                )
                yield spec, timeline
            finally:
                # The consumer may retain the yielded object deliberately, but
                # this frame must not keep a second copy while moving on.
                chunk_groups.clear()
                del chunk_groups
                timeline = None

    return _iter()


__all__ = [
    "NATIVE_TR_SECONDS",
    "RAW_CACHE_INDEX_NAME",
    "RAW_CACHE_SCHEMA_VERSION",
    "ChunkIndexRecord",
    "NativeTimelineSpec",
    "RawCacheChunk",
    "RawCacheError",
    "RawCacheIntegrityError",
    "RawCacheTransactionError",
    "chunks_from_explicit_extractor",
    "export_raw_chunks",
    "iter_indexed_timelines",
    "iter_tribev2_raw_chunks",
    "load_explicit_extractor",
    "neuralset_batch_to_chunk",
    "read_chunk_index",
    "sample_center_timestamps",
    "support_mask_from_events",
    "support_mask_from_intervals",
]
