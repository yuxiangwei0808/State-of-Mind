"""Shared file I/O for offline preparation and native-TR runtime loading."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
import numpy as np

class BundleValidationError(ValueError):
    """Raised when a bundle or one of its stable artifacts is invalid."""


def utc_now() -> str:
    """Return a canonical UTC timestamp suitable for manifests and ledgers."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def safe_run_filename(run_id: str, *, suffix: str = ".npy") -> str:
    """Map an arbitrary stable ID to a reversible, contained filename.

    The old implementation appended a truncated SHA-256 digest.  That made
    otherwise human-readable paths impossible to recover and forced every
    caller that only needed a filename to perform a content-independent hash.
    Percent-encoding is injective and keeps the original identifier available
    to readers (``urllib.parse.unquote``) without ever introducing a path
    separator.  Do not truncate the encoded value: truncation would recreate
    collisions.  Very long identifiers are rejected before they can hit a
    platform-specific filename limit.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    normalized = run_id.strip()
    if not isinstance(suffix, str) or "/" in suffix or "\\" in suffix or "\x00" in suffix:
        raise ValueError("suffix must be a single filename suffix")
    encoded = quote(normalized, safe="-_.~")
    # A suffix-less ``..`` would otherwise be interpreted as the parent
    # directory by ``Path``.  Encode dots only for these two special values;
    # ordinary IDs remain pleasantly readable while the mapping stays
    # reversible (the caller removes the suffix and applies ``unquote``).
    if encoded in {".", ".."}:
        encoded = quote(normalized, safe="-_~")
    filename = f"{encoded}{suffix}"
    if len(filename.encode("utf-8")) > 240:
        raise ValueError("run_id is too long for a safe bundle filename")
    return filename


def contained_path(
    root: str | Path,
    relative: str | Path,
    *,
    must_exist: bool = False,
    kind: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Resolve a manifest path while proving that it remains under ``root``.

    Existing symlinks are resolved before containment is checked.  Absolute,
    parent-traversing, empty, and NUL-containing paths are rejected even when
    they happen to resolve inside the root, keeping manifests portable and
    unambiguous.
    """

    root_path = Path(root).expanduser().resolve(strict=False)
    raw = os.fspath(relative)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise BundleValidationError("Manifest path must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute() or candidate == Path(".") or ".." in candidate.parts:
        raise BundleValidationError(f"Path must be bundle-relative and contained: {raw!r}")
    if suffix is not None and candidate.suffix.lower() != suffix.lower():
        raise BundleValidationError(f"Path must have suffix {suffix!r}: {raw!r}")
    resolved = (root_path / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise BundleValidationError(f"Path escapes bundle root: {raw!r}") from exc
    if must_exist and not resolved.exists():
        raise BundleValidationError(f"Bundle artifact does not exist: {resolved}")
    if kind == "file" and must_exist and not resolved.is_file():
        raise BundleValidationError(f"Bundle artifact is not a regular file: {resolved}")
    if kind == "directory" and must_exist and not resolved.is_dir():
        raise BundleValidationError(f"Bundle artifact is not a directory: {resolved}")
    if kind not in {None, "file", "directory"}:
        raise ValueError("kind must be None, 'file', or 'directory'")
    return resolved


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _commit_temp(temp: Path, target: Path, *, overwrite: bool) -> None:
    try:
        if overwrite:
            os.replace(temp, target)
        else:
            # A hard-link commit is atomic and, unlike os.replace, fails rather
            # than clobbering a target created by a concurrent process.
            try:
                os.link(temp, target)
            except FileExistsError as exc:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {target}") from exc
            os.unlink(temp)
        _fsync_directory(target.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _temporary_path(target: Path) -> tuple[int, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(name)


def atomic_write_bytes(
    path: str | Path, value: bytes, *, overwrite: bool = False
) -> Path:
    target = Path(path)
    descriptor, temp = _temporary_path(target)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _commit_temp(temp, target, overwrite=overwrite)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any] | Sequence[Any],
    *,
    overwrite: bool = False,
) -> Path:
    try:
        encoded = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(f"Value is not strict JSON: {exc}") from exc
    return atomic_write_bytes(path, encoded, overwrite=overwrite)


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    lines: list[str] = []
    try:
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise TypeError(f"JSONL record {index} is not an object")
            lines.append(
                json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
            )
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(f"Cannot encode strict JSONL: {exc}") from exc
    payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def atomic_write_npy(
    path: str | Path,
    array: Any,
    *,
    dtype: np.dtype[Any] | str | type | None = np.float32,
    ndim: int | None = None,
    finite: bool = True,
    overwrite: bool = False,
) -> Path:
    """Atomically write a safe, C-contiguous, non-pickled NumPy array."""

    value = np.asarray(array, dtype=dtype, order="C")
    if value.dtype.hasobject:
        raise BundleValidationError("Object arrays are forbidden in stable bundles")
    if ndim is not None and value.ndim != ndim:
        raise BundleValidationError(f"Expected a {ndim}-D array, got shape {value.shape}")
    if finite and np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise BundleValidationError("Stable bundle arrays must contain only finite values")
    value = np.ascontiguousarray(value)
    target = Path(path)
    descriptor, temp = _temporary_path(target)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _commit_temp(temp, target, overwrite=overwrite)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Cannot read strict JSON {path}: {exc}") from exc


def read_json_object(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    allowed_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    file_path = Path(path)
    value = _json_load(file_path)
    if not isinstance(value, dict):
        raise BundleValidationError(f"JSON root must be an object: {file_path}")
    required = set(required_keys)
    missing = required - set(value)
    if missing:
        raise BundleValidationError(f"{file_path} is missing fields: {sorted(missing)}")
    if allowed_keys is not None:
        unknown = set(value) - set(allowed_keys)
        if unknown:
            raise BundleValidationError(f"{file_path} has unknown fields: {sorted(unknown)}")
    return value


def read_jsonl(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleValidationError(f"Cannot read JSONL {file_path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    required = set(required_keys)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BundleValidationError(
                f"Blank JSONL record at {file_path}:{line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleValidationError(
                f"Invalid JSONL at {file_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise BundleValidationError(
                f"JSONL record must be an object at {file_path}:{line_number}"
            )
        missing = required - set(value)
        if missing:
            raise BundleValidationError(
                f"JSONL record {line_number} is missing fields: {sorted(missing)}"
            )
        records.append(value)
    if not records and not allow_empty:
        raise BundleValidationError(f"JSONL file is empty: {file_path}")
    return records
