"""Hard-replacement offline Schaefer-1000 preparation pipeline.

This module is intentionally small.  Source discovery, atlas validation,
stimulus pooling, fMRI preparation, and bundle storage remain owned by their
respective modules; this file only wires their contracts together.  It keeps
all inputs on the raw native-clock feature path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuroworld.nbs_data import bundle as _bundle
from neuroworld.nbs_data.fmri import _adapters, _sources
from neuroworld.nbs_data.fmri import atlas as _atlas
from neuroworld.nbs_data.stimuli import (
    algonauts as _stimulus_algonauts,
    hcp_movies as _stimulus_hcp_movies,
    raw_cache as _raw_cache,
)
from neuroworld.nbs_data.stimuli.raw_cache import NativeTimelineSpec

# Public module handles keep dependency seams explicit and easy to monkeypatch.
sources = _sources
atlas = _atlas
raw_cache = _raw_cache
adapters = _adapters
bundle = _bundle


EXPECTED_RUN_COUNTS = dict(_sources.EXPECTED_RUN_COUNTS)
DATASET_MODALITIES = dict(_bundle.DATASET_MODALITIES)
REQUIRED_MASKS = frozenset(_bundle.REQUIRED_MASKS)
_STIMULUS_DATASET_MODULES = {
    "algonauts": _stimulus_algonauts,
    "hcp_movies": _stimulus_hcp_movies,
}


class PipelineError(ValueError):
    """Raised when an offline pipeline contract is not satisfied."""


# These wrappers are deliberately public and patch-friendly.  A caller can
# monkeypatch either this module or the owning module in a synthetic test.
def discover_runs(*args: Any, **kwargs: Any) -> Any:
    return _sources.discover_runs(*args, **kwargs)


def audit_run_sources(*args: Any, **kwargs: Any) -> Any:
    return _sources.audit_run_sources(*args, **kwargs)


def load_atlas(*args: Any, **kwargs: Any) -> Any:
    return _atlas.load_atlas(*args, **kwargs)


def resample_labels_nearest(*args: Any, **kwargs: Any) -> Any:
    return _atlas.resample_labels_nearest(*args, **kwargs)


def prepare_run(*args: Any, **kwargs: Any) -> Any:
    return _adapters.prepare_run(*args, **kwargs)


def read_chunk_index(*args: Any, **kwargs: Any) -> Any:
    return _raw_cache.read_chunk_index(*args, **kwargs)


BundleWriter = _bundle.BundleWriter
BundlePaths = _bundle.BundlePaths
committed_run_ids = _bundle.committed_run_ids
read_json_object = _bundle.read_json_object


def _invoke(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function while allowing compact test doubles with fewer kwargs."""

    if not kwargs:
        return function(*args)
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return function(*args, **kwargs)
    accepted = {parameter.name for parameter in parameters}
    return function(*args, **{key: value for key, value in kwargs.items() if key in accepted})


def _canonical_dataset(value: str) -> str:
    aliases = {
        "algonaut": "algonauts",
        "algonauts_2025": "algonauts",
        "hcp": "hcp_movies",
        "hcp_movie": "hcp_movies",
        "hcp_movies": "hcp_movies",
    }
    if not isinstance(value, str) or not value.strip():
        raise PipelineError("dataset_id must be a non-empty string")
    dataset = aliases.get(value.strip().lower().replace("-", "_"), value.strip().lower())
    if dataset not in EXPECTED_RUN_COUNTS:
        raise PipelineError(
            "Unsupported dataset_id {!r}; expected {}".format(
                value, sorted(EXPECTED_RUN_COUNTS)
            )
        )
    return dataset


def _path(value: Any, *, name: str) -> Path:
    if value is None:
        raise PipelineError("{} is required".format(name))
    return Path(value).expanduser().resolve(strict=False)


def _limit_value(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError("limit must be a positive integer")
    return value


def _worker_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError("workers must be a positive integer")
    return value


def _source_id(source: Any) -> str:
    value = getattr(source, "run_id", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("run_id")
    if value is None or not str(value).strip():
        raise PipelineError("discovered source is missing run_id")
    return str(value).strip()


def _timeline_id(source: Any) -> str:
    value = getattr(source, "timeline_id", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("timeline_id", source.get("run_id"))
    if value is None or not str(value).strip():
        raise PipelineError("source is missing timeline_id")
    return str(value).strip()


def _subject_id(source: Any) -> str:
    value = getattr(source, "subject_id", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("subject_id")
    if value is None or not str(value).strip():
        raise PipelineError("source is missing subject_id")
    return str(value).strip()


def _tr(source: Any) -> float:
    value = getattr(source, "tr_seconds", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("tr_seconds")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError("source has an invalid TR") from exc
    if not np.isfinite(result) or result <= 0:
        raise PipelineError("source TR must be finite and positive")
    return result


def _n_volumes(source: Any) -> int:
    value = getattr(source, "n_volumes", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("n_volumes", source.get("n_timepoints"))
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise PipelineError("source volume count must be a finite integer")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise PipelineError("source volume count must be a finite integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError("source has an invalid volume count") from exc
    if not np.isfinite(numeric) or numeric != result:
        raise PipelineError("source volume count must be a finite integer")
    if result <= 0:
        raise PipelineError("source volume count must be positive")
    return result


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
    """Drop retired SHA/hash provenance from active metadata."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_retired_hash_key(key_text):
                continue
            result[key_text] = _without_hash_fields(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_without_hash_fields(item) for item in value]
    return value


def _file_stats(value: Any, *, name: str, allow_missing: bool = False) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        candidate = Path(value).expanduser().resolve(strict=False)
    except (TypeError, ValueError):
        if allow_missing:
            return None
        raise PipelineError("{} is not a valid path".format(name))
    try:
        stat = candidate.stat()
    except OSError as exc:
        if allow_missing:
            return None
        raise PipelineError("cannot stat {}: {}".format(name, exc)) from exc
    if not candidate.is_file():
        if allow_missing:
            return None
        raise PipelineError("{} is not a regular file: {}".format(name, candidate))
    return {
        "path": str(candidate),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_dict(source: Any) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(_without_hash_fields(source))
    as_dict = getattr(source, "as_dict", None)
    if callable(as_dict):
        value = as_dict()
        if isinstance(value, Mapping):
            return dict(_without_hash_fields(value))
    return {
        "dataset_id": str(getattr(source, "dataset_id", "")),
        "run_id": _source_id(source),
        "timeline_id": _timeline_id(source),
        "subject_id": _subject_id(source),
        "tr_seconds": _tr(source),
        "n_volumes": _n_volumes(source),
    }


def _source_identity(source: Any) -> Dict[str, Any]:
    """Return exact source paths and cheap filesystem identities.

    Source reuse is guarded by path, byte size, and nanosecond mtime.  This is
    deliberately a structural identity: the active Schaefer contract does not
    read or manufacture content digests.
    """

    result = _source_dict(source)
    required_paths = getattr(source, "required_paths", None)
    if required_paths is None and isinstance(source, Mapping):
        required_paths = source.get("required_paths")
    if required_paths is not None:
        try:
            resolved = [Path(value).expanduser().resolve(strict=False) for value in required_paths]
        except (TypeError, ValueError):
            resolved = [Path(str(value)).expanduser().resolve(strict=False) for value in required_paths]
        result["required_paths"] = [str(value) for value in resolved]
        result["required_files"] = [
            _file_stats(value, name="source file", allow_missing=True) for value in resolved
        ]
    # ``NativeRunSource.as_dict`` keeps source-relative labels convenient for
    # discovery reports.  Published run records need exact paths so a resume
    # can detect a moved input without reconstructing the source root.
    if not isinstance(source, Mapping):
        for attribute in (
            "fmri_path",
            "sidecar_json",
            "brain_mask",
            "confounds_tsv",
            "motion_regressors",
            "tissue_labels",
        ):
            value = getattr(source, attribute, None)
            if value is not None:
                result[attribute] = str(_path(value, name="source " + attribute))
    return dict(_without_hash_fields(result))


def _source_role_path(source: Any, role: str, *, stored_path: Any = None) -> Optional[Path]:
    attribute_by_role = {
        "fmri": "fmri_path",
        "sidecar_json": "sidecar_json",
        "brain_mask": "brain_mask",
        "confounds_tsv": "confounds_tsv",
        "motion_regressors": "motion_regressors",
        "tissue_labels": "tissue_labels",
    }
    if role in attribute_by_role:
        attribute = attribute_by_role[role]
        value = getattr(source, attribute, None)
        if value is None and isinstance(source, Mapping):
            value = source.get(attribute)
        if value is None:
            return None
        return _path(value, name="source {}".format(role))
    return None


def _validate_preprocessing_source(
    record: Mapping[str, Any],
    source: Any,
    *,
    run_id: str,
) -> None:
    preprocessing = record.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise PipelineError("committed run preprocessing is invalid for {}".format(run_id))
    source_provenance = preprocessing.get("source")
    if source_provenance is None:
        raise PipelineError("committed run preprocessing source is missing for {}".format(run_id))
    if not isinstance(source_provenance, Mapping):
        raise PipelineError("committed run preprocessing source is invalid for {}".format(run_id))
    files = source_provenance.get("files", [])
    if not isinstance(files, list):
        raise PipelineError("committed run preprocessing source files are invalid for {}".format(run_id))
    for file_record in files:
        if not isinstance(file_record, Mapping):
            raise PipelineError("committed run preprocessing file record is invalid for {}".format(run_id))
        role = str(file_record.get("role", ""))
        # The atlas cache is validated against bundle provenance separately;
        # its path is deliberately relative to the output bundle.
        if role == "schaefer_atlas":
            continue
        current_path = _source_role_path(source, role, stored_path=file_record.get("path"))
        if current_path is None:
            raise PipelineError("committed run source role mismatch for {}".format(run_id))
        recorded_path = file_record.get("path")
        try:
            if _path(recorded_path, name="recorded source").resolve(strict=False) != current_path:
                raise PipelineError("committed run source path mismatch for {}".format(run_id))
        except (TypeError, ValueError):
            raise PipelineError("committed run source path mismatch for {}".format(run_id))
        try:
            current_stat = current_path.stat()
            if int(file_record.get("size_bytes")) != int(current_stat.st_size):
                raise PipelineError("committed run source size mismatch for {}".format(run_id))
            if int(file_record.get("mtime_ns")) != int(current_stat.st_mtime_ns):
                raise PipelineError("committed run source mtime mismatch for {}".format(run_id))
        except (OSError, TypeError, ValueError) as exc:
            raise PipelineError("committed run source artifact metadata mismatch for {}".format(run_id)) from exc
        # A record containing a retired hash field is never accepted on resume.
        if any(_is_retired_hash_key(key) for key in file_record):
            raise PipelineError("committed run source contains retired hash metadata for {}".format(run_id))


def _raw_index_identity(value: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return the raw-index path and current structural file stats."""

    if isinstance(value, (str, Path)):
        path = _path(value, name="raw_index")
        return str(path), _file_stats(path, name="raw_index", allow_missing=True)
    return None, None


def _raw_record_field(record: Any, name: str, default: Any = None) -> Any:
    value = getattr(record, name, default)
    if value is default and isinstance(record, Mapping):
        value = record.get(name, default)
    return value


def _raw_index_content_identity(value: Any) -> Optional[Dict[str, Any]]:
    """Return only sorted timeline IDs and exact structural counts."""

    index_path, index_stats = _raw_index_identity(value)
    if index_path is None or index_stats is None:
        return None
    try:
        records = tuple(_invoke(read_chunk_index, Path(index_path)))
    except Exception as exc:
        raise PipelineError(
            "cannot read raw cache index {}: {}".format(index_path, exc)
        ) from exc
    modalities_by_timeline: Dict[str, set[str]] = {}
    chunk_counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        timeline = str(_raw_record_field(record, "timeline_id", ""))
        modality = str(_raw_record_field(record, "modality", ""))
        modalities_by_timeline.setdefault(timeline, set()).add(modality)
        modality_counts = chunk_counts.setdefault(timeline, {})
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    timeline_ids = sorted(modalities_by_timeline)
    modalities = {
        timeline: sorted(modalities_by_timeline[timeline])
        for timeline in timeline_ids
    }
    ordered_counts = {
        timeline: {
            modality: chunk_counts[timeline][modality]
            for modality in modalities[timeline]
        }
        for timeline in timeline_ids
    }
    return {
        "index": index_stats,
        "record_count": len(records),
        "timeline_count": len(timeline_ids),
        "timelines": timeline_ids,
        "modalities": modalities,
        "chunk_counts": ordered_counts,
    }


def _hcp_blockers(runs: Sequence[Any]) -> List[str]:
    blockers: List[str] = []
    for source in runs:
        run_id = _source_id(source)
        for attribute, label in (("brain_mask", "brainmask"), ("tissue_labels", "wmparc")):
            value = getattr(source, attribute, None)
            if value is None and attribute == "tissue_labels":
                value = getattr(source, "wmparc", None)
            if value is None and isinstance(source, Mapping):
                value = source.get(attribute, source.get("wmparc") if attribute == "tissue_labels" else None)
            try:
                present = value is not None and Path(value).is_file()
            except (TypeError, ValueError):
                present = False
            if not present:
                blockers.append("HCP run {} is missing {}".format(run_id, label))
    return blockers


def _grid_descriptor(value: Any) -> Optional[Dict[str, Any]]:
    image = _grid_image(value)
    if image is None:
        return None
    try:
        shape = [int(size) for size in image.shape[:3]]
        affine = np.asarray(image.affine, dtype=np.float64)
        if len(shape) != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
            return None
        dtype = None
        header = getattr(image, "header", None)
        if header is not None and hasattr(header, "get_data_dtype"):
            dtype = np.dtype(header.get_data_dtype()).name
        if dtype is None:
            dataobj = getattr(image, "dataobj", None)
            if dataobj is not None and hasattr(dataobj, "dtype"):
                dtype = np.dtype(dataobj.dtype).name
        return {"shape": shape, "affine": affine.tolist(), "dtype": dtype}
    except Exception:
        return None


def _atlas_provenance(atlas_value: Any, *, label_image: Any = None) -> Dict[str, Any]:
    """Describe atlas inputs using paths, stats, geometry, and label semantics."""

    if atlas_value is None:
        return {}
    if isinstance(atlas_value, Mapping):
        manifest = atlas_value.get("manifest", {})
        manifest_path = atlas_value.get("manifest_path")
    else:
        manifest = getattr(atlas_value, "manifest", {})
        manifest_path = getattr(atlas_value, "manifest_path", None)
    if not isinstance(manifest, Mapping):
        manifest = {}
    manifest_stats = _file_stats(
        manifest_path, name="atlas manifest", allow_missing=True
    )
    source_stats = _file_stats(
        label_image, name="atlas label image", allow_missing=True
    )
    source_grid = _grid_descriptor(label_image)
    result: Dict[str, Any] = {
        "artifact": str(manifest.get("artifact", "schaefer2018_1000parcels_7networks")),
        "observation_space": str(manifest.get("observation_space", "schaefer1000")),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_stats": manifest_stats,
        "source_path": None if label_image is None else str(label_image),
        "source_stats": source_stats,
        "source_grid": source_grid,
    }
    # Keep atlas input records discoverable without copying a potentially
    # large manifest into every run record.  Retired hash fields are scrubbed.
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping):
        result["inputs"] = _without_hash_fields(
            {str(key): dict(value) if isinstance(value, Mapping) else str(value)
             for key, value in inputs.items()}
        )
    return dict(_without_hash_fields(result))


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _raw_index_audit(
    raw_index: Any,
    runs: Sequence[Any],
    *,
    dataset: str,
    selection: Optional[Tuple[str, ...]],
) -> Dict[str, Any]:
    index_path, index_stats = _raw_index_identity(raw_index)
    report: Dict[str, Any] = {
        "index_path": index_path,
        "index_stats": index_stats,
        "record_count": 0,
        "timeline_count": 0,
        "timelines": [],
        "modalities": {},
        "missing_timelines": [],
        "missing_modalities": {},
        "blockers": [],
        "ready": False,
    }
    if raw_index is None:
        report["blockers"] = ["raw chunk index is required"]
        return report
    try:
        records = tuple(
            _invoke(read_chunk_index, raw_index)
            if isinstance(raw_index, (str, Path))
            else tuple(raw_index)
        )
    except Exception as exc:
        report["blockers"] = ["raw chunk index audit failed: {}".format(exc)]
        return report
    by_timeline: Dict[str, set] = {}
    for record in records:
        timeline = str(_record_value(record, "timeline_id", "")).strip()
        modality = str(_record_value(record, "modality", "")).strip().lower()
        if not timeline:
            report["blockers"].append("raw chunk index contains a record without timeline_id")
            continue
        by_timeline.setdefault(timeline, set()).add(modality)
    expected_timelines = {_timeline_id(source) for source in runs}
    expected_modalities = set(DATASET_MODALITIES[dataset])
    missing_timelines = sorted(expected_timelines - set(by_timeline))
    report["missing_timelines"] = missing_timelines
    for timeline in sorted(expected_timelines & set(by_timeline)):
        missing = sorted(expected_modalities - by_timeline[timeline])
        if missing:
            report["missing_modalities"][timeline] = missing
    if missing_timelines:
        report["blockers"].append(
            "raw chunk index is missing timelines: {}".format(missing_timelines[:10])
        )
    if report["missing_modalities"]:
        report["blockers"].append("raw chunk index is missing required modalities")
    # A limited run may legitimately point at a full cache. A full run must not
    # silently accept an index whose timeline membership differs from its run
    # inventory.
    if selection is None:
        extra = sorted(set(by_timeline) - expected_timelines)
        if extra:
            report["blockers"].append(
                "raw chunk index contains unselected timelines: {}".format(extra[:10])
            )
    report["record_count"] = len(records)
    report["timeline_count"] = len(by_timeline)
    report["timelines"] = sorted(by_timeline)
    report["modalities"] = {
        timeline: sorted(modalities) for timeline, modalities in sorted(by_timeline.items())
    }
    if index_path is not None and index_stats is not None:
        report["identity"] = _raw_index_content_identity(raw_index)
    report["ready"] = not report["blockers"]
    return dict(_without_hash_fields(report))


@dataclass
class _Context:

    dataset: str
    mode: str
    full_inventory: bool
    selection: Optional[Tuple[str, ...]]
    runs: Tuple[Any, ...]
    expected_run_count: int
    atlas_value: Any
    atlas_provenance: Dict[str, Any]
    source_audit: Dict[str, Any]
    raw_audit: Dict[str, Any]
    blockers: List[str] = field(default_factory=list)

    def report(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset,
            "mode": self.mode,
            "full_inventory": self.full_inventory,
            "selection": None if self.selection is None else list(self.selection),
            "run_count": len(self.runs),
            "expected_run_count": self.expected_run_count,
            "run_ids": sorted(_source_id(source) for source in self.runs),
            "source_audit": self.source_audit,
            "atlas": self.atlas_provenance,
            "raw_cache": self.raw_audit,
            "blockers": list(self.blockers),
            "ready": not self.blockers,
        }


def _context(
    dataset_id: str,
    source_root: Any,
    atlas_manifest: Any,
    raw_index: Any,
    *,
    limit: Optional[int],
    audit_raw: bool,
) -> _Context:
    dataset = _canonical_dataset(dataset_id)
    requested_limit = _limit_value(limit)
    root = _path(source_root, name="source_root")
    discovered = tuple(
        _invoke(
            discover_runs,
            dataset,
            root,
            strict=True,
        )
    )
    if not discovered:
        raise PipelineError("source discovery returned no runs")
    ordered = tuple(sorted(discovered, key=_source_id))
    if requested_limit is not None and requested_limit > len(ordered):
        raise PipelineError(
            "limit {} exceeds the {} discovered runs".format(
                requested_limit, len(ordered)
            )
        )
    runs = ordered if requested_limit is None else ordered[:requested_limit]
    selected = None if requested_limit is None else tuple(_source_id(run) for run in runs)
    full_inventory = requested_limit is None
    mode = "full" if full_inventory else "limited"
    source_audit = dict(
        _invoke(
            audit_run_sources,
            runs,
            require_full_inventory=full_inventory,
        )
    )
    blockers = [str(value) for value in source_audit.get("blockers", [])]
    expected = EXPECTED_RUN_COUNTS[dataset] if full_inventory else len(runs)
    if full_inventory and len(runs) != EXPECTED_RUN_COUNTS[dataset]:
        message = "expected {} runs, found {}".format(EXPECTED_RUN_COUNTS[dataset], len(runs))
        if message not in blockers:
            blockers.append(message)
    if dataset == "hcp_movies":
        blockers.extend(_hcp_blockers(runs))

    atlas_value: Any = None
    atlas_info: Dict[str, Any] = {}
    if dataset != "algonauts":
        if atlas_manifest is None:
            blockers.append("atlas manifest is required for volume sources")
        else:
            try:
                atlas_value = (
                    atlas_manifest
                    if hasattr(atlas_manifest, "mni_label_image")
                    or hasattr(atlas_manifest, "hcp_label_image")
                    else _invoke(load_atlas, atlas_manifest)
                )
                selected_label = getattr(atlas_value, "hcp_label_image", None)
                if selected_label is None and isinstance(atlas_value, Mapping):
                    selected_label = atlas_value.get("hcp_label_image")
                if selected_label is None:
                    raise PipelineError("loaded atlas has no hcp_label_image")
                atlas_info = _atlas_provenance(atlas_value, label_image=selected_label)
            except Exception as exc:
                blockers.append("atlas load failed: {}".format(exc))
    raw_report = (
        _raw_index_audit(
            raw_index,
            runs,
            dataset=dataset,
            selection=selected,
        )
        if audit_raw
        else {
            "index_path": _raw_index_identity(raw_index)[0],
            "index_stats": _raw_index_identity(raw_index)[1],
            "audited": False,
            "blockers": [],
            "ready": raw_index is not None,
        }
    )
    if audit_raw:
        blockers.extend(str(value) for value in raw_report.get("blockers", []))
    return _Context(
        dataset=dataset,
        mode=mode,
        full_inventory=full_inventory,
        selection=selected,
        runs=runs,
        expected_run_count=expected,
        atlas_value=atlas_value,
        atlas_provenance=atlas_info,
        source_audit=source_audit,
        raw_audit=raw_report,
        blockers=sorted(set(blockers)),
    )


def prepare(
    dataset_id: str,
    *,
    source_root: Any,
    raw_index: Any,
    atlas_manifest: Any = None,
) -> Dict[str, Any]:
    """Dry-run the complete source, atlas, and raw-feature inventory."""

    context = _context(
        dataset_id,
        source_root,
        atlas_manifest,
        raw_index,
        limit=None,
        audit_raw=True,
    )
    return context.report()


def _raw_cache_iterator(
    index_path: Any,
    timeline_specs: Sequence[NativeTimelineSpec | Mapping[str, Any]],
    *,
    allow_unrequested: bool = False,
) -> Iterator[Any]:
    """Delegate to the one-timeline-at-a-time raw-cache API."""

    function = getattr(_raw_cache, "iter_indexed_timelines", None)
    if function is None:
        raise PipelineError("raw_cache.iter_indexed_timelines is unavailable")
    result = _invoke(
        function,
        index_path,
        timeline_specs,
        allow_unrequested=allow_unrequested,
    )
    yield from result


def iter_indexed_timelines(
    index_path: Any,
    timeline_specs: Sequence[NativeTimelineSpec | Mapping[str, Any]],
    *,
    allow_unrequested: bool = False,
) -> Iterator[Any]:
    """Public patch point for the raw-cache one-timeline-at-a-time iterator."""

    yield from _raw_cache_iterator(
        index_path,
        timeline_specs,
        allow_unrequested=allow_unrequested,
    )


def _timeline_specs(runs: Sequence[Any], dataset: str) -> List[NativeTimelineSpec]:
    stimulus_module = _STIMULUS_DATASET_MODULES[_canonical_dataset(dataset)]
    by_id: Dict[str, NativeTimelineSpec] = {}
    for source in runs:
        timeline = _timeline_id(source)
        spec = stimulus_module.timeline_spec(
            timeline,
            origin_seconds=0.0,
            n_bins=_n_volumes(source),
        )
        previous = by_id.get(timeline)
        if previous is not None and (
            previous.native_tr_seconds != spec.native_tr_seconds
            or previous.n_bins != spec.n_bins
        ):
            raise PipelineError("runs sharing timeline {} disagree on timing".format(timeline))
        by_id[timeline] = spec
    return [by_id[key] for key in sorted(by_id)]


def _flatten_timeline(item: Any) -> Iterator[Any]:
    if isinstance(item, Mapping) and "timeline_id" not in item and "modalities" not in item:
        # Some compact iterators yield {timeline_id: NativeStimulusTimeline}
        # for each batch.  Flatten that representation without interpreting a
        # native timeline's own metadata mapping as a collection.
        values = list(item.values())
        if values and all(hasattr(value, "modalities") or isinstance(value, Mapping) for value in values):
            for value in values:
                yield from _flatten_timeline(value)
            return
    if isinstance(item, tuple) and len(item) == 2 and not hasattr(item, "modalities"):
        first, second = item
        first_id = first if isinstance(first, str) else getattr(first, "timeline_id", None)
        if first_id is not None and (
            hasattr(second, "modalities") or isinstance(second, Mapping)
        ):
            yield from _flatten_timeline(second)
            return
    yield item


def _timeline_modalities(timeline: Any) -> Mapping[str, Any]:
    if isinstance(timeline, Mapping):
        value = timeline.get("modalities")
        if value is None:
            value = timeline.get("features")
    else:
        value = getattr(timeline, "modalities", None)
    if not isinstance(value, Mapping):
        raise PipelineError("raw timeline is missing its modalities mapping")
    return value


def _timeline_value(timeline: Any, key: str, default: Any = None) -> Any:
    if isinstance(timeline, Mapping):
        return timeline.get(key, default)
    return getattr(timeline, key, default)


def _coerce_bool_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, order="C")
    except (TypeError, ValueError) as exc:
        raise PipelineError("{} must be boolean [{}]".format(label, length)) from exc
    if result.shape != (length,) or result.dtype != np.bool_:
        raise PipelineError("{} must be boolean [{}]".format(label, length))
    return np.ascontiguousarray(result, dtype=np.bool_)


def _feature_array(value: Any, modality: str) -> np.ndarray:
    if isinstance(value, Mapping):
        array = value.get("values", value.get("array"))
        support = value.get("support")
    else:
        array = getattr(value, "values", value)
        support = getattr(value, "support", None)
    if array is None:
        raise PipelineError("{} timeline modality has no values".format(modality))
    result = np.asarray(array, dtype=np.float32, order="C")
    expected_dim = _bundle.MODALITY_DIMS[modality]
    if result.ndim != 3 or result.shape[1:] != (2, expected_dim):
        raise PipelineError(
            "{} features must have shape [T,2,{}], found {}".format(
                modality, expected_dim, result.shape
            )
        )
    if result.shape[0] <= 0 or not np.isfinite(result).all():
        raise PipelineError("{} features must be non-empty and finite".format(modality))
    if support is not None:
        _coerce_bool_vector(
            support,
            length=result.shape[0],
            label="{} support".format(modality),
        )
    return result


def _feature_support(value: Any, n_time: int) -> np.ndarray:
    if isinstance(value, Mapping):
        support = value.get("support")
    else:
        support = getattr(value, "support", None)
    if support is None:
        return np.ones(n_time, dtype=np.bool_)
    return _coerce_bool_vector(
        support,
        length=n_time,
        label="feature support",
    )


def _timeline_arrays(
    timeline: Any,
    *,
    dataset: str,
    expected_t: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    modalities = _timeline_modalities(timeline)
    expected = set(DATASET_MODALITIES[dataset])
    supplied = {str(key).strip().lower() for key in modalities}
    if supplied != expected:
        raise PipelineError(
            "{} timeline modalities must be exactly {}; found {}".format(
                dataset, sorted(expected), sorted(supplied)
            )
        )
    arrays: Dict[str, np.ndarray] = {}
    support: Optional[np.ndarray] = None
    for modality in sorted(expected):
        value = next(
            (raw for key, raw in modalities.items() if str(key).strip().lower() == modality),
            None,
        )
        if value is None:
            raise PipelineError("timeline is missing modality {}".format(modality))
        arrays[modality] = _feature_array(value, modality)
        if arrays[modality].shape[0] != expected_t:
            raise PipelineError(
                "timeline {} has {} bins; expected {}".format(
                    modality, arrays[modality].shape[0], expected_t
                )
            )
        modality_support = _feature_support(value, expected_t)
        support = modality_support if support is None else np.logical_and(support, modality_support)
    if support is None:
        raise PipelineError("timeline has no modalities")
    return arrays, np.ascontiguousarray(support, dtype=np.bool_)


def _grid_image(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "affine"):
        return value
    try:
        import nibabel as nib

        return nib.load(str(value))
    except Exception:
        return None


def _grid_signature(value: Any) -> Optional[Tuple[Any, ...]]:
    image = _grid_image(value)
    if image is None:
        return None
    try:
        shape = tuple(int(size) for size in image.shape[:3])
        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            return None
        return shape + tuple(float(item) for item in affine.reshape(-1))
    except Exception:
        return None


class _AtlasGridCache:
    """Cache nearest-neighbour labels by sequential, geometry-checked grids."""

    def __init__(self, atlas_value: Any, *, dataset: str, output_root: Path, atlas_info: Mapping[str, Any]):
        self.atlas_value = atlas_value
        self.dataset = dataset
        self.output_root = output_root
        self.atlas_info = dict(_without_hash_fields(atlas_info))
        self.cache_dir = output_root / "preparation" / "atlas_grids"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        selected_name = "hcp_label_image"
        self.selected = getattr(atlas_value, selected_name, None)
        if self.selected is None and isinstance(atlas_value, Mapping):
            self.selected = atlas_value.get(selected_name)
        if self.selected is None:
            raise PipelineError("loaded atlas has no {}".format(selected_name))
        self.atlas_grid = _grid_signature(self.selected)
        self._resolved: Dict[Any, Any] = {}
        self.records: List[Dict[str, Any]] = []
        used = []
        for candidate in sorted(self.cache_dir.glob("grid-*.nii.gz")):
            try:
                used.append(int(candidate.name.removeprefix("grid-").removesuffix(".nii.gz")))
            except ValueError:
                continue
        self._next_index = max(used, default=-1) + 1

    @staticmethod
    def _grid_id(path: Path) -> str:
        return path.name.removesuffix(".nii.gz")

    def _validate_cached_target(
        self, target: Path, source_grid: Tuple[Any, ...], *, grid_id: str
    ) -> bool:
        if not target.exists():
            return False
        if target.is_symlink() or not target.is_file():
            raise PipelineError("atlas grid cache artifact is not a regular file: {}".format(target))
        if _grid_signature(target) != source_grid:
            return False
        try:
            _invoke(
                _atlas.validate_label_image,
                target,
                parcel_count=_bundle.N_PARCELS,
                name="cached Schaefer atlas",
            )
        except Exception as exc:
            raise PipelineError("atlas grid cache content is invalid for {}: {}".format(grid_id, exc)) from exc
        return True

    def _target_for_grid(self, source_grid: Tuple[Any, ...]) -> Tuple[Path, bool]:
        for candidate in sorted(self.cache_dir.glob("grid-*.nii.gz")):
            if self._validate_cached_target(
                candidate, source_grid, grid_id=self._grid_id(candidate)
            ):
                return candidate, True
        while True:
            target = self.cache_dir / "grid-{:03d}.nii.gz".format(self._next_index)
            self._next_index += 1
            if not target.exists():
                return target, False

    def resolve(self, source: Any) -> Any:
        fmri_path = getattr(source, "fmri_path", None)
        if fmri_path is None and isinstance(source, Mapping):
            fmri_path = source.get("fmri_path")
        source_grid = _grid_signature(fmri_path)
        if source_grid is None:
            # The adapter will provide the authoritative grid error.  We still
            # use a stable per-path key for synthetic image stand-ins.
            source_grid = ("source", str(fmri_path or ""))
        if source_grid in self._resolved:
            return self._resolved[source_grid]
        target, was_cached = self._target_for_grid(source_grid)
        if was_cached:
            result: Any = target
        else:
            result = _invoke(
                resample_labels_nearest,
                self.selected,
                fmri_path,
                output_path=target,
                overwrite=False,
            )
            if result is None:
                result = target
        grid_id = self._grid_id(target)
        if not self._validate_cached_target(target, source_grid, grid_id=grid_id):
            raise PipelineError("atlas grid cache resampling did not produce a valid {}".format(grid_id))
        stat = target.stat()
        descriptor = _grid_descriptor(target)
        if descriptor is None:
            raise PipelineError("atlas grid cache artifact has no valid geometry/dtype")
        self._resolved[source_grid] = result
        self.records.append(
            {
                "grid_id": grid_id,
                "grid": list(source_grid),
                "shape": list(descriptor["shape"]),
                "affine": descriptor["affine"],
                "dtype": descriptor["dtype"],
                "path": str(target.relative_to(self.output_root)).replace(os.sep, "/"),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "cached": was_cached,
                "content": "schaefer1000_integer_labels_0_to_1000",
            }
        )
        return result


def _prepared_masks(prepared: Any) -> Dict[str, np.ndarray]:
    masks = getattr(prepared, "masks", None)
    if not isinstance(masks, Mapping) or set(masks) != REQUIRED_MASKS:
        raise PipelineError(
            "prepared fMRI masks must be exactly {}".format(sorted(REQUIRED_MASKS))
        )
    try:
        brain_shape = np.asarray(getattr(prepared, "brain")).shape
    except (TypeError, ValueError) as exc:
        raise PipelineError("prepared fMRI brain shape is invalid") from exc
    expected_length = brain_shape[0] if brain_shape else None
    result: Dict[str, np.ndarray] = {}
    for name in REQUIRED_MASKS:
        if expected_length is None:
            try:
                value = np.asarray(masks[name], order="C")
            except (TypeError, ValueError) as exc:
                raise PipelineError("prepared mask {!r} is invalid".format(name)) from exc
            if value.ndim != 1 or value.dtype != np.bool_:
                raise PipelineError("prepared mask {!r} must be boolean".format(name))
            result[name] = np.ascontiguousarray(value, dtype=np.bool_)
        else:
            result[name] = _coerce_bool_vector(
                masks[name],
                length=expected_length,
                label="prepared mask {!r}".format(name),
            )
    return result


def _timeline_id_from_item(timeline: Any) -> str:
    value = _timeline_value(timeline, "timeline_id")
    if value is None:
        value = _timeline_value(timeline, "id")
    if value is None:
        raise PipelineError("raw timeline is missing timeline_id")
    return str(value).strip()


def _timeline_origin(timeline: Any) -> float:
    value = _timeline_value(timeline, "origin_seconds", _timeline_value(timeline, "time_origin_seconds", 0.0))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError("timeline origin_seconds is invalid") from exc
    if not np.isfinite(result):
        raise PipelineError("timeline origin_seconds must be finite")
    return result


def _timeline_provenance(timeline: Any) -> Dict[str, Any]:
    value = _timeline_value(timeline, "provenance", {})
    return dict(_without_hash_fields(value)) if isinstance(value, Mapping) else {}


def _run_atlas_image(cache: Optional[_AtlasGridCache], source: Any, dataset: str) -> Any:
    if dataset == "algonauts":
        return None
    if cache is None:
        raise PipelineError("volume source has no atlas cache")
    return cache.resolve(source)



def _resume_record(path: Path, label: str) -> Mapping[str, Any]:
    try:
        record = read_json_object(path)
    except Exception as exc:
        raise PipelineError("cannot read committed {} record {}: {}".format(label, path, exc)) from exc
    if not isinstance(record, Mapping):
        raise PipelineError("committed {} record must be an object: {}".format(label, path))
    return record


def _artifact_for_resume(
    paths: BundlePaths,
    record: Any,
    *,
    label: str,
    suffix: Optional[str] = None,
) -> None:
    if not isinstance(record, Mapping):
        raise PipelineError("{} artifact descriptor is invalid".format(label))
    missing = {"path", "size_bytes", "mtime_ns"} - set(record)
    if missing:
        raise PipelineError("{} artifact descriptor is missing {}".format(label, sorted(missing)))
    if any(_is_retired_hash_key(key) for key in record):
        raise PipelineError("{} artifact contains retired hash metadata".format(label))
    try:
        artifact_path = _bundle.contained_path(
            paths.root,
            record["path"],
            must_exist=True,
            kind="file",
            suffix=suffix,
        )
    except Exception as exc:
        raise PipelineError("{} artifact is invalid: {}".format(label, exc)) from exc
    try:
        stat = artifact_path.stat()
        if int(record["size_bytes"]) != int(stat.st_size):
            raise PipelineError("{} artifact size differs from its record".format(label))
        if int(record["mtime_ns"]) != int(stat.st_mtime_ns):
            raise PipelineError("{} artifact mtime differs from its record".format(label))
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineError("{} artifact metadata is invalid".format(label)) from exc


def _validate_resumed_feature_record(
    paths: BundlePaths,
    *,
    timeline_id: str,
    dataset: str,
    tr_seconds: float,
    n_timepoints: int,
    time_origin_seconds: float,
    raw_index_path: Optional[str],
    raw_index_stats: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    record = _resume_record(paths.feature_record(timeline_id), "feature")
    if record.get("timeline_id") != timeline_id:
        raise PipelineError("feature record timeline_id mismatch for {}".format(timeline_id))
    if record.get("feature_view") != _bundle.FEATURE_VIEW:
        raise PipelineError("feature record has the wrong feature view for {}".format(timeline_id))
    for field_name, expected in (
        ("n_timepoints", n_timepoints),
        ("tr_seconds", tr_seconds),
        ("time_origin_seconds", time_origin_seconds),
    ):
        actual = record.get(field_name)
        if field_name == "n_timepoints":
            if actual != expected:
                raise PipelineError("feature record time length mismatch for {}".format(timeline_id))
        else:
            try:
                matches = np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-8)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise PipelineError("feature record {} mismatch for {}".format(field_name, timeline_id))
    if record.get("normalization") != "none":
        raise PipelineError("feature record normalization is not none for {}".format(timeline_id))
    modalities = record.get("modalities")
    expected_modalities = set(DATASET_MODALITIES[dataset])
    if not isinstance(modalities, Mapping) or set(modalities) != expected_modalities:
        raise PipelineError("feature record modalities mismatch for {}".format(timeline_id))
    for modality in expected_modalities:
        descriptor = modalities.get(modality)
        if not isinstance(descriptor, Mapping):
            raise PipelineError("feature record {} artifact is invalid".format(modality))
        if descriptor.get("shape") != [n_timepoints, _bundle.N_LAYER_GROUPS, _bundle.MODALITY_DIMS[modality]]:
            raise PipelineError("feature record {} shape mismatch for {}".format(modality, timeline_id))
        if descriptor.get("dtype") != "float32":
            raise PipelineError("feature record {} dtype mismatch for {}".format(modality, timeline_id))
        _artifact_for_resume(
            paths,
            descriptor,
            label="{}/{}".format(timeline_id, modality),
            suffix=".npy",
        )
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise PipelineError("feature record source is invalid for {}".format(timeline_id))
    required_source = {"raw_cache_index_path", "raw_cache_index_stats"}
    missing_source = required_source - set(source)
    if missing_source:
        raise PipelineError(
            "feature record source is missing {} for {}".format(sorted(missing_source), timeline_id)
        )
    if source.get("raw_cache_index_path") != raw_index_path:
        raise PipelineError("feature record raw cache index path mismatch for {}".format(timeline_id))
    if source.get("raw_cache_index_stats") != raw_index_stats:
        raise PipelineError("feature record raw cache index stats mismatch for {}".format(timeline_id))
    pooling = record.get("pooling")
    if not isinstance(pooling, Mapping) or pooling.get("normalization") != "none":
        raise PipelineError("feature record pooling is not raw for {}".format(timeline_id))
    return record


def _validate_committed_run(
    paths: BundlePaths,
    *,
    source: Any,
    dataset: str,
    timeline_id: str,
    n_timepoints: int,
    time_origin_seconds: float,
) -> Mapping[str, Any]:
    run_id = _source_id(source)
    record = _resume_record(paths.run_record(run_id), "run")
    if record.get("run_id") != run_id or record.get("timeline_id") != timeline_id:
        raise PipelineError("committed run record identity mismatch for {}".format(run_id))
    if record.get("observation_space") != _bundle.OBSERVATION_SPACE:
        raise PipelineError("committed run observation space mismatch for {}".format(run_id))
    if record.get("n_timepoints") != n_timepoints:
        raise PipelineError("committed run time length mismatch for {}".format(run_id))
    try:
        clock_matches = (
            np.isclose(float(record.get("tr_seconds")), _tr(source), rtol=0.0, atol=1e-8)
            and np.isclose(
                float(record.get("time_origin_seconds")),
                float(time_origin_seconds),
                rtol=0.0,
                atol=1e-8,
            )
        )
    except (TypeError, ValueError):
        clock_matches = False
    if not clock_matches:
        raise PipelineError("committed run clock mismatch for {}".format(run_id))
    source_record = record.get("source")
    if not isinstance(source_record, Mapping) or source_record.get("dataset_id") != dataset:
        raise PipelineError("committed run dataset mismatch for {}".format(run_id))
    if source_record.get("run_id", run_id) != run_id:
        raise PipelineError("committed run source identity mismatch for {}".format(run_id))
    if source_record.get("timeline_id", timeline_id) != timeline_id:
        raise PipelineError("committed run source timeline mismatch for {}".format(run_id))
    current_source = _source_identity(source)
    if dict(source_record) != current_source:
        raise PipelineError("committed run source metadata mismatch for {}".format(run_id))
    _validate_preprocessing_source(record, source, run_id=run_id)
    brain_views = record.get("brain_views")
    if not isinstance(brain_views, Mapping) or set(brain_views) != {_bundle.BRAIN_VIEW}:
        raise PipelineError("committed run brain views mismatch for {}".format(run_id))
    brain = brain_views[_bundle.BRAIN_VIEW]
    if not isinstance(brain, Mapping) or brain.get("shape") != [n_timepoints, _bundle.N_PARCELS] or brain.get("dtype") != "float32":
        raise PipelineError("committed run brain artifact metadata mismatch for {}".format(run_id))
    _artifact_for_resume(paths, brain, label="{}/{}".format(run_id, _bundle.BRAIN_VIEW), suffix=".npy")
    masks = record.get("masks")
    if not isinstance(masks, Mapping) or set(masks) != REQUIRED_MASKS:
        raise PipelineError("committed run masks mismatch for {}".format(run_id))
    for name in REQUIRED_MASKS:
        mask = masks[name]
        if not isinstance(mask, Mapping) or mask.get("shape") != [n_timepoints] or mask.get("dtype") != "bool":
            raise PipelineError("committed run {} mask metadata mismatch for {}".format(name, run_id))
        _artifact_for_resume(paths, mask, label="{}/{}".format(run_id, name), suffix=".npy")
    _artifact_for_resume(paths, record.get("qc"), label="{}/qc".format(run_id))
    return record


def _check_grid_cache(
    atlas_provenance: Mapping[str, Any],
    *,
    output_root: Path,
    require_nonempty: bool = False,
) -> None:
    entries = atlas_provenance.get("grid_cache", [])
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise PipelineError("atlas grid cache is invalid")
    if require_nonempty and not entries:
        raise PipelineError(
            "full volume output requires a non-empty atlas grid cache"
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PipelineError("atlas grid cache record is invalid")
        grid_id = entry.get("grid_id")
        if (
            not isinstance(grid_id, str)
            or re.fullmatch(r"grid-[0-9]{3,}", grid_id) is None
            or grid_id in seen_ids
        ):
            raise PipelineError("atlas grid cache has duplicate/invalid grid_id")
        seen_ids.add(grid_id)
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise PipelineError("atlas grid cache path is invalid")
        seen_paths.add(relative)
        try:
            cache_path = _bundle.contained_path(
                output_root,
                relative,
                must_exist=True,
                kind="file",
            )
        except Exception as exc:
            raise PipelineError("atlas grid cache path is invalid") from exc
        expected_relative = "preparation/atlas_grids/{}.nii.gz".format(grid_id)
        if relative != expected_relative or cache_path.name != grid_id + ".nii.gz" or cache_path.is_symlink():
            raise PipelineError("atlas grid cache filename is invalid")
        try:
            stat = cache_path.stat()
            if int(entry.get("size_bytes")) != int(stat.st_size):
                raise PipelineError("atlas grid cache size mismatch")
            if int(entry.get("mtime_ns")) != int(stat.st_mtime_ns):
                raise PipelineError("atlas grid cache mtime mismatch")
        except (OSError, TypeError, ValueError) as exc:
            raise PipelineError("atlas grid cache stats are invalid") from exc
        expected_grid = entry.get("grid")
        if not isinstance(expected_grid, list):
            raise PipelineError("atlas grid cache geometry is missing")
        if _grid_signature(cache_path) != tuple(expected_grid):
            raise PipelineError("atlas grid cache geometry mismatch")
        descriptor = _grid_descriptor(cache_path)
        if descriptor is None:
            raise PipelineError("atlas grid cache descriptor is invalid")
        if entry.get("shape") != descriptor["shape"] or entry.get("dtype") != descriptor["dtype"]:
            raise PipelineError("atlas grid cache shape/dtype mismatch")
        try:
            if not np.allclose(
                np.asarray(entry.get("affine"), dtype=np.float64),
                np.asarray(descriptor["affine"], dtype=np.float64),
                rtol=0.0,
                atol=1e-8,
            ):
                raise PipelineError("atlas grid cache affine mismatch")
        except (TypeError, ValueError):
            raise PipelineError("atlas grid cache affine is invalid")
        try:
            _invoke(
                _atlas.validate_label_image,
                cache_path,
                parcel_count=_bundle.N_PARCELS,
                name="cached Schaefer atlas",
            )
        except Exception as exc:
            raise PipelineError("atlas grid cache content is invalid") from exc


def _check_completed_output(
    manifest: Mapping[str, Any],
    context: _Context,
    *,
    raw_index: Any,
) -> None:
    if manifest.get("dataset_id") != context.dataset:
        raise PipelineError("completed output dataset mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PipelineError("completed output is missing provenance")
    expected_selection = None if context.selection is None else list(context.selection)
    checks = (
        ("mode", context.mode),
        ("full_inventory", context.full_inventory),
        ("selection", expected_selection),
        ("expected_run_count", context.expected_run_count),
    )
    for key, expected in checks:
        if provenance.get(key) != expected:
            raise PipelineError("completed output request mismatch for {}".format(key))
    if (
        manifest.get("expected_run_count") != context.expected_run_count
        or manifest.get("run_count") != context.expected_run_count
    ):
        raise PipelineError("completed output run count mismatch")
    raw_path, raw_stats = _raw_index_identity(raw_index)
    raw_identity = _raw_index_content_identity(raw_index)
    for key, expected in (
        ("raw_cache_index_path", raw_path),
        ("raw_cache_index_stats", raw_stats),
        ("raw_cache_index_identity", raw_identity),
    ):
        if provenance.get(key) != expected:
            raise PipelineError("completed output request mismatch for {}".format(key))
    atlas_provenance = provenance.get("atlas")
    if not isinstance(atlas_provenance, Mapping):
        raise PipelineError("completed output atlas provenance is missing")
    for key in ("manifest_path", "manifest_stats", "source_path", "source_stats", "source_grid"):
        if atlas_provenance.get(key) != context.atlas_provenance.get(key):
            raise PipelineError("completed output atlas mismatch for {}".format(key))


def _prepare_source_run(
    dataset: str,
    source: Any,
    atlas_image: Any,
    support: np.ndarray,
    *,
    eager_volumes: bool,
) -> Any:
    """Prepare one run without writing shared bundle state."""

    prepare_kwargs: Dict[str, Any] = {"stimulus_support": support}
    if dataset == "algonauts":
        return _invoke(prepare_run, source, **prepare_kwargs)
    prepare_kwargs["allow_atlas_resampling"] = False
    prepare_kwargs["eager_volume"] = eager_volumes
    return _invoke(prepare_run, source, atlas_image, **prepare_kwargs)


def _prepare_source_worker(
    task: Tuple[str, Any, Any, np.ndarray, bool],
) -> Tuple[Any, Any]:
    """Spawn-safe worker entry point for one independent fMRI run."""

    dataset, source, atlas_image, support, eager_volumes = task
    run_id = _source_id(source)
    try:
        prepared = _prepare_source_run(
            dataset,
            source,
            atlas_image,
            support,
            eager_volumes=eager_volumes,
        )
    except Exception as exc:
        raise PipelineError(
            "parallel fMRI preparation failed for {}: {}".format(run_id, exc)
        ) from exc
    return source, prepared


def _commit_prepared_run(
    writer: Any,
    prepared: Any,
    source: Any,
    *,
    timeline_id: str,
    time_origin_seconds: float,
    atlas_provenance: Mapping[str, Any],
) -> None:
    """Validate and atomically commit one worker result in the parent."""

    masks = _prepared_masks(prepared)
    qc_summary = getattr(prepared, "qc_summary", None)
    if not callable(qc_summary):
        raise PipelineError("prepared fMRI run has no qc_summary()")
    preprocessing = dict(getattr(prepared, "provenance", {}) or {})
    preprocessing["pipeline"] = "neuroworld-schaefer1000-offline-v2"
    preprocessing["atlas"] = dict(atlas_provenance)
    _invoke(
        writer.write_run,
        run_id=_source_id(source),
        subject_id=_subject_id(source),
        timeline_id=timeline_id,
        tr_seconds=_tr(source),
        time_origin_seconds=time_origin_seconds,
        brain=np.asarray(getattr(prepared, "brain"), dtype=np.float32, order="C"),
        masks=masks,
        source=_source_identity(source),
        preprocessing=preprocessing,
        qc=qc_summary(),
    )


def run_preparation(
    dataset_id: str,
    *,
    source_root: Any,
    raw_index: Any,
    output: Any,
    atlas_manifest: Any = None,
    limit: Optional[int] = None,
    workers: int = 1,
    eager_volumes: bool = False,
) -> Dict[str, Any]:
    """Write or resume a bundle directly at ``output``.

    ``limit`` selects the first N canonical run IDs for a smoke run. Feature
    arrays are committed before their corresponding fMRI runs, and committed
    runs are checked and skipped when the same command is rerun.
    """

    output_root = _path(output, name="output")
    worker_count = _worker_count(workers)
    if not isinstance(eager_volumes, bool):
        raise PipelineError("eager_volumes must be boolean")
    context = _context(
        dataset_id,
        source_root,
        atlas_manifest,
        raw_index,
        limit=limit,
        audit_raw=False,
    )
    if context.blockers:
        raise PipelineError("preparation blockers: {}".format("; ".join(context.blockers)))
    if eager_volumes and context.dataset == "algonauts":
        raise PipelineError("eager_volumes is only valid for volume datasets")
    if raw_index is None:
        raise PipelineError("raw chunk index is required")

    # bundle.json is written last. Its presence means this exact output is
    # complete; an interrupted directory without it is resumed run by run.
    if (output_root / "bundle.json").is_file():
        manifest = read_json_object(output_root / "bundle.json")
        _check_completed_output(
            manifest,
            context,
            raw_index=raw_index,
        )
        return manifest

    raw_index_path, raw_index_stats = _raw_index_identity(raw_index)
    writer = _invoke(BundleWriter, output_root, context.dataset)
    committed = set(_invoke(committed_run_ids, output_root))
    paths = BundlePaths.from_root(output_root, create=True)
    atlas_cache = (
        None
        if context.dataset == "algonauts"
        else _AtlasGridCache(
            context.atlas_value,
            dataset=context.dataset,
            output_root=output_root,
            atlas_info=context.atlas_provenance,
        )
    )
    runs_by_timeline: Dict[str, List[Any]] = {}
    for source in context.runs:
        runs_by_timeline.setdefault(_timeline_id(source), []).append(source)
    specs = _timeline_specs(context.runs, context.dataset)
    expected_timelines = set(runs_by_timeline)
    seen_timelines: set = set()
    timeline_origins: Dict[str, float] = {}
    parallel_tasks: List[Tuple[str, Any, Any, np.ndarray, Optional[int], bool]] = []
    # A limited smoke run may select one timeline from a full raw index.
    stream_kwargs = {"allow_unrequested": context.selection is not None}
    stream = _invoke(iter_indexed_timelines, raw_index, specs, **stream_kwargs)
    for item in stream:
        for timeline in _flatten_timeline(item):
            timeline_id = _timeline_id_from_item(timeline)
            if timeline_id not in expected_timelines:
                if context.selection is not None:
                    continue
                raise PipelineError("raw cache yielded unselected timeline {}".format(timeline_id))
            source_group = runs_by_timeline[timeline_id]
            expected_t = _n_volumes(source_group[0])
            arrays, support = _timeline_arrays(
                timeline,
                dataset=context.dataset,
                expected_t=expected_t,
            )
            origin = _timeline_origin(timeline)
            timeline_origins[timeline_id] = origin
            feature_record = paths.feature_record(timeline_id)
            if feature_record.exists():
                _validate_resumed_feature_record(
                    paths,
                    timeline_id=timeline_id,
                    dataset=context.dataset,
                    tr_seconds=_tr(source_group[0]),
                    n_timepoints=expected_t,
                    time_origin_seconds=origin,
                    raw_index_path=raw_index_path,
                    raw_index_stats=raw_index_stats,
                )
            else:
                _invoke(
                    writer.write_features,
                    timeline_id=timeline_id,
                    tr_seconds=_tr(source_group[0]),
                    arrays=arrays,
                    source={
                        "timeline_id": timeline_id,
                        "raw_cache_index": str(raw_index),
                        "raw_cache_index_path": raw_index_path,
                        "raw_cache_index_stats": raw_index_stats,
                    },
                    pooling={
                        "recipe": "native_stimulus_v1",
                        "normalization": "none",
                        "hemodynamic_lag_seconds": None,
                        "timeline_provenance": _timeline_provenance(timeline),
                    },
                    time_origin_seconds=origin,
                )
            # Features are now committed.  Only then perform expensive atlas
            # resolution and fMRI preparation for uncommitted runs.
            for source in source_group:
                run_id = _source_id(source)
                if run_id in committed:
                    _validate_committed_run(
                        paths,
                        source=source,
                        dataset=context.dataset,
                        timeline_id=timeline_id,
                        n_timepoints=expected_t,
                        time_origin_seconds=origin,
                    )
                    continue
                atlas_image = _run_atlas_image(atlas_cache, source, context.dataset)
                if worker_count > 1:
                    parallel_tasks.append(
                        (
                            context.dataset,
                            source,
                            atlas_image,
                            np.ascontiguousarray(support, dtype=np.bool_),
                            eager_volumes,
                        )
                    )
                else:
                    prepared = _prepare_source_run(
                        context.dataset,
                        source,
                        atlas_image,
                        support,
                        eager_volumes=eager_volumes,
                    )
                    _commit_prepared_run(
                        writer,
                        prepared,
                        source,
                        timeline_id=timeline_id,
                        time_origin_seconds=origin,
                        atlas_provenance=context.atlas_provenance,
                    )
                    committed.add(run_id)
            seen_timelines.add(timeline_id)
    missing_timelines = sorted(expected_timelines - seen_timelines)
    if missing_timelines:
        raise PipelineError("raw cache did not yield timelines: {}".format(missing_timelines))
    if parallel_tasks:
        process_count = min(worker_count, len(parallel_tasks))
        print(
            "[parallel] starting {} workers for {} uncommitted {} runs".format(
                process_count, len(parallel_tasks), context.dataset
            ),
            file=sys.stderr,
            flush=True,
        )
        spawn_context = mp.get_context("spawn")
        # Eager HCP volumes are several GiB each. Recycling after every task
        # guarantees that allocator or library caches cannot accumulate over
        # a multi-day/full-inventory run.
        pool = spawn_context.Pool(processes=process_count, maxtasksperchild=1)
        completed_count = 0
        try:
            for source, prepared in pool.imap_unordered(
                _prepare_source_worker,
                parallel_tasks,
                chunksize=1,
            ):
                timeline_id = _timeline_id(source)
                _commit_prepared_run(
                    writer,
                    prepared,
                    source,
                    timeline_id=timeline_id,
                    time_origin_seconds=timeline_origins[timeline_id],
                    atlas_provenance=context.atlas_provenance,
                )
                run_id = _source_id(source)
                committed.add(run_id)
                completed_count += 1
                print(
                    "[parallel] committed {}/{} {}".format(
                        completed_count, len(parallel_tasks), run_id
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    raw_identity = _raw_index_content_identity(raw_index)
    provenance: Dict[str, Any] = {
        "pipeline": "neuroworld-schaefer1000-offline-v2",
        "dataset_id": context.dataset,
        "mode": context.mode,
        "full_inventory": context.full_inventory,
        "selection": None if context.selection is None else list(context.selection),
        "expected_run_count": context.expected_run_count,
        "source_audit": context.source_audit,
        "atlas": dict(context.atlas_provenance),
        "training_semantics": "offline_tmi_encoding",
        "raw_cache": dict(context.raw_audit),
        "raw_cache_index_path": raw_index_path,
        "raw_cache_index_stats": raw_index_stats,
        "raw_cache_index_identity": raw_identity,
        "hemodynamic_lag_seconds": None,
        "feature_normalization": "none",
    }
    if atlas_cache is not None:
        provenance["atlas"]["grid_cache"] = list(atlas_cache.records)
        _check_grid_cache(
            provenance["atlas"],
            output_root=output_root,
            require_nonempty=(
                context.full_inventory and context.dataset == "hcp_movies"
            ),
        )
    return _invoke(
        writer.finalize,
        expected_run_count=context.expected_run_count,
        provenance=provenance,
    )


__all__ = [
    "EXPECTED_RUN_COUNTS",
    "PipelineError",
    "audit_run_sources",
    "committed_run_ids",
    "discover_runs",
    "iter_indexed_timelines",
    "load_atlas",
    "prepare_run",
    "prepare",
    "read_chunk_index",
    "run_preparation",
    "sources",
    "atlas",
    "raw_cache",
    "adapters",
    "bundle",
]
