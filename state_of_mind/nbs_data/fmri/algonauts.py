"""Algonauts fMRI source and adapter entry points.

Algonauts' official source is already in Schaefer-1000 parcel order, so this
dataset adapter reads it directly and does not take an atlas image.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from state_of_mind.data.algonauts_contract import (
    algonauts_identity_from_hdf5_key,
    canonical_algonauts_raw_timeline,
    canonical_algonauts_run_id,
)
from state_of_mind.nbs_data.fmri.common import N_SCHAEFER_PARCELS

from ._adapters import (
    AdapterError,
    MissingSourceAssetError,
    PreparedFmriRun,
    SourceShapeError,
    _as_path,
    _base_provenance,
    _validate_prepare_source,
)
from ._sources import (
    NativeRunSource,
    SourceDiscoveryError,
    TR_SECONDS,
)


_ALGONAUTS_RE = re.compile(
    r"^sub-(?P<subject>\d+)_task-(?P<task>friends|movie10)_.*bold\.h5$"
)


def discover_algonauts(
    root: str | Path,
    *,
    run_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> tuple[NativeRunSource, ...]:
    """Discover official Algonauts Schaefer-1000 HDF5 runs."""

    try:
        import h5py
    except ImportError as exc:
        raise SourceDiscoveryError("Algonauts source discovery requires h5py") from exc
    source_root = Path(root).expanduser().resolve(strict=False)
    selected = None if run_ids is None else {str(value) for value in run_ids}
    rows: list[NativeRunSource] = []
    seen_run_ids: set[str] = set()
    for hdf5_path in sorted(source_root.glob("fmri/sub-*/func/*.h5")):
        match = _ALGONAUTS_RE.fullmatch(hdf5_path.name)
        if match is None:
            if strict:
                raise SourceDiscoveryError(f"Unrecognized Algonauts HDF5 name {hdf5_path.name}")
            continue
        subject = match.group("subject").zfill(2)
        task = match.group("task")
        parent_subject = hdf5_path.parent.parent.name
        parent_match = re.fullmatch(r"sub-(\d+)", parent_subject)
        if parent_match is None or int(parent_match.group(1)) != int(subject):
            message = (
                f"Algonauts HDF5 subject mismatch: filename {hdf5_path.name} "
                f"declares sub-{subject}, but parent directory is {parent_subject!r}"
            )
            if strict:
                raise SourceDiscoveryError(message)
            continue
        try:
            handle = h5py.File(hdf5_path, "r")
        except (OSError, ValueError) as exc:
            raise SourceDiscoveryError(f"Cannot open Algonauts HDF5 {hdf5_path}: {exc}") from exc
        with handle:
            for key in sorted(handle.keys()):
                value = handle[key]
                if not isinstance(value, h5py.Dataset):
                    if strict:
                        raise SourceDiscoveryError(f"HDF5 member is not a dataset: {hdf5_path}:{key}")
                    continue
                shape = tuple(int(item) for item in value.shape)
                if len(shape) != 2 or shape[0] <= 0 or shape[1] != 1000:
                    if strict:
                        raise SourceDiscoveryError(
                            f"Expected Algonauts [T,1000] at {hdf5_path}:{key}, found {shape}"
                        )
                    continue
                if np.dtype(value.dtype) != np.dtype(np.float32):
                    message = (
                        f"Algonauts dataset must have dtype float32 at "
                        f"{hdf5_path}:{key}, got {value.dtype}"
                    )
                    if strict:
                        raise SourceDiscoveryError(message)
                    continue
                identity = algonauts_identity_from_hdf5_key(task, key)
                if identity.is_tribev2_skip:
                    continue
                run_id = canonical_algonauts_run_id(subject, identity)
                if selected is not None and run_id not in selected:
                    continue
                if run_id in seen_run_ids:
                    raise SourceDiscoveryError(
                        f"Algonauts discovery contains duplicate canonical run ID {run_id!r}"
                    )
                seen_run_ids.add(run_id)
                rows.append(
                    NativeRunSource(
                        dataset_id="algonauts",
                        run_id=run_id,
                        timeline_id=canonical_algonauts_raw_timeline(subject, identity),
                        subject_id=f"Algonauts2025/{subject}",
                        tr_seconds=TR_SECONDS["algonauts"],
                        n_volumes=shape[0],
                        fmri_path=hdf5_path.resolve(),
                        source_root=source_root,
                        source_space="official_Schaefer18_1000Par7Net_HDF5",
                        hdf5_key=key,
                    )
                )
    if selected is not None:
        missing_ids = selected - {row.run_id for row in rows}
        if missing_ids:
            raise SourceDiscoveryError(
                f"Requested Algonauts run IDs were not found: {sorted(missing_ids)}"
            )
    if not rows:
        raise SourceDiscoveryError(f"No Algonauts HDF5 runs found under {source_root}")
    rows.sort(key=lambda row: row.run_id)
    return tuple(rows)


_ALGONAUTS_CANONICAL_RUN_RE = re.compile(
    r"^Algonauts2025/(?P<subject>\d+)/task-(?P<task>friends|movie10)/"
    r"[^/]+/run-(?P<run>\d+)$"
)


def _validate_algonauts_hdf5_identity(
    source: NativeRunSource,
    key: str,
) -> None:
    """Bind an official HDF5 member to the canonical source identity.

    Synthetic adapter fixtures intentionally use non-canonical run metadata;
    those are left available for numerical tests.  Discovered/official
    Algonauts sources always use the canonical ``Algonauts2025/...`` form and
    therefore receive the strict key/run/timeline/subject check.
    """

    match = _ALGONAUTS_CANONICAL_RUN_RE.fullmatch(str(source.run_id))
    canonical_prefix = str(source.run_id).startswith("Algonauts2025/")
    canonical_subject = str(source.subject_id).startswith("Algonauts2025/")
    if match is None:
        if canonical_prefix or canonical_subject:
            raise SourceShapeError(
                "Algonauts source must use canonical run and subject identity"
            )
        return
    subject = match.group("subject")
    task = match.group("task")
    try:
        identity = algonauts_identity_from_hdf5_key(task, key)
        expected_run_id = canonical_algonauts_run_id(subject, identity)
        expected_timeline = canonical_algonauts_raw_timeline(subject, identity)
    except ValueError as exc:
        raise SourceShapeError(
            f"Invalid Algonauts HDF5 identity {key!r}: {exc}"
        ) from exc
    expected_subject = f"Algonauts2025/{int(subject):02d}"
    if (
        source.run_id != expected_run_id
        or source.timeline_id != expected_timeline
        or source.subject_id != expected_subject
    ):
        raise SourceShapeError(
            "Algonauts HDF5 key identity disagrees with source run/timeline/subject: "
            f"key={key!r}, expected_run_id={expected_run_id!r}, "
            f"expected_timeline={expected_timeline!r}, expected_subject={expected_subject!r}"
        )


def _prepare_algonauts(
    source: NativeRunSource,
    *,
    stimulus_support: Sequence[bool] | np.ndarray | None,
) -> PreparedFmriRun:
    if source.dataset_id != "algonauts":
        raise AdapterError(f"Algonauts adapter received {source.dataset_id!r}")
    if source.hdf5_key is None:
        raise MissingSourceAssetError(f"Algonauts run {source.run_id} has no HDF5 key")
    path = _as_path(source.fmri_path)
    if not path.is_file():
        raise MissingSourceAssetError(f"Missing Algonauts HDF5 source: {path}")
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if source.hdf5_key not in handle:
                raise MissingSourceAssetError(f"Missing HDF5 key {source.hdf5_key!r} in {path}")
            _validate_algonauts_hdf5_identity(source, source.hdf5_key)
            dataset = handle[source.hdf5_key]
            shape = tuple(int(v) for v in dataset.shape)
            if shape != (int(source.n_volumes), N_SCHAEFER_PARCELS):
                raise SourceShapeError(
                    f"Algonauts {path}:{source.hdf5_key} must have shape "
                    f"[{source.n_volumes},1000], got {shape}"
                )
            if np.dtype(dataset.dtype) != np.dtype(np.float32):
                raise SourceShapeError(
                    f"Algonauts dataset must have dtype float32, got {dataset.dtype}"
                )
            brain = np.asarray(dataset[...], dtype=np.float32)
    except OSError as exc:
        raise AdapterError(f"Cannot read Algonauts HDF5 {path}: {exc}") from exc
    if not np.isfinite(brain).all():
        raise SourceShapeError(f"Algonauts dataset {path}:{source.hdf5_key} contains non-finite values")
    support = (
        np.ones(source.n_volumes, dtype=np.bool_)
        if stimulus_support is None
        else np.asarray(stimulus_support)
    )
    if support.shape != (source.n_volumes,) or support.dtype != np.bool_:
        raise SourceShapeError(f"stimulus_support must be boolean [{source.n_volumes}]")
    masks = {
        "valid_time": np.ascontiguousarray(support, dtype=np.bool_),
        "nonsteady": np.zeros(source.n_volumes, dtype=np.bool_),
        "motion_qc": np.ones(source.n_volumes, dtype=np.bool_),
        "stimulus_support": np.ascontiguousarray(support, dtype=np.bool_),
    }
    provenance = _base_provenance(source, adapter="algonauts_official_schaefer1000_identity")
    provenance["source"]["hdf5_key"] = source.hdf5_key
    provenance["processing"] = {
        "official_array_identity": True,
        "re_normalization": False,
        "smoothing": False,
        "confound_regression": False,
        "temporal_normalization": False,
    }
    provenance["n_timepoints"] = int(source.n_volumes)
    provenance["finite_time"] = {
        "n_finite": int(source.n_volumes),
        "n_nonfinite": 0,
        "finite_fraction": 1.0,
    }
    provenance["valid_time"] = {
        "formula": "fMRI_valid AND stimulus_support",
        "n_fmri_valid": int(source.n_volumes),
        "n_stimulus_supported": int(support.sum()),
        "n_valid": int(support.sum()),
        "n_invalid": int((~support).sum()),
    }
    zeros = np.zeros(source.n_volumes, dtype=np.float32)
    qc = {
        "framewise_displacement": zeros,
        "standardized_dvars": zeros.copy(),
        "fd": zeros,
        "std_dvars": zeros.copy(),
        "n_timepoints": int(source.n_volumes),
        "n_valid": int(support.sum()),
        "n_fmri_valid": int(source.n_volumes),
        "n_finite": int(source.n_volumes),
        "finite_fraction": 1.0,
        "valid_fraction": float(support.mean()),
    }
    return PreparedFmriRun(source, brain, masks, qc, provenance)


def prepare_algonauts(
    source: NativeRunSource,
    *,
    stimulus_support: Sequence[bool] | np.ndarray | None = None,
) -> PreparedFmriRun:
    """Prepare one official Algonauts Schaefer-1000 run."""

    _validate_prepare_source(source, dataset="algonauts")
    return _prepare_algonauts(source, stimulus_support=stimulus_support)


# Each dataset module exposes the same local spelling for simple dispatchers.
prepare_run = prepare_algonauts
discover = discover_algonauts


__all__ = [
    "AdapterError",
    "MissingSourceAssetError",
    "NativeRunSource",
    "PreparedFmriRun",
    "SourceDiscoveryError",
    "SourceShapeError",
    "TR_SECONDS",
    "discover",
    "discover_algonauts",
    "prepare_algonauts",
    "prepare_run",
]
