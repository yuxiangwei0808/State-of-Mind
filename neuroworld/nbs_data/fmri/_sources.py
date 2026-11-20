"""Shared source records, structural helpers, dispatch, and inventory audit.

Dataset-specific discovery lives beside each dataset name in
``algonauts.py`` and ``hcp_movies.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


EXPECTED_RUN_COUNTS = {
    "algonauts": 1392,
    "hcp_movies": 725,
}
TR_SECONDS = {
    "algonauts": 1.49,
    "hcp_movies": 1.0,
}

_HCP_RUN_ID_RE = re.compile(
    r"^HCP7T/(?P<subject>[0-9]{6})/task-movie/run-0?(?P<run>[1-4])/dir-(?P<direction>AP|PA)$"
)


class SourceDiscoveryError(ValueError):
    """Raised when source membership or an input contract is ambiguous."""


@dataclass(frozen=True)
class NativeRunSource:
    """One run and every file required before signal preprocessing begins."""

    dataset_id: str
    run_id: str
    timeline_id: str
    subject_id: str
    tr_seconds: float
    n_volumes: int
    fmri_path: Path
    source_root: Path
    source_space: str
    hdf5_key: str | None = None
    sidecar_json: Path | None = None
    brain_mask: Path | None = None
    confounds_tsv: Path | None = None
    motion_regressors: Path | None = None
    tissue_labels: Path | None = None
    stimulus_version: str | None = None
    stimulus_name: str | None = None
    source_provenance: Mapping[str, Any] | None = None

    @property
    def required_paths(self) -> tuple[Path, ...]:
        values: list[Path] = [self.fmri_path]
        for value in (
            self.sidecar_json,
            self.brain_mask,
            self.confounds_tsv,
            self.motion_regressors,
            self.tissue_labels,
        ):
            if value is not None:
                values.append(value)
        return tuple(values)

    @property
    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(path for path in self.required_paths if not path.is_file())

    def source_relative(self, path: Path) -> str:
        resolved_root = self.source_root.resolve(strict=False)
        resolved = path.resolve(strict=False)
        try:
            return resolved.relative_to(resolved_root).as_posix()
        except ValueError:
            return str(resolved)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "timeline_id": self.timeline_id,
            "subject_id": self.subject_id,
            "tr_seconds": self.tr_seconds,
            "n_volumes": self.n_volumes,
            "fmri_path": self.source_relative(self.fmri_path),
            "source_space": self.source_space,
            "hdf5_key": self.hdf5_key,
            "sidecar_json": None if self.sidecar_json is None else self.source_relative(self.sidecar_json),
            "brain_mask": None if self.brain_mask is None else self.source_relative(self.brain_mask),
            "confounds_tsv": None if self.confounds_tsv is None else self.source_relative(self.confounds_tsv),
            "motion_regressors": None
            if self.motion_regressors is None
            else self.source_relative(self.motion_regressors),
            "tissue_labels": None
            if self.tissue_labels is None
            else self.source_relative(self.tissue_labels),
            "stimulus_version": self.stimulus_version,
            "stimulus_name": self.stimulus_name,
            "source_provenance": (
                None if self.source_provenance is None else dict(self.source_provenance)
            ),
        }
        return result


def _nifti_shape(path: Path) -> tuple[int, ...]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SourceDiscoveryError("NIfTI source discovery requires nibabel") from exc
    try:
        image = nib.load(str(path))
        return tuple(int(value) for value in image.shape)
    except (OSError, ValueError) as exc:
        raise SourceDiscoveryError(f"Cannot inspect NIfTI source {path}: {exc}") from exc


def _nifti_frames(path: Path) -> int:
    shape = _nifti_shape(path)
    if len(shape) != 4 or shape[3] <= 0:
        raise SourceDiscoveryError(f"Expected a non-empty 4-D NIfTI at {path}, found {shape}")
    return shape[3]


def _is_retired_hash_key(key: object) -> bool:
    """Recognize explicit legacy digest fields without matching ``shape``."""

    text = str(key)
    # Split common camelCase spellings before lower-casing, while retaining
    # compact spellings such as ``sha256sum`` and ``md5sum`` as whole tokens.
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized.lower()) if token]
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
    """Drop retired digest keys from metadata crossing the active boundary."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if _is_retired_hash_key(text):
                continue
            result[text] = _without_hash_fields(item)
        return result
    if isinstance(value, list):
        return [_without_hash_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_without_hash_fields(item) for item in value]
    return value


def _file_stats(path: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise SourceDiscoveryError(f"Cannot inspect source file {path}: {exc}") from exc
    record: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if source_root is not None:
        try:
            record["source_relative_path"] = resolved.relative_to(
                source_root.resolve(strict=False)
            ).as_posix()
        except ValueError:
            record["source_relative_path"] = str(resolved)
    return record


def discover_runs(
    dataset_id: str,
    root: str | Path,
    *,
    run_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> tuple[NativeRunSource, ...]:
    dataset = str(dataset_id).strip().lower().replace("-", "_")
    if dataset == "algonauts":
        from .algonauts import discover_algonauts

        return discover_algonauts(root, run_ids=run_ids, strict=strict)
    if dataset in {"hcp", "hcp_movie", "hcp_movies"}:
        from .hcp_movies import discover_hcp_movies

        return discover_hcp_movies(
            root,
            run_ids=run_ids,
            strict=strict,
        )
    raise SourceDiscoveryError(f"Unsupported offline dataset {dataset_id!r}")


def audit_run_sources(
    runs: Iterable[NativeRunSource],
    *,
    require_full_inventory: bool,
) -> dict[str, object]:
    values = tuple(runs)
    if not values:
        raise SourceDiscoveryError("Cannot audit an empty source inventory")
    datasets = {run.dataset_id for run in values}
    if len(datasets) != 1:
        raise SourceDiscoveryError(f"Source audit cannot mix datasets: {sorted(datasets)}")
    dataset_id = next(iter(datasets))
    duplicate_ids = sorted(
        run_id
        for run_id in {run.run_id for run in values}
        if sum(run.run_id == run_id for run in values) > 1
    )
    missing = {run.run_id: [str(path) for path in run.missing_paths] for run in values if run.missing_paths}
    expected = EXPECTED_RUN_COUNTS[dataset_id]
    blockers: list[str] = []
    if duplicate_ids:
        blockers.append(f"duplicate run IDs: {duplicate_ids[:10]}")
    if missing:
        blockers.append(f"{len(missing)} runs have missing required inputs")
    if require_full_inventory and len(values) != expected:
        blockers.append(f"expected {expected} runs, found {len(values)}")
    if dataset_id == "hcp_movies" and require_full_inventory:
        try:
            from neuroworld.nbs_data.fmri.hcp_registry import (
                HCP_EXPECTED_VERSION_SUBJECTS,
                HCP_MOVIE_RUNS,
                HCP_MOVIE_TIMELINES,
            )

            observed_timelines = {run.timeline_id for run in values}
            expected_subject_count = sum(int(value) for value in HCP_EXPECTED_VERSION_SUBJECTS.values())
            brain_masks = {str(run.brain_mask.resolve(strict=False)) for run in values if run.brain_mask is not None}
            tissue_labels = {str(run.tissue_labels.resolve(strict=False)) for run in values if run.tissue_labels is not None}
            if len(brain_masks) != len(values):
                blockers.append(
                    "HCP full inventory expected one run-local brainmask per run; "
                    f"found {len(brain_masks)} for {len(values)} runs"
                )
            if len(tissue_labels) != expected_subject_count:
                blockers.append(
                    "HCP full inventory expected one subject wmparc per subject; "
                    f"found {len(tissue_labels)} for {expected_subject_count} subjects"
                )
            expected_timelines = set(HCP_MOVIE_TIMELINES)
            if observed_timelines != expected_timelines:
                blockers.append(
                    "HCP full inventory must contain exactly the canonical eight timelines "
                    f"(missing={sorted(expected_timelines - observed_timelines)}, "
                    f"extra={sorted(observed_timelines - expected_timelines)})"
                )
            if len({run.subject_id for run in values}) != expected_subject_count:
                blockers.append(
                    f"HCP full inventory expected {expected_subject_count} subjects, "
                    f"found {len({run.subject_id for run in values})}"
                )
            observed_ids = [source.run_id for source in values]
            if len(observed_ids) != len(set(observed_ids)):
                blockers.append("HCP full inventory contains duplicate run IDs")
            observed_pairs = [
                (source.subject_id, source.stimulus_name, source.stimulus_version)
                for source in values
            ]
            if len(observed_pairs) != len(set(observed_pairs)):
                blockers.append("HCP full inventory contains duplicate subject/run identities")
            for source in values:
                match = _HCP_RUN_ID_RE.fullmatch(source.run_id)
                if match is None:
                    blockers.append(f"HCP run ID is not canonical: {source.run_id}")
                    continue
                run_number = int(match.group("run"))
                contract = HCP_MOVIE_RUNS[run_number]
                if source.n_volumes != int(contract["frames"]):
                    blockers.append(f"HCP run {source.run_id} has a noncanonical frame count")
                expected_timeline = f"hcp7t:{source.stimulus_version}:{contract['movie']}"
                if source.timeline_id != expected_timeline:
                    blockers.append(f"HCP run {source.run_id} has inconsistent timeline/version")
        except Exception as exc:
            blockers.append(f"HCP full inventory structural audit failed: {exc}")
    brain_mask_paths = sorted({str(run.brain_mask.resolve(strict=False)) for run in values if run.brain_mask is not None})
    tissue_label_paths = sorted({str(run.tissue_labels.resolve(strict=False)) for run in values if run.tissue_labels is not None})
    report: dict[str, object] = {
        "dataset_id": dataset_id,
        "run_count": len(values),
        "expected_run_count": expected,
        "full_inventory": len(values) == expected,
        "subjects": len({run.subject_id for run in values}),
        "subject_ids": sorted({run.subject_id for run in values}),
        "timelines": sorted({run.timeline_id for run in values}),
        "run_ids": sorted(run.run_id for run in values),
        "brain_mask_count": len(brain_mask_paths),
        "brain_mask_paths": brain_mask_paths,
        "tissue_label_count": len(tissue_label_paths),
        "tissue_label_paths": tissue_label_paths,
        "missing_inputs": missing,
        "blockers": blockers,
        "ready": not blockers,
    }
    return report


__all__ = [
    "EXPECTED_RUN_COUNTS",
    "TR_SECONDS",
    "NativeRunSource",
    "SourceDiscoveryError",
    "audit_run_sources",
    "discover_runs",
]
