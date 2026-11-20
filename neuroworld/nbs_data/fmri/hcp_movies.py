"""HCP movie source discovery and fMRI preparation.

This module owns the corrected-MNI source contract, the release-list join and
HCP's motion metrics. Runs are prepared with TRIBEv2's recipe (d'Ascoli et al.
2026): parcel means of HCP's minimally preprocessed volumes, z-scored and
detrended per run, with no smoothing or confound regression. Global-signal
and nuisance regression removed movie-driven signal (TRIBEv2's group
correlation on MOVIE2 fell from 0.38 to 0.29). Shared numerical kernels and
atlas IO remain in the common sibling modules.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neuroworld.nbs_data.fmri.common import (
    FmriError,
    power_framewise_displacement,
    process_volume_minimal,
    standardized_dvars,
)

from ._adapters import (
    AdapterError,
    MissingSourceAssetError,
    PreparedFmriRun,
    SourceShapeError,
    _as_path,
    _atlas_resampling_record,
    _base_provenance,
    _coerce_stimulus_support,
    _load_nifti,
    _materialize_nifti_float32,
    _nifti_values,
    _require_source_assets,
    _resolve_atlas,
    _result_from_fmri,
    _spatial_mask,
    _validate_prepare_source,
)
from ._sources import (
    EXPECTED_RUN_COUNTS,
    NativeRunSource,
    SourceDiscoveryError,
    TR_SECONDS,
    _file_stats,
    _nifti_frames,
    _without_hash_fields,
)


_HCP_RE = re.compile(r"^tfMRI_MOVIE(?P<run>[1-4])_7T_(?P<direction>AP|PA)\.nii\.gz$")
_SUBJECT_RE = re.compile(r"^[0-9]{6}$")


def _release_list_candidates(root: Path, filename: str) -> list[Path]:
    values = [root / filename, root / "movie_stimulus" / "movie_stimulus_2025" / filename]
    return [path.resolve(strict=False) for path in values if path.is_file()]


def _load_hcp_movie_versions_nohash(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Read the official movie-version join without any content digest."""

    try:
        from neuroworld.nbs_data.fmri.hcp_registry import HCP_EXPECTED_VERSION_SUBJECTS, HCP_VERSION_LISTS
    except Exception as exc:  # pragma: no cover - constants are part of this package
        raise SourceDiscoveryError(f"Cannot load HCP release-list contract: {exc}") from exc
    subject_versions: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}
    for version, filename in HCP_VERSION_LISTS.items():
        candidates = _release_list_candidates(root, filename)
        if not candidates:
            raise SourceDiscoveryError(f"HCP movie release list is absent: {filename}")
        # Duplicate official copies are accepted only when their structural
        # byte layout agrees.  The lists are tiny, so an exact byte comparison
        # is both deterministic and still free of any digest computation.
        try:
            first_bytes = candidates[0].read_bytes()
            if any(path.read_bytes() != first_bytes for path in candidates[1:]):
                raise SourceDiscoveryError(
                    f"Duplicate HCP release-list copies disagree for {filename}: {candidates}"
                )
        except OSError as exc:
            raise SourceDiscoveryError(f"Cannot read HCP release list {filename}: {exc}") from exc
        path = candidates[0]
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if set(reader.fieldnames or ()) != {"Subject", "M/F"}:
                    raise SourceDiscoveryError(
                        f"Unexpected HCP release-list columns in {path}: {reader.fieldnames}"
                    )
                version_subjects: list[str] = []
                for line_number, row in enumerate(reader, start=2):
                    subject = str(row.get("Subject", "")).strip()
                    sex = str(row.get("M/F", "")).strip()
                    if _SUBJECT_RE.fullmatch(subject) is None or sex not in {"M", "F"}:
                        raise SourceDiscoveryError(
                            f"Invalid HCP release-list row {path}:{line_number}: {row}"
                        )
                    if subject in subject_versions:
                        raise SourceDiscoveryError(
                            f"HCP subject {subject} appears in both/duplicate version lists"
                        )
                    subject_versions[subject] = version
                    version_subjects.append(subject)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise SourceDiscoveryError(f"Cannot read HCP release list {path}: {exc}") from exc
        expected_count = HCP_EXPECTED_VERSION_SUBJECTS.get(version)
        if expected_count is not None and len(version_subjects) != int(expected_count):
            raise SourceDiscoveryError(
                f"Expected {expected_count} HCP {version} subjects, found {len(version_subjects)}"
            )
        files[version] = {
            **_file_stats(path, source_root=root),
            "copies": [
                _file_stats(candidate, source_root=root)
                for candidate in candidates
            ],
            "subjects": sorted(version_subjects),
            "subject_count": len(version_subjects),
        }
    ordered = sorted(subject_versions)
    return subject_versions, {
        "files": files,
        "subjects": ordered,
        "subject_count": len(ordered),
        "version_counts": {
            version: len(entry["subjects"]) for version, entry in files.items()
        },
    }


def _validate_hcp_registered_mni_header(
    source_root: Path,
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Run the canonical header-only corrected registered-MNI validation."""

    try:
        from neuroworld.nbs_data.fmri.hcp_registry import _validate_registered_mni_nifti

        result = dict(
            _validate_registered_mni_nifti(
                source_root,
                path,
                expected_frames=expected_frames,
            )
        )
        stat = path.stat()
        result["path"] = str(path.resolve(strict=False))
        result["source_relative_path"] = path.resolve(strict=False).relative_to(
            source_root.resolve(strict=False)
        ).as_posix()
        result["size_bytes"] = int(stat.st_size)
        result["mtime_ns"] = int(stat.st_mtime_ns)
        return result
    except Exception as exc:
        raise SourceDiscoveryError(
            f"HCP registered-MNI header contract failed for {path}: {exc}"
        ) from exc


def discover_hcp_movies(
    root: str | Path,
    *,
    run_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> tuple[NativeRunSource, ...]:
    """Discover corrected HCP movie runs and their direct tissue assets.

    The active path parses the official pre/post release lists structurally and
    never calls the legacy registry digest helpers. Each run uses the adjacent
    ``Results/<run>/brainmask_fs.1.60.nii.gz`` mask; ``wmparc`` is the
    subject-level ROI under ``MNINonLinear/ROIs``. Historical acquisition
    sidecars are neither required nor copied.
    """

    source_root = Path(root).expanduser().resolve(strict=False)
    # Production discovery is structural and fails closed when the official
    # release lists are absent or malformed. Tests may patch this local parser
    # seam with a tiny authoritative fixture; the digesting legacy registry
    # loader is never consulted.
    subject_versions, version_provenance = _load_hcp_movie_versions_nohash(source_root)
    selected = None if run_ids is None else {str(value) for value in run_ids}
    rows: list[NativeRunSource] = []
    try:
        from neuroworld.nbs_data.fmri.hcp_registry import (
            HCP_EXPECTED_VERSION_SUBJECTS,
            HCP_MOVIE_RUNS,
            HCP_MOVIE_TIMELINES,
        )
    except Exception as exc:  # pragma: no cover
        raise SourceDiscoveryError(f"Cannot load HCP run contract: {exc}") from exc
    pattern = "*/MNINonLinear/Results/tfMRI_MOVIE*_7T_??/tfMRI_MOVIE*_7T_??.nii.gz"
    for nifti in sorted(source_root.glob(pattern)):
        match = _HCP_RE.fullmatch(nifti.name)
        if match is None or nifti.parent.name != nifti.name.removesuffix(".nii.gz"):
            continue
        subject = nifti.parts[-5]
        if _SUBJECT_RE.fullmatch(subject) is None:
            if strict:
                raise SourceDiscoveryError(f"Cannot parse HCP subject from {nifti}")
            continue
        run = int(match.group("run"))
        direction = match.group("direction")
        contract = HCP_MOVIE_RUNS[run]
        expected_direction = str(contract["direction"])
        if direction != expected_direction:
            raise SourceDiscoveryError(
                f"HCP movie run {run} has direction {direction}; expected {expected_direction}: {nifti}"
            )
        if subject not in subject_versions:
            raise SourceDiscoveryError(
                f"HCP movie subject {subject} is absent from the release lists"
            )
        stimulus_version = str(subject_versions[subject])
        if stimulus_version not in {"pre", "post"}:
            raise SourceDiscoveryError(
                f"HCP movie subject {subject} has invalid stimulus version {stimulus_version!r}"
            )
        stimulus_name = str(contract["movie"])
        stimulus_timeline = f"hcp7t:{stimulus_version}:{stimulus_name}"
        run_id = f"HCP7T/{subject}/task-movie/run-{run:02d}/dir-{direction}"
        if selected is not None and run_id not in selected:
            continue
        run_dir_name = nifti.parent.name
        relative_run_dir = Path(subject) / "MNINonLinear" / "Results" / run_dir_name
        # Official 1.6-mm mask is adjacent to this run's corrected volume.
        brain_mask = source_root / relative_run_dir / "brainmask_fs.1.60.nii.gz"
        motion = source_root / relative_run_dir / "Movement_Regressors.txt"
        tissue = source_root / subject / "MNINonLinear" / "ROIs" / "wmparc.1.60.nii.gz"
        required = [brain_mask, motion, tissue]
        if strict and any(not path.is_file() for path in required):
            missing = [path for path in required if not path.is_file()]
            raise SourceDiscoveryError(f"HCP run {run_id} is missing required assets: {missing}")
        if not motion.is_file() and strict:
            raise SourceDiscoveryError(f"HCP run {run_id} lacks Movement_Regressors.txt")
        n_volumes = _nifti_frames(nifti)
        expected_frames = int(contract["frames"])
        if n_volumes != expected_frames:
            raise SourceDiscoveryError(
                f"HCP run {run_id} has {n_volumes} frames; expected {expected_frames}"
            )
        registered_mni_validation = _validate_hcp_registered_mni_header(
            source_root, nifti, expected_frames=expected_frames
        )
        run_provenance: dict[str, Any] = {
            "version_registry": _without_hash_fields(version_provenance),
            "run_contract": {
                "run": run,
                "direction": expected_direction,
                "movie": stimulus_name,
                "frames": expected_frames,
                "tr_seconds": TR_SECONDS["hcp_movies"],
            },
            "registered_mni_validation": _without_hash_fields(registered_mni_validation),
            "asset_paths": {
                "brain_mask": str(brain_mask.resolve(strict=False)),
                "movement_regressors": str(motion.resolve(strict=False)),
                "wmparc": str(tissue.resolve(strict=False)),
            },
        }
        rows.append(
            NativeRunSource(
                dataset_id="hcp_movies",
                run_id=run_id,
                timeline_id=stimulus_timeline,
                subject_id=f"HCP7T/{subject}",
                tr_seconds=TR_SECONDS["hcp_movies"],
                n_volumes=n_volumes,
                fmri_path=nifti.resolve(),
                source_root=source_root,
                source_space="HCP_S1200_MNINonLinear_minimally_preprocessed_corrected",
                brain_mask=brain_mask.resolve(strict=False),
                motion_regressors=motion.resolve(strict=False),
                tissue_labels=tissue.resolve(strict=False),
                stimulus_version=stimulus_version,
                stimulus_name=stimulus_name,
                source_provenance=run_provenance,
            )
        )
    if selected is not None:
        missing_ids = selected - {row.run_id for row in rows}
        if missing_ids:
            raise SourceDiscoveryError(f"Requested HCP run IDs were not found: {sorted(missing_ids)}")
    if not rows:
        raise SourceDiscoveryError(f"No corrected HCP movie MNI runs found under {source_root}")
    if selected is None:
        observed_timelines = {row.timeline_id for row in rows}
        expected_timelines = set(HCP_MOVIE_TIMELINES)
        if observed_timelines != expected_timelines:
            raise SourceDiscoveryError(
                "HCP full discovery must expose exactly the canonical eight movie timelines; "
                f"missing={sorted(expected_timelines - observed_timelines)}, "
                f"extra={sorted(observed_timelines - expected_timelines)}"
            )
        if len(rows) != EXPECTED_RUN_COUNTS["hcp_movies"]:
            raise SourceDiscoveryError(
                f"HCP full discovery expected {EXPECTED_RUN_COUNTS['hcp_movies']} runs, found {len(rows)}"
            )
        expected_subject_count = sum(int(value) for value in HCP_EXPECTED_VERSION_SUBJECTS.values())
        if len({row.subject_id for row in rows}) != expected_subject_count:
            raise SourceDiscoveryError(
                f"HCP full discovery expected {expected_subject_count} subjects, "
                f"found {len({row.subject_id for row in rows})}"
            )
        observed_ids = [row.run_id for row in rows]
        if len(observed_ids) != len(set(observed_ids)):
            raise SourceDiscoveryError("HCP full discovery contains duplicate run IDs")
        observed_pairs = [
            (row.subject_id, row.stimulus_name, row.stimulus_version)
            for row in rows
        ]
        if len(observed_pairs) != len(set(observed_pairs)):
            raise SourceDiscoveryError("HCP full discovery contains duplicate subject/run identities")
        brain_masks = {str(row.brain_mask.resolve(strict=False)) for row in rows if row.brain_mask is not None}
        tissue_labels = {str(row.tissue_labels.resolve(strict=False)) for row in rows if row.tissue_labels is not None}
        if len(brain_masks) != EXPECTED_RUN_COUNTS["hcp_movies"]:
            raise SourceDiscoveryError(
                "HCP full discovery expected one run-local brainmask per run; "
                f"found {len(brain_masks)} for {len(rows)} runs"
            )
        if len(tissue_labels) != expected_subject_count:
            raise SourceDiscoveryError(
                "HCP full discovery expected one subject wmparc per subject; "
                f"found {len(tissue_labels)} for {expected_subject_count} subjects"
            )
        for row in rows:
            provenance = row.source_provenance
            if not isinstance(provenance, Mapping) or not isinstance(provenance.get("run_contract"), Mapping):
                raise SourceDiscoveryError(f"HCP run {row.run_id} lacks release run-contract provenance")
    return tuple(rows)


def _stream_hcp_dvars(image: Any, *, spatial_mask: np.ndarray, n_time: int, max_voxels: int = 16384) -> np.ndarray:
    """Standardized DVARS over the brain mask, computed in spatial chunks."""

    shape = tuple(int(v) for v in getattr(image, "shape", ()))
    if len(shape) != 4 or shape[-1] != int(n_time):
        raise SourceShapeError(f"HCP fMRI must be XYZT with {n_time} frames, got {shape}")
    dataobj = getattr(image, "dataobj", None)
    if dataobj is None:
        dataobj = _nifti_values(image, name="HCP fMRI")
    x, y, z, _ = shape
    if spatial_mask.shape != (x, y, z):
        raise SourceShapeError("HCP brain mask does not match the fMRI grid")
    z_step = max(1, min(z, int(max_voxels) // max(1, x * y)))
    dvars_sum_squares = np.zeros(n_time, dtype=np.float64)
    dvars_voxel_count = 0
    for z0 in range(0, z, z_step):
        z1 = min(z, z0 + z_step)
        raw = np.asarray(dataobj[..., z0:z1, :], dtype=np.float32)
        if raw.shape != (x, y, z1 - z0, n_time):
            raise SourceShapeError(f"HCP data proxy returned unexpected chunk shape {raw.shape}")
        matrix = np.moveaxis(raw, -1, 0).reshape(n_time, -1)
        finite_columns = np.isfinite(matrix).all(axis=0)
        scales = np.zeros(matrix.shape[1], dtype=np.float64)
        if np.any(finite_columns):
            scales[finite_columns] = np.std(np.asarray(matrix[:, finite_columns], dtype=np.float64), axis=0, ddof=0)
        usable = spatial_mask[..., z0:z1].reshape(-1) & finite_columns & (scales > np.finfo(np.float64).eps)
        if np.any(usable):
            chunk_dvars = standardized_dvars(matrix, valid_voxel_mask=usable)
            count = int(usable.sum())
            dvars_sum_squares += np.square(np.asarray(chunk_dvars, dtype=np.float64)) * count
            dvars_voxel_count += count
    if dvars_voxel_count <= 0:
        raise SourceShapeError("HCP fMRI has no finite non-flat voxels for standardized DVARS")
    dvars = np.sqrt(dvars_sum_squares / float(dvars_voxel_count)).astype(np.float32)
    if not np.isfinite(dvars).all():
        raise SourceShapeError("HCP standardized DVARS are non-finite")
    return dvars


def _read_hcp_motion(path: Path, *, n_time: int) -> np.ndarray:
    if not path.is_file():
        raise MissingSourceAssetError(f"Missing HCP Movement_Regressors.txt: {path}")
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise AdapterError(f"Cannot read HCP motion file {path}: {exc}") from exc
    if values.shape != (n_time, 12):
        raise SourceShapeError(f"HCP motion must have shape ({n_time},12), got {values.shape}")
    return np.ascontiguousarray(values, dtype=np.float64)


def _prepare_hcp(
    source: NativeRunSource,
    atlas_image: Any,
    *,
    stimulus_support: Sequence[bool] | np.ndarray | None,
    allow_atlas_resampling: bool,
    eager_volume: bool,
) -> PreparedFmriRun:
    if source.dataset_id not in {"hcp_movies", "hcp", "hcp_movie"}:
        raise AdapterError(f"HCP adapter received {source.dataset_id!r}")
    support = _coerce_stimulus_support(source, stimulus_support)
    _require_source_assets(source, ("brain_mask", "motion_regressors"))
    required = source.missing_paths
    if required:
        raise MissingSourceAssetError(f"HCP run {source.run_id} is missing {list(required)}")
    image, image_path = _load_nifti(source.fmri_path, name="HCP fMRI")
    mask_image, mask_path = _load_nifti(source.brain_mask, name="HCP official brain mask")
    atlas, atlas_path, atlas_resampled = _resolve_atlas(atlas_image, reference=image, allow_resample=allow_atlas_resampling)
    atlas_resampling = _atlas_resampling_record(
        atlas_image, image, atlas_path=atlas_path, performed=atlas_resampled
    )
    spatial_mask = _spatial_mask(mask_image, reference=image, name="HCP official brain mask")
    motion = _read_hcp_motion(_as_path(source.motion_regressors), n_time=source.n_volumes)
    fd = power_framewise_displacement(motion, rotation_in_degrees=True)
    # FD reads missing motion as zero; a row with missing motion is not valid.
    valid_time = np.isfinite(motion).all(axis=1)
    if support is not None:
        valid_time &= np.asarray(support, dtype=np.bool_)
    processing_image = (
        _materialize_nifti_float32(image, name="HCP fMRI")
        if eager_volume
        else image
    )
    dvars = _stream_hcp_dvars(processing_image, spatial_mask=spatial_mask, n_time=source.n_volumes)
    try:
        result = process_volume_minimal(
            processing_image,
            atlas,
            framewise_displacement=fd,
            standardized_dvars=dvars,
            valid_time=valid_time,
            spatial_mask=spatial_mask,
        )
    except FmriError as exc:
        raise AdapterError(f"HCP fMRI processing failed for {source.run_id}: {exc}") from exc
    extras = []
    for role, path in (("fmri", image_path), ("brain_mask", mask_path), ("schaefer_atlas", atlas_path)):
        if path is not None:
            extras.append((role, path))
    provenance = _base_provenance(
        source,
        adapter="hcp_s1200_corrected_mninonlinear",
        extra_files=extras,
    )
    provenance["processing"] = {
        **result.provenance,
        "recipe_source": "TRIBEv2, d'Ascoli et al. 2026 (arXiv 2605.04326), sections 5.7-5.8",
        "framewise_displacement": "Power_FD_degrees_to_radians_radius_50mm",
        "motion_source": "existing_12_column_Movement_Regressors",
        # HCP minimally preprocessed inputs do not expose a trusted
        # nonsteady-state flag stream. The all-false array is therefore a
        # storage mask, never evidence that every frame is steady state.
        "nonsteady": "flags_unavailable_all_false_storage_mask",
        "nonsteady_flags_available": False,
        "atlas_resampled": bool(atlas_resampled),
        "atlas_resampling": atlas_resampling,
        "volume_loading": "eager_float32" if eager_volume else "proxy_streaming",
    }
    return _result_from_fmri(source, result, provenance=provenance, stimulus_support=stimulus_support)


def prepare_hcp_movies(
    source: NativeRunSource,
    atlas_image: Any,
    *,
    stimulus_support: Sequence[bool] | np.ndarray | None = None,
    allow_atlas_resampling: bool = False,
    eager_volume: bool = False,
) -> PreparedFmriRun:
    """Prepare one corrected HCP movie native-TR run (TRIBEv2's minimal recipe)."""

    _validate_prepare_source(source, dataset="hcp_movies")
    if atlas_image is None:
        raise MissingSourceAssetError(f"A Schaefer atlas is required for HCP run {source.run_id}")
    if not isinstance(eager_volume, bool):
        raise AdapterError("eager_volume must be boolean")
    return _prepare_hcp(
        source,
        atlas_image,
        stimulus_support=stimulus_support,
        allow_atlas_resampling=allow_atlas_resampling,
        eager_volume=eager_volume,
    )


prepare_run = prepare_hcp_movies
discover = discover_hcp_movies


__all__ = [
    "AdapterError",
    "MissingSourceAssetError",
    "NativeRunSource",
    "PreparedFmriRun",
    "SourceDiscoveryError",
    "SourceShapeError",
    "TR_SECONDS",
    "discover",
    "discover_hcp_movies",
    "prepare_hcp_movies",
    "prepare_run",
]
