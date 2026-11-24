"""Minimal HCP-YA 7T movie source contract.

Only the constants and header validation needed by the active Schaefer-1000
movie pipeline live here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HCP_STUDY = "HCP7T"
HCP_TR_SECONDS = 1.0
HCP_EXPECTED_RUNS = 725
HCP_EXPECTED_VERSION_SUBJECTS = {"pre": 147, "post": 37}
HCP_VERSION_LISTS = {
    "pre": "HCP_7T_movie_V1_ReleasedSubjs.csv",
    "post": "HCP_7T_movie_V2_ReleasedSubjs.csv",
}
HCP_MOVIE_RUNS: Mapping[int, Mapping[str, Any]] = {
    1: {"direction": "AP", "movie": "7T_MOVIE1_CC1", "frames": 921},
    2: {"direction": "PA", "movie": "7T_MOVIE2_HO1", "frames": 918},
    3: {"direction": "PA", "movie": "7T_MOVIE3_CC2", "frames": 915},
    4: {"direction": "AP", "movie": "7T_MOVIE4_HO2", "frames": 901},
}
HCP_MOVIE_TIMELINES = frozenset(
    f"hcp7t:{version}:{run['movie']}"
    for version in ("pre", "post")
    for run in HCP_MOVIE_RUNS.values()
)


class HcpRegistryError(ValueError):
    """Raised when an HCP movie source violates the active input contract."""


def _valid_spatial_affine(value: object) -> bool:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        return False
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        return False
    determinant = float(np.linalg.det(affine[:3, :3]))
    return math.isfinite(determinant) and abs(determinant) > 1e-8


def _validate_registered_mni_nifti(
    root: Path,
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Validate a corrected registered-MNI movie header without loading data."""

    source_root = Path(root).resolve(strict=False)
    source_path = Path(path).resolve(strict=False)
    try:
        relative = source_path.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise HcpRegistryError(
            f"registered MNI NIfTI is outside the source root: {source_path}"
        ) from exc
    try:
        size_bytes = int(source_path.stat().st_size)
    except OSError as exc:
        raise HcpRegistryError(
            f"registered MNI NIfTI cannot be stat'ed ({relative}): {exc}"
        ) from exc
    if size_bytes <= 0:
        raise HcpRegistryError(f"registered MNI NIfTI is empty: {relative}")

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - prepare checks dependencies
        raise HcpRegistryError(
            "nibabel is required to validate registered HCP MNI NIfTI headers"
        ) from exc
    try:
        image = nib.load(str(source_path), mmap=False)
    except Exception as exc:
        raise HcpRegistryError(
            f"registered MNI NIfTI is unreadable ({relative}): {exc}"
        ) from exc

    shape = tuple(int(item) for item in image.shape)
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise HcpRegistryError(
            f"registered MNI NIfTI must be nonempty 4D ({relative}); shape={shape}"
        )
    if shape[3] != int(expected_frames):
        raise HcpRegistryError(
            f"registered MNI NIfTI frame count mismatch ({relative}): "
            f"found {shape[3]}, expected {expected_frames}"
        )

    zooms = tuple(float(item) for item in image.header.get_zooms())
    tr_seconds = zooms[3]
    _, time_unit = image.header.get_xyzt_units()
    if time_unit != "sec":
        raise HcpRegistryError(
            f"registered MNI NIfTI time unit must be seconds ({relative}); "
            f"found {time_unit!r}"
        )
    if not math.isfinite(tr_seconds) or not math.isclose(
        tr_seconds,
        HCP_TR_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise HcpRegistryError(
            f"registered MNI NIfTI TR mismatch ({relative}): "
            f"found {tr_seconds}, expected {HCP_TR_SECONDS} seconds"
        )
    if not _valid_spatial_affine(image.affine):
        raise HcpRegistryError(
            f"registered MNI NIfTI has an invalid or singular affine: {relative}"
        )

    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    qform_code = int(qform_code)
    sform_code = int(sform_code)
    if qform_code <= 0 and sform_code <= 0:
        raise HcpRegistryError(
            f"registered MNI NIfTI has neither a coded qform nor sform: {relative}"
        )
    for label, form, code in (
        ("qform", qform, qform_code),
        ("sform", sform, sform_code),
    ):
        if code > 0 and not _valid_spatial_affine(form):
            raise HcpRegistryError(
                f"registered MNI NIfTI has an invalid coded {label}: {relative}"
            )

    return {
        "path": relative,
        "size_bytes": size_bytes,
        "shape": list(shape),
        "tr_seconds": tr_seconds,
        "time_unit": time_unit,
        "qform_code": qform_code,
        "sform_code": sform_code,
    }


__all__ = [
    "HCP_EXPECTED_RUNS",
    "HCP_EXPECTED_VERSION_SUBJECTS",
    "HCP_MOVIE_RUNS",
    "HCP_MOVIE_TIMELINES",
    "HCP_STUDY",
    "HCP_TR_SECONDS",
    "HCP_VERSION_LISTS",
    "HcpRegistryError",
]
