"""Minimal, checksum-free Schaefer-1000 atlas assets.

The active offline pipeline needs exactly two label images and one shared LUT.
This module copies those inputs into a self-contained directory, validates their
semantic contents, and records only structural metadata. It has no download,
pin, lock, grey-matter, or digest machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np


PARCEL_COUNT = 1000
OBSERVATION_SPACE = "schaefer1000"
ARTIFACT = "schaefer2018_1000parcels_7networks"
SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
MNI_LABEL_IMAGE_FILENAME = (
    "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_"
    "desc-1000Parcels7Networks_dseg.nii.gz"
)
HCP_LABEL_IMAGE_FILENAME = (
    "Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
)
COMMON_LUT_FILENAME = "Schaefer2018_1000Parcels_7Networks_order.tsv"

_ASSET_KEYS = frozenset({"mni_label_image", "hcp_label_image", "common_lut"})
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "observation_space",
        "parcel_count",
        "hashes_computed",
        "assets",
        "validation",
    }
)
_LABEL_KEYS = frozenset(
    {
        "path",
        "size_bytes",
        "mtime_ns",
        "kind",
        "role",
        "template",
        "shape",
        "affine",
        "dtype",
        "labels",
    }
)
_LUT_KEYS = frozenset(
    {
        "path",
        "size_bytes",
        "mtime_ns",
        "kind",
        "role",
        "row_count",
        "index_column",
        "name_column",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "regular_files",
        "integer_labels",
        "complete_labels",
        "lut_order_valid",
        "affine_grid_valid",
        "manifest_committed_last",
    }
)
_DIGEST_TOKENS = frozenset(
    {
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
)


class AtlasBuildError(ValueError):
    """Raised when an atlas input or committed atlas directory is invalid."""


@dataclass(frozen=True)
class LutRow:
    index: int
    name: str


@dataclass(frozen=True)
class AtlasImage:
    path: Path | None
    image: Any
    values: np.ndarray
    shape: tuple[int, int, int]
    affine: np.ndarray
    dtype: str


@dataclass(frozen=True)
class AtlasBundle:
    manifest_path: Path
    manifest: Mapping[str, Any]
    mni_label_image: Path
    hcp_label_image: Path
    common_lut: Path
    lut_names: tuple[str, ...]
    observation_space: str = OBSERVATION_SPACE

    @property
    def lut(self) -> Path:
        return self.common_lut


def _strict_json_load(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AtlasBuildError(f"Cannot read atlas manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AtlasBuildError("Atlas manifest must be a JSON object")
    return value


def _digest_key(key: object) -> bool:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower()
    tokens = [item for item in re.split(r"[^a-z0-9]+", text) if item]
    return any(item in _DIGEST_TOKENS for item in tokens)


def _assert_no_digest_fields(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key == "hashes_computed" and not path:
                pass
            elif _digest_key(key):
                location = ".".join((*path, key))
                raise AtlasBuildError(f"Atlas manifest contains retired digest field {location!r}")
            _assert_no_digest_fields(child, path=(*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_digest_fields(child, path=(*path, str(index)))


def _lexical_absolute(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _input_file(value: str | Path, *, label: str) -> Path:
    lexical = _lexical_absolute(value)
    try:
        path = lexical.resolve(strict=True)
    except OSError as exc:
        raise AtlasBuildError(f"{label} is not readable: {lexical}") from exc
    if not path.is_file():
        raise AtlasBuildError(f"{label} is not a regular file: {path}")
    return path


def _load_nifti(value: str | Path | Any, *, label: str) -> tuple[Any, Path | None]:
    if hasattr(value, "affine") and hasattr(value, "dataobj"):
        return value, None
    path = _input_file(value, label=label)
    try:
        import nibabel as nib

        return nib.load(str(path)), path
    except Exception as exc:
        raise AtlasBuildError(f"Cannot load {label} NIfTI {path}: {exc}") from exc


def validate_label_image(
    value: str | Path | Any,
    *,
    parcel_count: int = PARCEL_COUNT,
    name: str = "Schaefer label image",
) -> AtlasImage:
    """Validate a finite 3-D image whose nonzero labels are exactly ``1..N``."""

    if isinstance(parcel_count, bool) or not isinstance(parcel_count, int) or parcel_count <= 0:
        raise AtlasBuildError("parcel_count must be a positive integer")
    image, path = _load_nifti(value, label=name)
    shape = tuple(int(item) for item in image.shape)
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise AtlasBuildError(f"{name} must be a non-empty 3-D image, got {shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not math.isfinite(float(np.linalg.det(affine[:3, :3])))
        or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12
    ):
        raise AtlasBuildError(f"{name} has an invalid affine")
    try:
        raw = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise AtlasBuildError(f"Cannot read {name} voxel data: {exc}") from exc
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise AtlasBuildError(f"{name} must contain finite numeric labels")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise AtlasBuildError(f"{name} contains non-integer label values")
    labels = np.unique(rounded.astype(np.int64, copy=False))
    nonzero = labels[labels != 0]
    expected = np.arange(1, parcel_count + 1, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels > parcel_count) or not np.array_equal(nonzero, expected):
        raise AtlasBuildError(
            f"{name} nonzero labels must be exactly 1..{parcel_count}"
        )
    values = np.ascontiguousarray(rounded.astype(np.int16 if parcel_count <= 32767 else np.int32))
    return AtlasImage(
        path=path,
        image=image,
        values=values,
        shape=shape,
        affine=affine,
        dtype=np.dtype(raw.dtype).name,
    )


def _lut_fields(rows: list[list[str]]) -> tuple[int, int, list[list[str]]]:
    first = [item.strip() for item in rows[0]]
    lowered = [item.lower() for item in first]
    if any(item in {"index", "id", "label"} for item in lowered):
        index_col = next(
            (i for i, item in enumerate(lowered) if item in {"index", "id", "label"}),
            None,
        )
        name_col = next(
            (i for i, item in enumerate(lowered) if item in {"name", "label_name", "region"}),
            None,
        )
        if index_col is None or name_col is None:
            raise AtlasBuildError("LUT header must contain index and name columns")
        return index_col, name_col, rows[1:]
    return 0, 1, rows


def validate_lut(value: str | Path, *, parcel_count: int = PARCEL_COUNT) -> tuple[LutRow, ...]:
    path = _input_file(value, label="common LUT")
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeError) as exc:
        raise AtlasBuildError(f"Cannot read common LUT {path}: {exc}") from exc
    if not lines:
        raise AtlasBuildError("Common LUT is empty")
    delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
    parsed = (
        list(csv.reader(lines, delimiter=delimiter))
        if delimiter is not None
        else [line.split() for line in lines]
    )
    index_col, name_col, data = _lut_fields(parsed)
    output: list[LutRow] = []
    for ordinal, row in enumerate(data, start=1):
        if max(index_col, name_col) >= len(row):
            raise AtlasBuildError(f"LUT row {ordinal} is missing index or name")
        try:
            index = int(row[index_col].strip())
        except ValueError as exc:
            raise AtlasBuildError(f"LUT row {ordinal} has a non-integer index") from exc
        name = row[name_col].strip()
        if not name:
            raise AtlasBuildError(f"LUT row {ordinal} has an empty name")
        output.append(LutRow(index=index, name=name))
    expected = list(range(1, parcel_count + 1))
    if [row.index for row in output] != expected:
        raise AtlasBuildError(f"LUT indices must be ordered exactly 1..{parcel_count}")
    names = [row.name for row in output]
    if len(set(names)) != len(names):
        raise AtlasBuildError("LUT parcel names must be unique")
    return tuple(output)


def assert_shared_lut_order(
    first: str | Path,
    second: str | Path,
    *,
    parcel_count: int = PARCEL_COUNT,
) -> bool:
    left = validate_lut(first, parcel_count=parcel_count)
    right = validate_lut(second, parcel_count=parcel_count)
    if left != right:
        raise AtlasBuildError("LUTs do not share the same ordered parcel names")
    return True


def _file_stats(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _label_descriptor(
    path: Path,
    info: AtlasImage,
    *,
    role: str,
    template: str,
    parcel_count: int,
) -> dict[str, Any]:
    return {
        "path": path.name,
        **_file_stats(path),
        "kind": "label_image",
        "role": role,
        "template": template,
        "shape": list(info.shape),
        "affine": info.affine.tolist(),
        "dtype": info.dtype,
        "labels": {
            "background_optional": True,
            "nonzero_min": 1,
            "nonzero_max": parcel_count,
            "nonzero_count": parcel_count,
        },
    }


def _lut_descriptor(path: Path, *, parcel_count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        **_file_stats(path),
        "kind": "lookup_table",
        "role": "common_parcel_order",
        "row_count": parcel_count,
        "index_column": "index",
        "name_column": "name",
    }


def _safe_asset(root: Path, descriptor: Mapping[str, Any], *, expected_name: str) -> Path:
    relative = descriptor.get("path")
    if relative != expected_name:
        raise AtlasBuildError(f"Atlas asset path must be {expected_name!r}")
    lexical = root / expected_name
    if lexical.is_symlink() or not lexical.is_file():
        raise AtlasBuildError(f"Atlas asset is not a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AtlasBuildError(f"Atlas asset escapes its directory: {lexical}") from exc
    try:
        expected_size = int(descriptor.get("size_bytes"))
        expected_mtime = int(descriptor.get("mtime_ns"))
    except (TypeError, ValueError) as exc:
        raise AtlasBuildError(f"Atlas asset has invalid filesystem statistics: {lexical}") from exc
    stat = lexical.stat()
    if stat.st_size != expected_size:
        raise AtlasBuildError(f"Atlas asset size differs from manifest: {lexical}")
    if stat.st_mtime_ns != expected_mtime:
        raise AtlasBuildError(f"Atlas asset mtime differs from manifest: {lexical}")
    return lexical


def validate_atlas_manifest(
    value: str | Path,
    *,
    expected_parcel_count: int = PARCEL_COUNT,
) -> Mapping[str, Any]:
    lexical = _lexical_absolute(value)
    if lexical.is_symlink():
        raise AtlasBuildError(f"Atlas manifest must not be a symlink: {lexical}")
    manifest_path = _input_file(lexical, label="atlas manifest")
    manifest = _strict_json_load(manifest_path)
    _assert_no_digest_fields(manifest)
    if set(manifest) != _ROOT_KEYS:
        raise AtlasBuildError("Atlas manifest has missing or unexpected root fields")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AtlasBuildError("Atlas manifest schema_version is invalid")
    if manifest.get("artifact") != ARTIFACT or manifest.get("observation_space") != OBSERVATION_SPACE:
        raise AtlasBuildError("Atlas manifest identity is invalid")
    if manifest.get("hashes_computed") is not False:
        raise AtlasBuildError("Atlas manifest hashes_computed must be false")
    if manifest.get("parcel_count") != expected_parcel_count:
        raise AtlasBuildError("Atlas manifest parcel_count is invalid")
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != _ASSET_KEYS:
        raise AtlasBuildError("Atlas manifest assets must be exactly the two labels and common LUT")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, Mapping)
        or set(validation) != _VALIDATION_KEYS
        or any(validation.get(key) is not True for key in _VALIDATION_KEYS)
    ):
        raise AtlasBuildError("Atlas manifest validation flags are invalid")
    root = manifest_path.parent
    specifications = {
        "mni_label_image": (MNI_LABEL_IMAGE_FILENAME, "mni2009c_labels", "MNI152NLin2009cAsym"),
        "hcp_label_image": (HCP_LABEL_IMAGE_FILENAME, "hcp_fslmni_labels", "FSLMNI152_1mm"),
    }
    for key, (filename, role, template) in specifications.items():
        descriptor = assets.get(key)
        if not isinstance(descriptor, Mapping) or set(descriptor) != _LABEL_KEYS:
            raise AtlasBuildError(f"Atlas {key} descriptor is invalid")
        if descriptor.get("kind") != "label_image" or descriptor.get("role") != role or descriptor.get("template") != template:
            raise AtlasBuildError(f"Atlas {key} semantic metadata is invalid")
        path = _safe_asset(root, descriptor, expected_name=filename)
        info = validate_label_image(path, parcel_count=expected_parcel_count, name=key)
        if descriptor.get("shape") != list(info.shape) or descriptor.get("dtype") != info.dtype:
            raise AtlasBuildError(f"Atlas {key} shape/dtype differs from manifest")
        try:
            affine = np.asarray(descriptor.get("affine"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AtlasBuildError(f"Atlas {key} affine is invalid") from exc
        if affine.shape != (4, 4) or not np.allclose(affine, info.affine, rtol=0.0, atol=1e-8):
            raise AtlasBuildError(f"Atlas {key} affine differs from manifest")
        labels = descriptor.get("labels")
        if labels != {
            "background_optional": True,
            "nonzero_min": 1,
            "nonzero_max": expected_parcel_count,
            "nonzero_count": expected_parcel_count,
        }:
            raise AtlasBuildError(f"Atlas {key} label metadata is invalid")
    lut_descriptor = assets.get("common_lut")
    if not isinstance(lut_descriptor, Mapping) or set(lut_descriptor) != _LUT_KEYS:
        raise AtlasBuildError("Atlas common_lut descriptor is invalid")
    if (
        lut_descriptor.get("kind") != "lookup_table"
        or lut_descriptor.get("role") != "common_parcel_order"
        or lut_descriptor.get("row_count") != expected_parcel_count
        or lut_descriptor.get("index_column") != "index"
        or lut_descriptor.get("name_column") != "name"
    ):
        raise AtlasBuildError("Atlas common_lut semantic metadata is invalid")
    lut_path = _safe_asset(root, lut_descriptor, expected_name=COMMON_LUT_FILENAME)
    validate_lut(lut_path, parcel_count=expected_parcel_count)
    return manifest


def load_atlas(
    value: str | Path,
    *,
    expected_parcel_count: int = PARCEL_COUNT,
) -> AtlasBundle:
    manifest_path = _input_file(value, label="atlas manifest")
    manifest = validate_atlas_manifest(
        manifest_path,
        expected_parcel_count=expected_parcel_count,
    )
    root = manifest_path.parent
    rows = validate_lut(root / COMMON_LUT_FILENAME, parcel_count=expected_parcel_count)
    return AtlasBundle(
        manifest_path=manifest_path,
        manifest=manifest,
        mni_label_image=root / MNI_LABEL_IMAGE_FILENAME,
        hcp_label_image=root / HCP_LABEL_IMAGE_FILENAME,
        common_lut=root / COMMON_LUT_FILENAME,
        lut_names=tuple(row.name for row in rows),
    )


load_atlas_bundle = load_atlas
load_schaefer1000_atlas = load_atlas


def build_schaefer1000_atlas(
    *,
    mni_label_image: str | Path,
    hcp_label_image: str | Path,
    common_lut: str | Path,
    output_dir: str | Path,
    parcel_count: int = PARCEL_COUNT,
) -> Path:
    """Copy and atomically commit the minimal two-template atlas directory."""

    mni_source = _input_file(mni_label_image, label="MNI2009c label image")
    hcp_source = _input_file(hcp_label_image, label="HCP/FSL-MNI label image")
    lut_source = _input_file(common_lut, label="common LUT")
    validate_label_image(mni_source, parcel_count=parcel_count, name="MNI2009c label image")
    validate_label_image(hcp_source, parcel_count=parcel_count, name="HCP/FSL-MNI label image")
    validate_lut(lut_source, parcel_count=parcel_count)

    target = _lexical_absolute(output_dir)
    parent = target.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / target.name
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite atlas output: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=parent))
    try:
        destinations = {
            "mni_label_image": stage / MNI_LABEL_IMAGE_FILENAME,
            "hcp_label_image": stage / HCP_LABEL_IMAGE_FILENAME,
            "common_lut": stage / COMMON_LUT_FILENAME,
        }
        for source, destination in (
            (mni_source, destinations["mni_label_image"]),
            (hcp_source, destinations["hcp_label_image"]),
            (lut_source, destinations["common_lut"]),
        ):
            shutil.copy2(source, destination, follow_symlinks=True)
            if destination.is_symlink() or not destination.is_file():
                raise AtlasBuildError(f"Atlas copy is not a regular file: {destination}")
        mni_info = validate_label_image(destinations["mni_label_image"], parcel_count=parcel_count, name="copied MNI label image")
        hcp_info = validate_label_image(destinations["hcp_label_image"], parcel_count=parcel_count, name="copied HCP label image")
        validate_lut(destinations["common_lut"], parcel_count=parcel_count)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact": ARTIFACT,
            "observation_space": OBSERVATION_SPACE,
            "parcel_count": parcel_count,
            "hashes_computed": False,
            "assets": {
                "mni_label_image": _label_descriptor(
                    destinations["mni_label_image"], mni_info,
                    role="mni2009c_labels", template="MNI152NLin2009cAsym",
                    parcel_count=parcel_count,
                ),
                "hcp_label_image": _label_descriptor(
                    destinations["hcp_label_image"], hcp_info,
                    role="hcp_fslmni_labels", template="FSLMNI152_1mm",
                    parcel_count=parcel_count,
                ),
                "common_lut": _lut_descriptor(destinations["common_lut"], parcel_count=parcel_count),
            },
            "validation": {
                "regular_files": True,
                "integer_labels": True,
                "complete_labels": True,
                "lut_order_valid": True,
                "affine_grid_valid": True,
                "manifest_committed_last": True,
            },
        }
        manifest_path = stage / MANIFEST_FILENAME
        encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        with manifest_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        validate_atlas_manifest(manifest_path, expected_parcel_count=parcel_count)
        if os.path.lexists(target):
            raise FileExistsError(f"Refusing to overwrite atlas output: {target}")
        os.rename(stage, target)
        return (target / MANIFEST_FILENAME).resolve(strict=True)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def resample_labels_nearest(
    label_image: str | Path | Any,
    reference_image: str | Path | Any,
    *,
    output_path: str | Path | None = None,
    parcel_count: int = PARCEL_COUNT,
    overwrite: bool = False,
) -> Any:
    """Nearest-neighbour resample labels and require the complete parcel domain."""

    source = validate_label_image(label_image, parcel_count=parcel_count, name="source labels")
    reference, _ = _load_nifti(reference_image, label="reference image")
    reference_shape = tuple(int(item) for item in reference.shape[:3])
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    if len(reference_shape) != 3 or any(item <= 0 for item in reference_shape):
        raise AtlasBuildError("Reference image must expose a non-empty 3-D grid")
    if reference_affine.shape != (4, 4) or not np.isfinite(reference_affine).all():
        raise AtlasBuildError("Reference image has an invalid affine")
    try:
        from nibabel.processing import resample_from_to
        import nibabel as nib

        sampled = resample_from_to(
            source.image,
            (reference_shape, reference_affine),
            order=0,
            mode="constant",
            cval=0.0,
        )
    except Exception as exc:
        raise AtlasBuildError(f"Cannot nearest-neighbour resample atlas labels: {exc}") from exc
    raw = np.asanyarray(sampled.dataobj)
    if not np.isfinite(raw).all() or not np.array_equal(raw, np.rint(raw)):
        raise AtlasBuildError("Nearest-neighbour resampled labels are not finite integers")
    values = np.rint(raw).astype(np.int16 if parcel_count <= 32767 else np.int32)
    result = nib.Nifti1Image(values, reference_affine)
    result.set_qform(reference_affine, code=1)
    result.set_sform(reference_affine, code=1)
    validate_label_image(result, parcel_count=parcel_count, name="resampled labels")
    if output_path is None:
        return result
    lexical = _lexical_absolute(output_path)
    parent = lexical.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / lexical.name
    if os.path.lexists(target) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite resampled atlas: {target}")
    temporary = Path(tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".nii.gz", dir=parent)[1])
    try:
        nib.save(result, str(temporary))
        validate_label_image(temporary, parcel_count=parcel_count, name="resampled labels")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target.resolve(strict=True)


__all__ = [
    "ARTIFACT",
    "AtlasBuildError",
    "AtlasBundle",
    "AtlasImage",
    "COMMON_LUT_FILENAME",
    "HCP_LABEL_IMAGE_FILENAME",
    "LutRow",
    "MANIFEST_FILENAME",
    "MNI_LABEL_IMAGE_FILENAME",
    "OBSERVATION_SPACE",
    "PARCEL_COUNT",
    "assert_shared_lut_order",
    "build_schaefer1000_atlas",
    "load_atlas",
    "load_atlas_bundle",
    "load_schaefer1000_atlas",
    "resample_labels_nearest",
    "validate_atlas_manifest",
    "validate_label_image",
    "validate_lut",
]
