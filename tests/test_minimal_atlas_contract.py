from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroworld.nbs_data.fmri.atlas as atlas


PARCELS = atlas.PARCEL_COUNT
SHAPE = (10, 10, 11)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def _save_nifti(path: Path, values: np.ndarray, affine: np.ndarray = AFFINE) -> Path:
    nib.save(nib.Nifti1Image(np.asarray(values), affine), str(path))
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    values = np.zeros(int(np.prod(SHAPE)), dtype=np.int16)
    values[1 : PARCELS + 1] = np.arange(1, PARCELS + 1, dtype=np.int16)
    values = values.reshape(SHAPE)
    mni = _save_nifti(tmp_path / "mni.nii.gz", values)
    hcp = _save_nifti(tmp_path / "hcp.nii.gz", values)
    lut = tmp_path / "common.tsv"
    lut.write_text(
        "index\tname\n" + "".join(f"{index}\tParcel_{index}\n" for index in range(1, PARCELS + 1)),
        encoding="utf-8",
    )
    return {"mni": mni, "hcp": hcp, "lut": lut}


def _assert_no_checksums(value: object) -> None:
    if isinstance(value, dict):
        assert "sha256" not in value
        assert "checksum" not in value
        for item in value.values():
            _assert_no_checksums(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_checksums(item)


def test_minimal_builder_copies_dereferenced_files_without_hash_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    linked_mni = tmp_path / "mni-link.nii.gz"
    linked_mni.symlink_to(paths["mni"])
    monkeypatch.setattr(
        atlas,
        "sha256_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("hash called")),
        raising=False,
    )

    manifest_path = atlas.build_schaefer1000_atlas(
        mni_label_image=linked_mni,
        hcp_label_image=paths["hcp"],
        common_lut=paths["lut"],
        output_dir=tmp_path / "atlas",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_no_checksums(manifest)
    assert manifest["hashes_computed"] is False
    assert manifest["validation"]["affine_grid_valid"] is True
    output_mni = manifest_path.parent / atlas.MNI_LABEL_IMAGE_FILENAME
    output_hcp = manifest_path.parent / atlas.HCP_LABEL_IMAGE_FILENAME
    output_lut = manifest_path.parent / atlas.COMMON_LUT_FILENAME
    assert output_mni.is_file() and not output_mni.is_symlink()
    assert output_hcp.read_bytes() == paths["hcp"].read_bytes()
    assert output_lut.read_bytes() == paths["lut"].read_bytes()
    assert output_mni.read_bytes() == paths["mni"].read_bytes()
    loaded = atlas.load_atlas(manifest_path, expected_parcel_count=PARCELS)
    assert "hashes_computed" not in manifest["validation"]
    assert not hasattr(loaded, "mni_gm_probseg")


def test_labels_require_complete_nonzero_domain_but_background_is_optional(
    tmp_path: Path,
) -> None:
    values = np.arange(1, PARCELS + 1, dtype=np.int16).reshape((1, 20, 50))
    image = _save_nifti(tmp_path / "no-background.nii.gz", values)
    info = atlas.validate_label_image(image, parcel_count=PARCELS)
    assert np.array_equal(np.unique(info.values), np.arange(1, PARCELS + 1))

    values.flat[-1] = 0
    bad = _save_nifti(tmp_path / "missing-label.nii.gz", values)
    with pytest.raises(atlas.AtlasBuildError, match="nonzero labels.*1..1000"):
        atlas.validate_label_image(bad, parcel_count=PARCELS)


def test_minimal_manifest_validates_recorded_stats_and_grid(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = atlas.build_schaefer1000_atlas(
        mni_label_image=paths["mni"],
        hcp_label_image=paths["hcp"],
        common_lut=paths["lut"],
        output_dir=tmp_path / "atlas",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["assets"]["mni_label_image"]
    assert descriptor["size_bytes"] == (manifest_path.parent / descriptor["path"]).stat().st_size
    assert descriptor["mtime_ns"] == (manifest_path.parent / descriptor["path"]).stat().st_mtime_ns

    target = manifest_path.parent / descriptor["path"]
    target.touch()
    with pytest.raises(atlas.AtlasBuildError, match="mtime differs"):
        atlas.validate_atlas_manifest(manifest_path, expected_parcel_count=PARCELS)
