from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import state_of_mind.nbs_data.fmri.atlas as atlas


PARCELS = 4
SHAPE = (3, 3, 3)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def _save(path: Path, values: np.ndarray, affine: np.ndarray = AFFINE) -> Path:
    nib.save(nib.Nifti1Image(np.asarray(values), affine), str(path))
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels.flat[:PARCELS] = np.arange(1, PARCELS + 1)
    mni = _save(tmp_path / "mni.nii.gz", labels)
    hcp = _save(tmp_path / "hcp.nii.gz", labels.astype(np.float32))
    lut = tmp_path / "lut.tsv"
    lut.write_text(
        "index\tname\tcolor\n"
        + "".join(
            f"{index}\tParcel_{index}\t#000000\n"
            for index in range(1, PARCELS + 1)
        ),
        encoding="utf-8",
    )
    reference = _save(
        tmp_path / "reference.nii.gz", np.zeros(SHAPE, dtype=np.float32)
    )
    return {"mni": mni, "hcp": hcp, "lut": lut, "reference": reference}


def test_label_lut_and_nearest_resampling_contract(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assert (
        atlas.validate_label_image(paths["mni"], parcel_count=PARCELS).shape
        == SHAPE
    )
    assert [
        row.index
        for row in atlas.validate_lut(paths["lut"], parcel_count=PARCELS)
    ] == [1, 2, 3, 4]

    output = tmp_path / "resampled.nii.gz"
    assert (
        atlas.resample_labels_nearest(
            paths["mni"],
            paths["reference"],
            output_path=output,
            parcel_count=PARCELS,
        )
        == output.resolve()
    )
    observed = nib.load(str(output))
    assert observed.shape == SHAPE
    np.testing.assert_allclose(observed.affine, AFFINE)
    np.testing.assert_array_equal(
        np.unique(observed.get_fdata()), np.arange(PARCELS + 1)
    )


def test_build_load_and_validate_use_only_structural_metadata(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    manifest_path = atlas.build_schaefer1000_atlas(
        mni_label_image=paths["mni"],
        hcp_label_image=paths["hcp"],
        common_lut=paths["lut"],
        output_dir=tmp_path / "atlas",
        parcel_count=PARCELS,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["hashes_computed"] is False
    assert set(payload["assets"]) == {
        "mni_label_image",
        "hcp_label_image",
        "common_lut",
    }
    bundle = atlas.load_atlas(manifest_path, expected_parcel_count=PARCELS)
    assert bundle.lut_names == tuple(
        f"Parcel_{index}" for index in range(1, PARCELS + 1)
    )
    for path in (
        bundle.mni_label_image,
        bundle.hcp_label_image,
        bundle.common_lut,
    ):
        assert path.is_file() and not path.is_symlink()


def test_manifest_rejects_nested_camel_case_digest_fields(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = atlas.build_schaefer1000_atlas(
        mni_label_image=paths["mni"],
        hcp_label_image=paths["hcp"],
        common_lut=paths["lut"],
        output_dir=tmp_path / "atlas",
        parcel_count=PARCELS,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["validation"]["contentHash"] = "retired"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(atlas.AtlasBuildError, match="retired digest"):
        atlas.validate_atlas_manifest(
            manifest_path, expected_parcel_count=PARCELS
        )


def test_builder_refuses_existing_or_dangling_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    dangling = tmp_path / "atlas"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        atlas.build_schaefer1000_atlas(
            mni_label_image=paths["mni"],
            hcp_label_image=paths["hcp"],
            common_lut=paths["lut"],
            output_dir=dangling,
            parcel_count=PARCELS,
        )

