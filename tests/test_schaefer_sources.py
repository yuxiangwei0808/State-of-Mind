from __future__ import annotations

import numpy as np
import pytest

from neuroworld.nbs_data.fmri._sources import (
    SourceDiscoveryError,
    audit_run_sources,
)
from neuroworld.nbs_data.fmri.algonauts import discover_algonauts
from neuroworld.nbs_data.fmri.hcp_movies import discover_hcp_movies
from neuroworld.nbs_data.fmri import hcp_movies


def _nifti(path, shape, tr: float = 1.0) -> None:
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    if len(shape) == 4:
        image.header.set_zooms((1.0, 1.0, 1.0, tr))
        image.header.set_xyzt_units(xyz="mm", t="sec")
    image.set_qform(np.eye(4), code=1)
    image.set_sform(np.eye(4), code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def test_algonauts_discovery_is_identity_and_excludes_declared_skip(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = (
        tmp_path
        / "fmri/sub-01/func/"
        "sub-01_task-friends_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    )
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ses-001_task-s01e01a", data=np.zeros((3, 1000), dtype=np.float32))
        handle.create_dataset("ses-001_task-s04e01a", data=np.zeros((3, 1000), dtype=np.float32))

    runs = discover_algonauts(tmp_path)
    assert len(runs) == 1
    assert runs[0].hdf5_key == "ses-001_task-s01e01a"
    assert runs[0].source_space == "official_Schaefer18_1000Par7Net_HDF5"


def test_algonauts_discovery_rejects_filename_parent_subject_mismatch(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = (
        tmp_path
        / "fmri/sub-02/func/"
        "sub-01_task-friends_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    )
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ses-001_task-s01e01a", data=np.zeros((3, 1000), dtype=np.float32))

    with pytest.raises(SourceDiscoveryError, match="subject mismatch"):
        discover_algonauts(tmp_path)


def test_algonauts_discovery_rejects_duplicate_canonical_run_ids(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    for suffix in ("a", "b"):
        subject = "01"
        path = (
            tmp_path
            / f"fmri/sub-{subject}/func/"
            f"sub-{subject}_task-friends_space-MNI152NLin2009cAsym_{suffix}_"
            "atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ses-001_task-s01e01a", data=np.zeros((3, 1000), dtype=np.float32))

    with pytest.raises(SourceDiscoveryError, match="duplicate canonical run ID"):
        discover_algonauts(tmp_path)


def test_algonauts_discovery_requires_float32_hdf5_dtype(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = (
        tmp_path
        / "fmri/sub-01/func/"
        "sub-01_task-friends_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    )
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ses-001_task-s01e01a", data=np.zeros((3, 1000), dtype=np.float64))

    with pytest.raises(SourceDiscoveryError, match="dtype float32"):
        discover_algonauts(tmp_path)


def test_hcp_discovery_requires_official_mask_and_wmparc_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        hcp_movies,
        "_load_hcp_movie_versions_nohash",
        lambda root: ({"100610": "post"}, {"subjects": ["100610"], "subject_count": 1}),
    )
    run_dir = tmp_path / "100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP"
    nifti = run_dir / "tfMRI_MOVIE1_7T_AP.nii.gz"
    _nifti(nifti, (2, 2, 2, 921), tr=1.0)
    motion = run_dir / "Movement_Regressors.txt"
    np.savetxt(motion, np.zeros((921, 12)))
    runs = discover_hcp_movies(
        tmp_path,
        run_ids=["HCP7T/100610/task-movie/run-01/dir-AP"],
        strict=False,
    )
    assert len(runs) == 1
    assert runs[0].timeline_id == "hcp7t:post:7T_MOVIE1_CC1"
    assert runs[0].stimulus_version == "post"
    assert runs[0].stimulus_name == "7T_MOVIE1_CC1"
    assert runs[0].source_provenance is not None
    assert not hasattr(runs[0], "acquisition_sidecars")
    assert "acquisition_sidecars" not in runs[0].as_dict()
    audit = audit_run_sources(runs, require_full_inventory=False)
    assert not audit["ready"]
    assert runs[0].brain_mask is not None
    assert runs[0].tissue_labels is not None
    assert runs[0].brain_mask == (
        tmp_path
        / "100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/brainmask_fs.1.60.nii.gz"
    ).resolve()
    assert runs[0].tissue_labels == (
        tmp_path / "100610/MNINonLinear/ROIs/wmparc.1.60.nii.gz"
    ).resolve()
    assert set(runs[0].missing_paths) == {runs[0].brain_mask, runs[0].tissue_labels}


def test_hcp_discovery_has_no_acquisition_sidecars(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        hcp_movies,
        "_load_hcp_movie_versions_nohash",
        lambda root: ({"100610": "post"}, {"subjects": ["100610"], "subject_count": 1}),
    )
    run_id = "HCP7T/100610/task-movie/run-01/dir-AP"
    run_dir = tmp_path / "100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP"
    nifti = run_dir / "tfMRI_MOVIE1_7T_AP.nii.gz"
    _nifti(nifti, (2, 2, 2, 921), tr=1.0)
    motion = run_dir / "Movement_Regressors.txt"
    np.savetxt(motion, np.zeros((921, 12)))
    runs = discover_hcp_movies(tmp_path, run_ids=[run_id], strict=False)
    assert len(runs) == 1
    assert not hasattr(runs[0], "acquisition_sidecars")
    assert "acquisition_sidecars" not in runs[0].as_dict()
