from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from state_of_mind.data import bundle as legacy_bundle
from state_of_mind.data.schaefer_v2_dataset import build_schaefer_v2_smoke_loader
from state_of_mind.nbs_data import pipeline
from state_of_mind.nbs_data.fmri import atlas
from state_of_mind.nbs_data.stimuli import raw_cache
from state_of_mind.nbs_data.stimuli import materialize as materializer
from state_of_mind.nbs_data.stimuli.raw_cache import RawCacheChunk
from state_of_mind.nbs_data.fmri._sources import NativeRunSource


def _chunks(timeline_id: str, n_time: int = 2):
    for modality, width in (("video", 1408), ("audio", 1024)):
        yield RawCacheChunk(
            timeline_id=timeline_id,
            modality=modality,
            values=np.arange(n_time * 2 * width, dtype=np.float32).reshape(n_time, 2, width),
            timestamps=np.arange(n_time, dtype=np.float64) / 2.0 + 0.25,
            support=np.array([True, False], dtype=np.bool_),
        )


class _Prepared:
    def __init__(self, source: NativeRunSource):
        n_time = source.n_volumes
        self.brain = np.zeros((n_time, 1000), dtype=np.float32)
        self.masks = {
            "valid_time": np.array([True, False], dtype=np.bool_),
            "nonsteady": np.zeros(n_time, dtype=np.bool_),
            "motion_qc": np.ones(n_time, dtype=np.bool_),
            "stimulus_support": np.array([True, False], dtype=np.bool_),
        }
        self.provenance = {"adapter": "synthetic", "source": {"files": []}}

    def qc_summary(self):
        return {"schema_version": 1, "synthetic": True}


def test_active_bundle_end_to_end_never_calls_digest_helpers(monkeypatch, tmp_path: Path):
    nib = pytest.importorskip("nibabel")

    def boom(*_args, **_kwargs):
        raise AssertionError("active Schaefer path computed a digest")

    monkeypatch.setattr(hashlib, "sha256", boom)
    monkeypatch.setattr(legacy_bundle, "sha256_file", boom, raising=False)
    monkeypatch.setattr(atlas, "sha256_file", boom, raising=False)

    timeline_id = "HCP/subject 01/timeline%1"
    monkeypatch.setattr(
        materializer,
        "discover_source_timeline_ids",
        lambda *_a, **_k: (timeline_id,),
    )

    def canonical_extractor(**kwargs):
        assert kwargs["dataset"] == "hcp"
        assert kwargs["selected_timeline_ids"] == [timeline_id]
        yield from _chunks(timeline_id)

    monkeypatch.setattr(
        materializer,
        "iter_tribev2_raw_chunks",
        canonical_extractor,
    )
    extractor_kwargs = {
        "events_path": [
            str(tmp_path / "events-1.tsv"),
            str(tmp_path / "events-2.tsv"),
            str(tmp_path / "events-3.tsv"),
            str(tmp_path / "events-4.tsv"),
        ],
        "cache_dir": str(tmp_path / "cache"),
        "config_path": str(tmp_path / "config.yaml"),
    }
    raw_index = materializer.materialize_native_features(
        dataset="hcp_movies",
        source_root=tmp_path,
        extractor=materializer.iter_tribev2_raw_chunks,
        extractor_kwargs=extractor_kwargs,
        generated_index_dir=tmp_path / "raw",
        export_raw_only=True,
    )

    fmri_path = tmp_path / "source.nii.gz"
    mni_label_path = tmp_path / "mni-labels.nii.gz"
    hcp_label_path = tmp_path / "hcp-labels.nii.gz"
    lut_path = tmp_path / "schaefer.lut.tsv"
    image = nib.Nifti1Image(np.zeros((1, 1, 1000, 2), dtype=np.float32), np.eye(4))
    image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
    image.header.set_xyzt_units(xyz="mm", t="sec")
    image.set_qform(np.eye(4), code=1)
    image.set_sform(np.eye(4), code=1)
    nib.save(image, str(fmri_path))

    labels = np.arange(1, 1001, dtype=np.int16).reshape(1, 1, 1000)
    for path in (mni_label_path, hcp_label_path):
        label_image = nib.Nifti1Image(labels, np.eye(4))
        label_image.set_qform(np.eye(4), code=1)
        label_image.set_sform(np.eye(4), code=1)
        nib.save(label_image, str(path))
    lut_path.write_text(
        "index\tname\n"
        + "".join(f"{index}\tParcel_{index}\n" for index in range(1, 1001)),
        encoding="utf-8",
    )
    atlas_manifest = atlas.build_schaefer1000_atlas(
        mni_label_image=mni_label_path,
        hcp_label_image=hcp_label_path,
        common_lut=lut_path,
        output_dir=tmp_path / "atlas",
    )
    atlas.validate_atlas_manifest(atlas_manifest, expected_parcel_count=1000)

    brain_mask_path = tmp_path / "brainmask.nii.gz"
    tissue_labels_path = tmp_path / "wmparc.nii.gz"
    for path in (brain_mask_path, tissue_labels_path):
        nib.save(nib.Nifti1Image(np.ones((1, 1, 1000), dtype=np.int16), np.eye(4)), str(path))
    source = NativeRunSource(
        dataset_id="hcp_movies",
        run_id="HCP/subject 01/run/1",
        timeline_id=timeline_id,
        subject_id="HCP/subject 01",
        tr_seconds=1.0,
        n_volumes=2,
        fmri_path=fmri_path,
        source_root=tmp_path,
        source_space="synthetic",
        brain_mask=brain_mask_path,
        tissue_labels=tissue_labels_path,
    )
    monkeypatch.setitem(pipeline.EXPECTED_RUN_COUNTS, "hcp_movies", 1)
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_a, **_k: (source,))
    monkeypatch.setattr(
        pipeline,
        "audit_run_sources",
        lambda *_a, **_k: {
            "dataset_id": "hcp_movies",
            "run_count": 1,
            "expected_run_count": 1,
            "full_inventory": True,
            "run_ids": [source.run_id],
            "timelines": [timeline_id],
            "blockers": [],
            "ready": True,
        },
    )
    monkeypatch.setattr(pipeline, "prepare_run", lambda run, *_a, **_k: _Prepared(run))

    output = tmp_path / "output-1"
    manifest = pipeline.run_preparation(
        "hcp_movies",
        source_root=tmp_path,
        raw_index=raw_index,
        atlas_manifest=atlas_manifest,
        output=output,
    )
    assert manifest["hashes_computed"] is False
    assert manifest["training_semantics"] == "offline_tmi_encoding"
    assert manifest["prospective_ready"] is False
    raw_identity = manifest["provenance"]["raw_cache_index_identity"]
    assert "records" not in raw_identity
    assert raw_identity["record_count"] == 2
    assert raw_identity["timeline_count"] == 1
    assert raw_identity["timelines"] == [timeline_id]
    assert raw_identity["modalities"] == {timeline_id: ["audio", "video"]}
    assert raw_identity["chunk_counts"] == {
        timeline_id: {"audio": 1, "video": 1}
    }
    retired = {"sha256", "checksum", "digest", "hash", "source_sha256", "content_hash"}

    def retired_keys(value):
        if isinstance(value, dict):
            found = []
            for key, item in value.items():
                lowered = key.lower()
                if (
                    key != "hashes_computed"
                    and (
                        lowered in retired
                        or lowered.endswith("_hash")
                        or lowered.endswith("_checksum")
                        or lowered.endswith("_digest")
                    )
                ):
                    found.append(key)
                found.extend(retired_keys(item))
            return found
        if isinstance(value, list):
            return [key for item in value for key in retired_keys(item)]
        return []

    assert not retired_keys(manifest)

    output2 = tmp_path / "output-2"
    pipeline.run_preparation(
        "hcp_movies",
        source_root=tmp_path,
        raw_index=raw_index,
        atlas_manifest=atlas_manifest,
        output=output2,
    )
    from state_of_mind.nbs_data.bundle import BundlePaths

    paths1 = BundlePaths.from_root(output)
    paths2 = BundlePaths.from_root(output2)
    comparable_paths = [
        (paths1.fmri_run(source.run_id), paths2.fmri_run(source.run_id)),
        *(
            (paths1.mask_run(name, source.run_id), paths2.mask_run(name, source.run_id))
            for name in sorted(_Prepared(source).masks)
        ),
        *(
            (paths1.feature_array(timeline_id, modality), paths2.feature_array(timeline_id, modality))
            for modality in ("video", "audio")
        ),
    ]
    for first, second in comparable_paths:
        assert np.array_equal(np.load(first), np.load(second))

    batch = next(iter(build_schaefer_v2_smoke_loader(output, run_id=source.run_id)))
    assert batch["y"].shape == (1, 1, 1000)


def test_raw_event_loader_never_calls_digest(monkeypatch, tmp_path: Path):
    def boom(*_args, **_kwargs):
        raise AssertionError("raw event loading computed a digest")

    monkeypatch.setattr(hashlib, "sha256", boom)
    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "timeline\tsubject\ttype\tstart\nmovie-1\tsub-1\tVideo\t0.0\n",
        encoding="utf-8",
    )

    events = raw_cache._load_tribev2_events(events_path)

    assert len(events) == 1


def test_hcp_discovery_uses_run_local_mask_and_subject_wmparc(monkeypatch, tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    from state_of_mind.nbs_data.fmri import hcp_movies

    def boom(*_args, **_kwargs):
        raise AssertionError("active HCP discovery computed a digest")

    # The reorganized HCP registry has no digest helper; patch the standard
    # constructor instead so this remains an end-to-end no-SHA assertion.
    monkeypatch.setattr(hashlib, "sha256", boom)
    monkeypatch.setattr(
        hcp_movies,
        "_load_hcp_movie_versions_nohash",
        lambda _root: ({"100610": "post"}, {"subjects": ["100610"], "subject_count": 1}),
    )
    run_id = "HCP7T/100610/task-movie/run-01/dir-AP"
    run_dir = tmp_path / "100610" / "MNINonLinear" / "Results" / "tfMRI_MOVIE1_7T_AP"
    run_path = run_dir / "tfMRI_MOVIE1_7T_AP.nii.gz"
    run_dir.mkdir(parents=True)
    image = nib.Nifti1Image(np.zeros((1, 1, 1, 921), dtype=np.float32), np.eye(4))
    image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
    image.header.set_xyzt_units(xyz="mm", t="sec")
    image.set_qform(np.eye(4), code=1)
    image.set_sform(np.eye(4), code=1)
    nib.save(image, str(run_path))
    mask_path = run_dir / "brainmask_fs.1.60.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4)), str(mask_path))
    tissue_path = tmp_path / "100610" / "MNINonLinear" / "ROIs" / "wmparc.1.60.nii.gz"
    tissue_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.int16), np.eye(4)), str(tissue_path))
    np.savetxt(run_dir / "Movement_Regressors.txt", np.zeros((921, 12), dtype=np.float32))

    rows = hcp_movies.discover_hcp_movies(tmp_path, run_ids=[run_id])
    assert len(rows) == 1
    row = rows[0]
    assert row.brain_mask == mask_path.resolve()
    assert row.tissue_labels == tissue_path.resolve()
    assert row.missing_paths == ()
    assert not hasattr(row, "acquisition_sidecars")
    assert "acquisition_sidecars" not in row.as_dict()
    assert "sha256" not in json.dumps(row.source_provenance or {}).lower()
