from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from state_of_mind.nbs_data import pipeline
from state_of_mind.nbs_data.bundle import BundleWriter
from state_of_mind.nbs_data.fmri._sources import NativeRunSource
from state_of_mind.nbs_data.stimuli.raw_cache import (
    RawCacheChunk,
    RawCacheIntegrityError,
    export_raw_chunks,
    read_chunk_index,
)


def _source(tmp_path: Path, dataset: str, run_id: str, timeline_id: str, n_time: int = 3) -> NativeRunSource:
    return NativeRunSource(
        dataset_id=dataset,
        run_id=run_id,
        timeline_id=timeline_id,
        subject_id="synthetic/01",
        tr_seconds=1.49 if dataset == "algonauts" else 1.0,
        n_volumes=n_time,
        fmri_path=tmp_path / (run_id.replace("/", "_") + ".nii.gz"),
        source_root=tmp_path,
        source_space="synthetic",
    )


def _timeline(timeline_id: str, n_time: int = 3, *, text: bool = False, support=None):
    values = {
        "video": SimpleNamespace(
            values=np.ones((n_time, 2, 1408), dtype=np.float32),
            support=np.ones(n_time, dtype=bool) if support is None else support[0],
        ),
        "audio": SimpleNamespace(
            values=np.ones((n_time, 2, 1024), dtype=np.float32),
            support=np.ones(n_time, dtype=bool) if support is None else support[1],
        ),
    }
    if text:
        values["text"] = SimpleNamespace(
            values=np.ones((n_time, 2, 3072), dtype=np.float32),
            support=np.ones(n_time, dtype=bool) if support is None else support[2],
        )
    return SimpleNamespace(
        timeline_id=timeline_id,
        origin_seconds=0.0,
        modalities=values,
        provenance={"offline_normalization": False, "hemodynamic_lag_seconds": None},
    )


def _write_raw_index(tmp_path: Path, timeline_id: str, *, name: str = "raw-cache") -> Path:
    chunks = [
        RawCacheChunk(
            timeline_id=timeline_id,
            modality=modality,
            values=np.zeros((3, 2, width), dtype=np.float32),
            timestamps=np.arange(3, dtype=np.float64) / 2.0 + 0.25,
            support=np.ones(3, dtype=np.bool_),
        )
        for modality, width in (("video", 1408), ("audio", 1024), ("text", 3072))
    ]
    return export_raw_chunks(chunks, tmp_path / name)


def _raw_index_stats(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_valid_label_grid(nib, reference: Path, output_path: Path) -> Path:
    reference_image = nib.load(str(reference))
    affine = np.asarray(reference_image.affine, dtype=np.float64)
    shape = tuple(int(size) for size in reference_image.shape[:3])
    labels = np.arange(1, 1001, dtype=np.int16).reshape(shape)
    image = nib.Nifti1Image(labels, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(output_path))
    return output_path


class _Prepared:
    def __init__(self, source: NativeRunSource):
        n_time = source.n_volumes
        self.brain = np.zeros((n_time, 1000), dtype=np.float32)
        self.masks = {
            "valid_time": np.ones(n_time, dtype=bool),
            "nonsteady": np.zeros(n_time, dtype=bool),
            "motion_qc": np.ones(n_time, dtype=bool),
            "stimulus_support": np.ones(n_time, dtype=bool),
        }
        # Synthetic runs retain the mandatory v2 source section.
        self.provenance = {"adapter": "synthetic", "source": {"files": []}}

    def qc_summary(self):
        return {"schema_version": 1, "synthetic": True}


def test_run_writes_features_first_and_intersects_all_support(monkeypatch, tmp_path: Path) -> None:
    run_id = "algonauts/01/run-01"
    source = _source(tmp_path, "algonauts", run_id, run_id)
    timeline = _timeline(
        run_id,
        support=(
            np.array([True, True, False]),
            np.array([True, False, True]),
            np.array([True, True, True]),
        ),
        text=True,
    )
    calls = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "discover_runs",
        lambda *_args, **_kwargs: calls.append("discover") or (source,),
    )
    monkeypatch.setattr(
        pipeline,
        "audit_run_sources",
        lambda *_args, **_kwargs: calls.append("audit") or {"blockers": []},
    )

    def fake_iter(*_args, **_kwargs):
        calls.append("timelines")
        yield timeline

    monkeypatch.setattr(pipeline, "iter_indexed_timelines", fake_iter)

    def fake_prepare(run, **kwargs):
        calls.append("prepare")
        captured["support"] = kwargs["stimulus_support"]
        return _Prepared(run)

    monkeypatch.setattr(pipeline, "prepare_run", fake_prepare)

    class RecordingWriter(BundleWriter):
        def write_features(self, **kwargs):
            calls.append("features")
            return super().write_features(**kwargs)

        def write_run(self, **kwargs):
            calls.append("run")
            return super().write_run(**kwargs)

        def finalize(self, **kwargs):
            calls.append("finalize")
            return super().finalize(**kwargs)

    monkeypatch.setattr(pipeline, "BundleWriter", RecordingWriter)
    manifest = pipeline.run_preparation(
        "algonauts",
        source_root=tmp_path,
        raw_index=tmp_path / "raw.jsonl",
        output=tmp_path / "output",
        limit=1,
    )

    assert manifest["run_count"] == 1
    assert calls == ["discover", "audit", "timelines", "features", "prepare", "run", "finalize"]
    np.testing.assert_array_equal(captured["support"], np.array([True, False, False]))


def test_prepare_full_inventory_gate_uses_exact_count(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path, "algonauts", "algonauts/01/run-01", "algonauts/01/run-01")
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    report = pipeline.prepare("algonauts", source_root=tmp_path, raw_index=[])
    assert report["mode"] == "full"
    assert report["expected_run_count"] == 1392
    assert any("expected 1392" in blocker for blocker in report["blockers"])
    assert report["ready"] is False


def test_limit_selects_first_canonical_run_ids(monkeypatch, tmp_path: Path) -> None:
    sources = (
        _source(tmp_path, "algonauts", "algonauts/01/run-03", "timeline-03"),
        _source(tmp_path, "algonauts", "algonauts/01/run-01", "timeline-01"),
        _source(tmp_path, "algonauts", "algonauts/01/run-02", "timeline-02"),
    )
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: sources)
    monkeypatch.setattr(
        pipeline,
        "audit_run_sources",
        lambda *_args, **_kwargs: {"blockers": []},
    )

    context = pipeline._context(
        "algonauts",
        tmp_path,
        None,
        [],
        limit=2,
        audit_raw=False,
    )

    assert context.mode == "limited"
    assert context.full_inventory is False
    assert context.selection == (
        "algonauts/01/run-01",
        "algonauts/01/run-02",
    )
    assert tuple(source.run_id for source in context.runs) == context.selection
    assert context.expected_run_count == 2


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_pipeline_rejects_invalid_limit(value, tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="limit must be a positive integer"):
        pipeline._context(
            "algonauts",
            tmp_path,
            None,
            [],
            limit=value,
            audit_raw=False,
        )


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_pipeline_rejects_invalid_worker_count(value, tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="workers must be a positive integer"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=tmp_path / "raw.jsonl",
            output=tmp_path / "output",
            limit=1,
            workers=value,
        )


def test_pipeline_rejects_eager_algonauts(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path, "algonauts", "algonauts/01/run-01", "timeline-01")
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(
        pipeline,
        "audit_run_sources",
        lambda *_args, **_kwargs: {"blockers": []},
    )
    with pytest.raises(pipeline.PipelineError, match="only valid for volume datasets"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=tmp_path / "raw.jsonl",
            output=tmp_path / "output",
            limit=1,
            eager_volumes=True,
        )


def test_pipeline_rejects_limit_larger_than_inventory(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path, "algonauts", "algonauts/01/run-01", "timeline-01")
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    with pytest.raises(pipeline.PipelineError, match="limit 2 exceeds the 1 discovered runs"):
        pipeline._context(
            "algonauts",
            tmp_path,
            None,
            [],
            limit=2,
            audit_raw=False,
        )


def test_hcp_missing_brainmask_or_wmparc_blocks_before_stream(monkeypatch, tmp_path: Path) -> None:
    run_id = "HCP7T/100610/task-movie/run-01/dir-AP"
    source = _source(tmp_path, "hcp_movies", run_id, run_id)
    atlas_value = SimpleNamespace(
        manifest_path=tmp_path / "atlas-manifest.json",
        manifest={},
        hcp_label_image=tmp_path / "hcp-labels.nii.gz",
    )
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "load_atlas", lambda *_args, **_kwargs: atlas_value)
    monkeypatch.setattr(
        pipeline,
        "iter_indexed_timelines",
        lambda *_args, **_kwargs: pytest.fail("HCP asset blocker must stop before streaming"),
    )
    with pytest.raises(pipeline.PipelineError, match="brainmask|wmparc"):
        pipeline.run_preparation(
            "hcp_movies",
            source_root=tmp_path,
            raw_index=tmp_path / "raw.jsonl",
            atlas_manifest=tmp_path / "atlas.json",
            output=tmp_path / "output",
            limit=1,
        )


def test_resume_derives_support_but_skips_committed_run(monkeypatch, tmp_path: Path) -> None:
    run_id = "algonauts/01/run-01"
    source = _source(tmp_path, "algonauts", run_id, run_id)
    stage = tmp_path / "stage"
    raw_index = _write_raw_index(tmp_path, run_id)
    raw_index_stats = _raw_index_stats(raw_index)
    writer = BundleWriter(stage, "algonauts")
    timeline = _timeline(run_id, text=True)
    arrays = {name: value.values for name, value in timeline.modalities.items()}
    writer.write_features(
        timeline_id=run_id,
        tr_seconds=1.49,
        arrays=arrays,
        source={
            "timeline_id": run_id,
            "raw_cache_index_path": str(raw_index.resolve()),
            "raw_cache_index_stats": raw_index_stats,
        },
        pooling={"normalization": "none", "hemodynamic_lag_seconds": None},
    )
    writer.write_run(
        run_id=run_id,
        subject_id=source.subject_id,
        timeline_id=run_id,
        tr_seconds=1.49,
        brain=np.zeros((3, 1000), dtype=np.float32),
        masks={
            "valid_time": np.ones(3, dtype=bool),
            "nonsteady": np.zeros(3, dtype=bool),
            "motion_qc": np.ones(3, dtype=bool),
            "stimulus_support": np.ones(3, dtype=bool),
        },
        source=pipeline._source_identity(source),
        preprocessing={"source": {"files": []}},
    )
    calls = []
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})

    def fake_iter(*_args, **_kwargs):
        calls.append("timelines")
        yield timeline

    monkeypatch.setattr(pipeline, "iter_indexed_timelines", fake_iter)
    monkeypatch.setattr(pipeline, "prepare_run", lambda *_args, **_kwargs: pytest.fail("committed run prepared"))
    original_fmri_path = source.fmri_path
    changed_fmri_path = tmp_path / "changed-source.h5"
    object.__setattr__(source, "fmri_path", changed_fmri_path)
    with pytest.raises(pipeline.PipelineError, match="source metadata"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=raw_index,
            output=stage,
            limit=1,
        )
    object.__setattr__(source, "fmri_path", original_fmri_path)
    other_index = _write_raw_index(tmp_path, run_id, name="other-raw-cache")
    with pytest.raises(pipeline.PipelineError, match="raw cache index"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=other_index,
            output=stage,
            limit=1,
        )
    calls.clear()
    manifest = pipeline.run_preparation(
        "algonauts",
        source_root=tmp_path,
        raw_index=raw_index,
        output=stage,
        limit=1,
    )
    assert calls == ["timelines"]
    assert manifest["run_count"] == 1





def test_raw_cache_structural_load_rejects_tampered_artifact(monkeypatch, tmp_path: Path) -> None:
    run_id = "algonauts/01/run-01"
    source = _source(tmp_path, "algonauts", run_id, run_id, n_time=2)
    raw_root = tmp_path / "raw-cache"
    raw_chunks = []
    for modality, width in (("video", 1408), ("audio", 1024), ("text", 3072)):
        raw_chunks.append(
            RawCacheChunk(
                timeline_id=run_id,
                modality=modality,
                values=np.zeros((2, 2, width), dtype=np.float32),
                timestamps=np.array([0.25, 0.75], dtype=np.float64),
                support=np.ones(2, dtype=bool),
            )
        )
    raw_index = export_raw_chunks(raw_chunks, raw_root)
    timeline = _timeline(run_id, n_time=2, text=True)
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "iter_indexed_timelines", lambda *_args, **_kwargs: iter([timeline]))
    monkeypatch.setattr(pipeline, "prepare_run", lambda run, **_kwargs: _Prepared(run))
    stage = tmp_path / "stage"
    pipeline.run_preparation(
        "algonauts",
        source_root=tmp_path,
        raw_index=raw_index,
        output=stage,
        limit=1,
    )
    record = read_chunk_index(raw_index)[0]
    values_path = raw_index.parent / record.values["path"]
    payload = bytearray(values_path.read_bytes())
    payload[-1] ^= 1
    values_path.write_bytes(payload)
    with pytest.raises(RawCacheIntegrityError, match="mtime"):
        record.load()

def test_resume_rejects_missing_feature_artifact(monkeypatch, tmp_path: Path) -> None:
    run_id = "algonauts/01/run-01"
    source = _source(tmp_path, "algonauts", run_id, run_id)
    raw_index = _write_raw_index(tmp_path, run_id)
    raw_index_stats = _raw_index_stats(raw_index)
    writer = BundleWriter(tmp_path / "stage", "algonauts")
    timeline = _timeline(run_id, text=True)
    writer.write_features(
        timeline_id=run_id,
        tr_seconds=1.49,
        arrays={name: value.values for name, value in timeline.modalities.items()},
        source={
            "timeline_id": run_id,
            "raw_cache_index_path": str(raw_index.resolve()),
            "raw_cache_index_stats": raw_index_stats,
        },
        pooling={"normalization": "none", "hemodynamic_lag_seconds": None},
    )
    writer.paths.feature_array(run_id, "audio").unlink()
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "iter_indexed_timelines", lambda *_args, **_kwargs: iter([timeline]))
    with pytest.raises(pipeline.PipelineError, match="artifact"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=raw_index,
            output=tmp_path / "stage",
            limit=1,
        )

def test_completed_output_rejects_raw_index_provenance_mismatch(monkeypatch, tmp_path: Path) -> None:
    run_id = "algonauts/01/run-01"
    source = _source(tmp_path, "algonauts", run_id, run_id)
    timeline = _timeline(run_id, text=True)
    raw_index = _write_raw_index(tmp_path, run_id)
    other_index = _write_raw_index(tmp_path, run_id, name="other-raw-cache")
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: (source,))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "iter_indexed_timelines", lambda *_args, **_kwargs: iter([timeline]))
    monkeypatch.setattr(pipeline, "prepare_run", lambda run, **_kwargs: _Prepared(run))
    pipeline.run_preparation(
        "algonauts",
        source_root=tmp_path,
        raw_index=raw_index,
        output=tmp_path / "stage",
        limit=1,
    )
    with pytest.raises(pipeline.PipelineError, match="raw_cache_index"):
        pipeline.run_preparation(
            "algonauts",
            source_root=tmp_path,
            raw_index=other_index,
            output=tmp_path / "stage",
            limit=1,
        )

def test_cross_grid_atlas_resampling_is_cached_per_grid(monkeypatch, tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    atlas_path = tmp_path / "atlas.nii.gz"
    labels = np.arange(1, 1001, dtype=np.int16).reshape(10, 10, 10)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), atlas_path)
    source_paths = []
    sources = []
    timelines = []
    for index in (1, 2):
        fmri_path = tmp_path / "fmri-{}.nii.gz".format(index)
        nib.save(
            nib.Nifti1Image(
                np.zeros((10, 10, 10, 3), dtype=np.float32),
                np.diag([2, 2, 2, 1]),
            ),
            fmri_path,
        )
        run_id = "HCP7T/10061{}/task-movie/run-01/dir-AP".format(index)
        source = _source(tmp_path, "hcp_movies", run_id, run_id)
        # Replace the synthetic path with the on-disk NIfTI grid.
        object.__setattr__(source, "fmri_path", fmri_path)
        for attribute in ("brain_mask", "tissue_labels"):
            path = tmp_path / (run_id.replace("/", "_") + "." + attribute)
            path.write_bytes(b"synthetic")
            object.__setattr__(source, attribute, path)
        sources.append(source)
        timelines.append(_timeline(run_id, n_time=3))
        source_paths.append(fmri_path)
    atlas_value = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json",
        manifest={},
        hcp_label_image=atlas_path,
    )
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: tuple(sources))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "load_atlas", lambda *_args, **_kwargs: atlas_value)
    monkeypatch.setattr(pipeline, "iter_indexed_timelines", lambda *_args, **_kwargs: iter(timelines))
    calls = []

    def fake_resample(source_label, reference, **kwargs):
        calls.append((source_label, reference))
        return _write_valid_label_grid(nib, reference, kwargs["output_path"])

    monkeypatch.setattr(pipeline, "resample_labels_nearest", fake_resample)
    monkeypatch.setattr(pipeline, "prepare_run", lambda source, *_args, **_kwargs: _Prepared(source))
    manifest = pipeline.run_preparation(
        "hcp_movies",
        source_root=tmp_path,
        raw_index=tmp_path / "raw.jsonl",
        atlas_manifest=tmp_path / "atlas.json",
        output=tmp_path / "output",
        limit=2,
    )
    assert manifest["run_count"] == 2
    assert len(calls) == 1
    assert (tmp_path / "output" / "preparation" / "atlas_grids").is_dir()

def test_hcp_shared_timeline_writes_features_once_and_fans_out_runs(monkeypatch, tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    timeline_id = "hcp7t:pre:7T_MOVIE1_CC1"
    sources = []
    for index in (1, 2):
        run_id = "HCP7T/10061{}/task-movie/run-01/dir-AP".format(index)
        source = _source(tmp_path, "hcp_movies", run_id, timeline_id, n_time=3)
        for attribute in ("fmri_path", "brain_mask", "tissue_labels", "motion_regressors"):
            suffix = ".nii.gz" if attribute == "fmri_path" else "." + attribute
            path = tmp_path / (run_id.replace("/", "_") + suffix)
            if attribute == "fmri_path":
                nib.save(
                    nib.Nifti1Image(
                        np.zeros((10, 10, 10, 3), dtype=np.float32),
                        np.diag([2, 2, 2, 1]),
                    ),
                    path,
                )
            else:
                path.write_bytes(b"synthetic")
            object.__setattr__(source, attribute, path)
        sources.append(source)
    timeline = _timeline(timeline_id, n_time=3)
    atlas_value = SimpleNamespace(hcp_label_image=tmp_path / "hcp-labels.nii.gz", manifest={})
    nib.save(
        nib.Nifti1Image(
            np.arange(1, 1001, dtype=np.int16).reshape(10, 10, 10),
            np.eye(4),
        ),
        atlas_value.hcp_label_image,
    )
    monkeypatch.setattr(pipeline, "discover_runs", lambda *_args, **_kwargs: tuple(sources))
    monkeypatch.setattr(pipeline, "audit_run_sources", lambda *_args, **_kwargs: {"blockers": []})
    monkeypatch.setattr(pipeline, "load_atlas", lambda *_args, **_kwargs: atlas_value)
    monkeypatch.setattr(pipeline, "iter_indexed_timelines", lambda *_args, **_kwargs: iter([(SimpleNamespace(timeline_id=timeline_id), timeline)]))
    monkeypatch.setattr(
        pipeline,
        "resample_labels_nearest",
        lambda _source, reference, **kwargs: _write_valid_label_grid(
            nib, reference, kwargs["output_path"]
        ),
    )
    calls = []
    monkeypatch.setattr(pipeline, "prepare_run", lambda source, *_args, **_kwargs: calls.append(source.run_id) or _Prepared(source))
    manifest = pipeline.run_preparation(
        "hcp_movies",
        source_root=tmp_path,
        raw_index=tmp_path / "raw.jsonl",
        atlas_manifest=tmp_path / "atlas.json",
        output=tmp_path / "output",
        limit=2,
    )
    assert manifest["run_count"] == 2
    assert manifest["feature_timeline_count"] == 1
    assert calls == [source.run_id for source in sources]


@pytest.mark.parametrize("value", (True, np.bool_(False), 3.5, np.float64(2.5)))
def test_pipeline_rejects_boolean_or_fractional_volume_counts(value) -> None:
    with pytest.raises(pipeline.PipelineError, match="volume count"):
        pipeline._n_volumes(SimpleNamespace(n_volumes=value))


def test_pipeline_rejects_nonboolean_feature_support() -> None:
    values = np.ones((2, 2, 1408), dtype=np.float32)
    malformed = SimpleNamespace(values=values, support=np.array([1, 0], dtype=np.uint8))
    with pytest.raises(pipeline.PipelineError, match="support.*boolean"):
        pipeline._feature_array(malformed, "video")
    with pytest.raises(pipeline.PipelineError, match="support.*boolean"):
        pipeline._feature_support(malformed, 2)


def test_pipeline_rejects_nonboolean_prepared_masks() -> None:
    masks = {
        "valid_time": np.ones(2, dtype=np.bool_),
        "nonsteady": np.zeros(2, dtype=np.bool_),
        "motion_qc": np.ones(2, dtype=np.uint8),
        "stimulus_support": np.ones(2, dtype=np.bool_),
    }
    prepared = SimpleNamespace(
        brain=np.zeros((2, 1000), dtype=np.float32),
        masks=masks,
    )
    with pytest.raises(pipeline.PipelineError, match="prepared mask.*boolean"):
        pipeline._prepared_masks(prepared)


def test_pipeline_raw_index_identity_fails_closed(monkeypatch, tmp_path: Path) -> None:
    index = tmp_path / "chunks.jsonl"
    index.write_text("not-json", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise ValueError("malformed index")

    monkeypatch.setattr(pipeline, "read_chunk_index", fail)
    with pytest.raises(pipeline.PipelineError, match="cannot read raw cache index"):
        pipeline._raw_index_content_identity(index)


def test_full_volume_grid_cache_requires_an_entry(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="non-empty atlas grid cache"):
        pipeline._check_grid_cache(
            {"grid_cache": []},
            output_root=tmp_path,
            require_nonempty=True,
        )
