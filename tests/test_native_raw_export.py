"""Focused contracts for checksum-free raw stimulus export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import neuroworld.nbs_data.stimuli.raw_cache as raw_cache
import neuroworld.nbs_data.fmri._sources as sources
from neuroworld.nbs_data.stimuli import materialize as materializer


def _values(modality: str) -> np.ndarray:
    width = {"video": 1408, "audio": 1024}[modality]
    return np.ones((1, 2, width), dtype=np.float32)


def _parallel_fake_extractor(*, selected_timeline_ids, **_):
    for timeline_id in selected_timeline_ids:
        for modality in ("video", "audio"):
            yield {
                "timeline_id": timeline_id,
                "modality": modality,
                "values": _values(modality),
                "timestamps": [0.25],
                "chunk_index": 0,
            }


def _raw_kwargs(tmp_path: Path) -> dict[str, object]:
    events = []
    for index in range(4):
        path = tmp_path / f"events-{index}.tsv"
        path.write_text("timeline\ttype\tstart\tstop\n", encoding="utf-8")
        events.append(str(path))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    return {"events_path": events, "cache_dir": str(cache_dir), "config_path": str(config_path)}


def _assert_no_checksum_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            assert lowered not in {
                "sha", "sha1", "sha224", "sha256", "sha384", "sha512",
                "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum",
                "md5", "md5sum", "hash", "hashes", "checksum", "checksums",
                "digest", "digests",
            }
            _assert_no_checksum_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_checksum_fields(child)


def test_raw_only_cli_does_not_require_native_specs_or_output_dir() -> None:
    args = materializer.build_arg_parser().parse_args(
        [
            "--export-raw-only",
            "--dataset",
            "hcp_movies",
            "--source-root",
            "/tmp/source",
            "--generated-index-dir",
            "/tmp/raw",
            "--extractor",
            "neuroworld.nbs_data.stimuli.raw_cache:iter_tribev2_raw_chunks",
            "--extractor-kwargs",
            "{}",
        ]
    )
    assert args.export_raw_only is True
    assert not hasattr(args, "timeline_specs")
    assert not hasattr(args, "output_dir")
    assert args.dataset == "hcp_movies"
    assert args.workers == 1


def test_parallel_timeline_partitions_are_deterministic_and_disjoint() -> None:
    timeline_ids = tuple(f"timeline-{index}" for index in range(7))
    partitions = materializer._partition_timeline_ids(timeline_ids, 3)

    assert partitions == (
        ("timeline-0", "timeline-3", "timeline-6"),
        ("timeline-1", "timeline-4"),
        ("timeline-2", "timeline-5"),
    )
    assert sorted(value for partition in partitions for value in partition) == sorted(
        timeline_ids
    )
    with pytest.raises(raw_cache.RawCacheError, match="positive integer"):
        materializer._partition_timeline_ids(timeline_ids, 0)


def test_parallel_export_spawns_partitions_and_publishes_once(
    tmp_path: Path, monkeypatch
) -> None:
    timeline_ids = tuple(f"timeline-{index}" for index in range(5))
    monkeypatch.setattr(
        materializer,
        "_TRIBE_EXTRACTOR",
        f"{__name__}:_parallel_fake_extractor",
    )

    index = materializer._export_raw_chunks_parallel(
        extractor_kwargs={},
        timeline_ids=timeline_ids,
        generated_index_dir=tmp_path / "raw",
        workers=2,
        overwrite=False,
    )

    records = raw_cache.read_chunk_index(index)
    assert len(records) == len(timeline_ids) * 2
    assert {record.timeline_id for record in records} == set(timeline_ids)
    assert {record.modality for record in records} == {"video", "audio"}
    assert not (tmp_path / ".raw.staging").exists()


def test_raw_only_command_surface_has_no_hash_toggle_and_requires_extractor(
    tmp_path: Path,
) -> None:
    parser = materializer.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-verify-hashes"])
    with pytest.raises(raw_cache.RawCacheError, match="requires --extractor"):
        materializer.materialize_native_features(
            export_raw_only=True,
            dataset="hcp_movies",
            source_root=tmp_path,
            generated_index_dir=tmp_path / "raw",
        )


def test_raw_only_discovers_source_timelines_maps_hcp_and_skips_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = tmp_path / "raw"
    timeline_ids = tuple(f"timeline-{index}" for index in range(8))
    discovered = [SimpleNamespace(timeline_id=timeline_ids[index % 8]) for index in range(725)]
    observed: dict[str, object] = {}

    def fake_discover_runs(dataset, root, **kwargs):
        observed["dataset"] = dataset
        observed["root"] = root
        observed["strict"] = kwargs.get("strict")
        return discovered

    def fake_audit(runs, *, require_full_inventory):
        observed["audit_required"] = require_full_inventory
        return {
            "dataset_id": "hcp_movies",
            "ready": True,
            "full_inventory": True,
            "run_count": len(runs),
            "timelines": list(timeline_ids),
            "blockers": [],
        }

    monkeypatch.setattr(sources, "discover_runs", fake_discover_runs)
    monkeypatch.setattr(sources, "audit_run_sources", fake_audit)

    def fake_tribe(**kwargs):
        observed["kwargs"] = kwargs
        assert kwargs["dataset"] == "hcp"
        assert kwargs["selected_timeline_ids"] == list(timeline_ids)
        for timeline_index, timeline_id in enumerate(timeline_ids):
            for modality_index, modality in enumerate(("video", "audio")):
                yield {
                    "timeline_id": timeline_id,
                    "modality": modality,
                    "values": _values(modality),
                    "timestamps": [0.25],
                    "metadata": {"config_sha256": "must-be-omitted", "kept": True},
                    "chunk_index": timeline_index * 2 + modality_index,
                }

    monkeypatch.setattr(materializer, "iter_tribev2_raw_chunks", fake_tribe)

    index = materializer.materialize_native_features(
        export_raw_only=True,
        dataset="hcp_movies",
        source_root=source_root,
        extractor=materializer.iter_tribev2_raw_chunks,
        extractor_kwargs=_raw_kwargs(tmp_path),
        generated_index_dir=output_root,
    )

    assert index == output_root / "chunks.jsonl"
    assert observed["dataset"] == "hcp_movies"
    assert observed["strict"] is True
    assert observed["audit_required"] is True
    assert Path(observed["root"]).resolve() == source_root.resolve()
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 16
    _assert_no_checksum_fields(rows)
    assert {row["modality"] for row in rows} == {"video", "audio"}
    # Structural descriptors remain available for shape/dtype validation.
    assert rows[0]["values"]["shape"][1:] == [2, {"audio": 1024, "video": 1408}[rows[0]["modality"]]]
    for row in rows:
        for name in ("values", "timestamps", "support"):
            descriptor = row[name]
            artifact = index.parent / descriptor["path"]
            assert descriptor["size_bytes"] == artifact.stat().st_size
            assert descriptor["mtime_ns"] == artifact.stat().st_mtime_ns
    loaded = raw_cache.read_chunk_index(index)
    assert len(loaded) == 16


def test_checksum_free_timeline_paths_are_reversible_and_collision_free(tmp_path: Path) -> None:
    chunks = [
        {
            "timeline_id": timeline_id,
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        }
        for timeline_id in ("a/b", "a_b")
    ]
    index = raw_cache.export_raw_chunks(
        chunks,
        tmp_path / "raw",
    )
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    paths = {row["values"]["path"] for row in rows}
    assert "chunks/a%2Fb/video/chunk-000000.values.npy" in paths
    assert "chunks/a_b/video/chunk-000001.values.npy" in paths
    assert len(paths) == 2


def test_checksum_free_structural_metadata_detects_size_and_mtime_drift(tmp_path: Path) -> None:
    index = raw_cache.export_raw_chunks(
        [
            {
                "timeline_id": "run-1",
                "modality": "video",
                "values": _values("video"),
                "timestamps": [0.25],
            }
        ],
        tmp_path / "raw",
    )
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    rows[0]["values"]["size_bytes"] += 1
    index.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(raw_cache.RawCacheIntegrityError, match="byte-size mismatch"):
        raw_cache.read_chunk_index(index)

    rows[0]["values"]["size_bytes"] -= 1
    rows[0]["values"]["mtime_ns"] += 1
    index.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(raw_cache.RawCacheIntegrityError, match="mtime mismatch"):
        raw_cache.read_chunk_index(index)


def test_checksum_free_export_remains_atomic_on_extractor_failure(tmp_path: Path) -> None:
    output = tmp_path / "raw"

    def failing_chunks():
        yield {
            "timeline_id": "run-1",
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        }
        raise RuntimeError("upstream failed")

    try:
        raw_cache.export_raw_chunks(
            failing_chunks(),
            output,
        )
    except raw_cache.RawCacheTransactionError:
        pass
    assert not output.exists()


def test_checksum_free_export_resumes_and_repairs_staged_chunks(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    chunks = [
        {
            "timeline_id": "run-1",
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        },
        {
            "timeline_id": "run-1",
            "modality": "audio",
            "values": _values("audio"),
            "timestamps": [0.25],
        },
    ]

    def interrupted():
        yield from chunks
        raise RuntimeError("interrupted")

    with pytest.raises(raw_cache.RawCacheTransactionError, match="interrupted"):
        raw_cache.export_raw_chunks(interrupted(), output)

    staging = tmp_path / ".raw.staging"
    video = staging / "chunks/run-1/video/chunk-000000.values.npy"
    audio_support = staging / "chunks/run-1/audio/chunk-000001.support.npy"
    assert not output.exists()
    assert not (staging / "chunks.jsonl").exists()
    assert video.is_file()
    assert audio_support.is_file()

    retained_ns = 1_600_000_000_000_000_000
    os.utime(video, ns=(retained_ns, retained_ns))
    audio_support.unlink()
    orphan = video.parent / ".interrupted.npy.deadbeef.tmp"
    orphan.write_bytes(b"partial")

    index = raw_cache.export_raw_chunks(iter(chunks), output)

    assert index == output / "chunks.jsonl"
    assert not staging.exists()
    retained = output / "chunks/run-1/video/chunk-000000.values.npy"
    assert retained.stat().st_mtime_ns == retained_ns
    assert not (retained.parent / orphan.name).exists()
    records = raw_cache.read_chunk_index(index)
    assert len(records) == 2
    for record in records:
        loaded = record.load()
        np.testing.assert_array_equal(loaded.values, _values(record.modality))
        assert loaded.support.tolist() == [True]
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    _assert_no_checksum_fields(rows)


def test_checksum_free_export_recovers_from_array_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "raw"
    chunk = {
        "timeline_id": "run-1",
        "modality": "video",
        "values": _values("video"),
        "timestamps": [0.25],
    }
    original_save = raw_cache.np.save

    def fail_during_save(handle, array, *, allow_pickle):
        handle.write(b"partial")
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(raw_cache.np, "save", fail_during_save)
    with pytest.raises(raw_cache.RawCacheTransactionError, match="write interrupted"):
        raw_cache.export_raw_chunks([chunk], output)

    staging = tmp_path / ".raw.staging"
    target = staging / "chunks/run-1/video/chunk-000000.values.npy"
    assert not output.exists()
    assert not target.exists()
    assert not list(staging.rglob(".*.tmp"))

    monkeypatch.setattr(raw_cache.np, "save", original_save)
    index = raw_cache.export_raw_chunks([chunk], output)
    record = raw_cache.read_chunk_index(index)[0]
    np.testing.assert_array_equal(
        record.load().values,
        _values("video"),
    )


def test_raw_only_rejects_nonliteral_extractor_kwargs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    kwargs = _raw_kwargs(tmp_path)
    kwargs["dataset"] = "algonauts"
    with pytest.raises(raw_cache.RawCacheError, match="exactly|unexpected"):
        materializer.materialize_native_features(
            export_raw_only=True,
            dataset="hcp_movies",
            source_root=source_root,
            extractor=materializer.iter_tribev2_raw_chunks,
            extractor_kwargs=kwargs,
            generated_index_dir=tmp_path / "raw",
        )


def test_raw_only_rejects_custom_extractor(tmp_path: Path) -> None:
    with pytest.raises(raw_cache.RawCacheError, match="canonical"):
        materializer.materialize_native_features(
            export_raw_only=True,
            dataset="hcp_movies",
            source_root=tmp_path,
            extractor=lambda **_: (),
            extractor_kwargs=_raw_kwargs(tmp_path),
            generated_index_dir=tmp_path / "raw",
        )


def _patch_full_hcp_source(monkeypatch, *, run_count: int = 725, timeline_count: int = 8) -> tuple[str, ...]:
    timeline_ids = tuple(f"timeline-{index}" for index in range(timeline_count))
    discovered = [
        SimpleNamespace(timeline_id=timeline_ids[index % timeline_count])
        for index in range(run_count)
    ]
    monkeypatch.setattr(
        sources,
        "discover_runs",
        lambda dataset, root, **kwargs: tuple(discovered),
    )
    monkeypatch.setattr(
        sources,
        "audit_run_sources",
        lambda runs, *, require_full_inventory: {
            "dataset_id": "hcp_movies",
            "ready": True,
            "full_inventory": True,
            "run_count": len(runs),
            "timelines": list(timeline_ids),
            "blockers": [],
        },
    )
    return timeline_ids


def _raw_only_call(tmp_path: Path, monkeypatch, extractor) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    return materializer.materialize_native_features(
        export_raw_only=True,
        dataset="hcp_movies",
        source_root=source_root,
        extractor=extractor,
        extractor_kwargs=_raw_kwargs(tmp_path),
        generated_index_dir=tmp_path / "raw",
    )


def test_raw_only_rejects_missing_extractor_timelines(tmp_path: Path, monkeypatch) -> None:
    timeline_ids = _patch_full_hcp_source(monkeypatch)

    def fake_tribe(**kwargs):
        for modality_index, modality in enumerate(("video", "audio")):
            yield {
                "timeline_id": timeline_ids[0],
                "modality": modality,
                "values": _values(modality),
                "timestamps": [0.25],
                "chunk_index": modality_index,
            }

    monkeypatch.setattr(materializer, "iter_tribev2_raw_chunks", fake_tribe)
    with pytest.raises(raw_cache.RawCacheError, match="every discovered"):
        _raw_only_call(tmp_path, monkeypatch, materializer.iter_tribev2_raw_chunks)
    assert not (tmp_path / "raw").exists()


def test_raw_only_rejects_extra_extractor_timelines(tmp_path: Path, monkeypatch) -> None:
    _patch_full_hcp_source(monkeypatch)

    def fake_tribe(**kwargs):
        yield {
            "timeline_id": "timeline-extra",
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        }

    monkeypatch.setattr(materializer, "iter_tribev2_raw_chunks", fake_tribe)
    with pytest.raises(raw_cache.RawCacheError, match="outside source-root"):
        _raw_only_call(tmp_path, monkeypatch, materializer.iter_tribev2_raw_chunks)
    assert not (tmp_path / "raw").exists()


def test_raw_only_rejects_non_string_extractor_timeline_id(tmp_path: Path, monkeypatch) -> None:
    _patch_full_hcp_source(monkeypatch)

    def fake_tribe(**kwargs):
        yield {
            "timeline_id": None,
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        }

    monkeypatch.setattr(materializer, "iter_tribev2_raw_chunks", fake_tribe)
    with pytest.raises(raw_cache.RawCacheError, match="non-empty string timeline_id"):
        _raw_only_call(tmp_path, monkeypatch, materializer.iter_tribev2_raw_chunks)


@pytest.mark.parametrize(
    "events_path",
    [
        lambda values: values[:3],
        lambda values: values + [values[0]],
        lambda values: [values[0], values[1], values[2], values[2]],
    ],
)
def test_raw_only_requires_exactly_four_distinct_event_tsv_paths(
    tmp_path: Path, events_path
) -> None:
    kwargs = _raw_kwargs(tmp_path)
    kwargs["events_path"] = events_path(kwargs["events_path"])
    with pytest.raises(raw_cache.RawCacheError, match="four"):
        materializer.materialize_native_features(
            export_raw_only=True,
            dataset="hcp_movies",
            source_root=tmp_path,
            extractor=materializer.iter_tribev2_raw_chunks,
            extractor_kwargs=kwargs,
            generated_index_dir=tmp_path / "raw",
        )


def test_raw_only_cli_rejects_retired_hcp_alias() -> None:
    with pytest.raises(SystemExit):
        materializer.build_arg_parser().parse_args(["--dataset", "hcp"])


@pytest.mark.parametrize(
    "dataset,run_count,timeline_count",
    [("algonauts", 1391, 1391), ("hcp_movies", 724, 8)],
)
def test_source_discovery_rejects_partial_full_inventory(
    tmp_path: Path, monkeypatch, dataset: str, run_count: int, timeline_count: int
) -> None:
    source_root = tmp_path / dataset
    source_root.mkdir()
    timeline_ids = tuple(f"timeline-{index}" for index in range(timeline_count))
    runs = [SimpleNamespace(timeline_id=timeline_ids[index % timeline_count]) for index in range(run_count)]
    monkeypatch.setattr(sources, "discover_runs", lambda dataset, root, **kwargs: tuple(runs))
    monkeypatch.setattr(
        sources,
        "audit_run_sources",
        lambda runs, *, require_full_inventory: {
            "dataset_id": dataset,
            "ready": True,
            "full_inventory": True,
            "run_count": len(runs),
            "timelines": list(timeline_ids),
            "blockers": [],
        },
    )
    with pytest.raises(raw_cache.RawCacheError, match="exact full inventory"):
        materializer.discover_source_timeline_ids(source_root, dataset=dataset)


def test_checksum_free_sanitizes_all_retired_digest_tokens(tmp_path: Path) -> None:
    keys = (
        "sha", "sha1", "sha224", "sha256", "sha384", "sha512",
        "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum",
        "md5", "md5sum", "hash", "hashes", "checksum", "checksums",
        "digest", "digests", "source_sha256", "configSHA256",
    )
    index = raw_cache.export_raw_chunks(
        [{
            "timeline_id": "run-1",
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
            "metadata": {key: "secret" for key in keys},
        }],
        tmp_path / "raw",
    )
    row = json.loads(index.read_text(encoding="utf-8"))
    assert row["metadata"] == {}


@pytest.mark.parametrize("field", ["shape", "dtype"])
def test_checksum_free_descriptor_requires_structural_fields(
    tmp_path: Path, field: str
) -> None:
    index = raw_cache.export_raw_chunks(
        [{
            "timeline_id": "run-1",
            "modality": "video",
            "values": _values("video"),
            "timestamps": [0.25],
        }],
        tmp_path / "raw",
    )
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    del rows[0]["values"][field]
    index.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(raw_cache.RawCacheIntegrityError, match=f"missing {field}"):
        raw_cache.read_chunk_index(index)


def test_tribev2_checksum_free_iterator_skips_config_digest(
    tmp_path: Path, monkeypatch
) -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0},
        ]
    )
    config = {
        "data": {
            "layers_to_use": [0.5, 0.75, 1.0],
            "layer_aggregation": "group_mean",
            "video_feature": {"infra": {"folder": "old"}, "image": {"cache_n_layers": 20}},
            "audio_feature": {"infra": {"folder": "old"}, "audio": {"cache_n_layers": 20}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(raw_cache, "_load_tribev2_config", lambda _: config)
    class FakeData:
        frequency = 2.0
        batch_size = 1
        num_workers = 0
        shuffle_train = False
        shuffle_val = False

        def get_loaders(self, *, events, split_to_build):
            return {
                "all": [
                    SimpleNamespace(
                        data={
                            "video": np.ones((1, 2, 1408, 2), dtype=np.float32),
                            "audio": np.ones((1, 2, 1024, 2), dtype=np.float32),
                        },
                        segments=[SimpleNamespace(start=0.0, duration=1.0)],
                    )
                ]
            }

    chunks = list(
        raw_cache.iter_tribev2_raw_chunks(
            events_path=events,
            config_path=config_path,
            cache_dir=cache_root,
            dataset="hcp",
            data=FakeData(),
        )
    )
    assert [chunk.modality for chunk in chunks] == ["video", "audio"]
    assert all("sha256" not in key.lower() for chunk in chunks for key in chunk.metadata)
