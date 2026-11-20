"""Focused contracts for the raw NeuralSet-cache adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import neuroworld.nbs_data.stimuli.raw_cache as raw_cache_module
from neuroworld.nbs_data.stimuli.raw_cache import (
    NativeTimelineSpec,
    RawCacheError,
    RawCacheIntegrityError,
    RawCacheChunk,
    chunks_from_explicit_extractor,
    export_raw_chunks,
    iter_indexed_timelines,
    neuralset_batch_to_chunk,
    read_chunk_index,
    sample_center_timestamps,
    support_mask_from_events,
)
from neuroworld.nbs_data.stimuli.timing import MODALITY_DIMS


def _values(modality: str, n: int, value: float = 0.0) -> np.ndarray:
    return np.full((n, 2, MODALITY_DIMS[modality]), value, dtype=np.float32)


def _chunk(timeline: str, modality: str, values: np.ndarray, timestamps: list[float], *, support=None) -> RawCacheChunk:
    return RawCacheChunk(
        timeline_id=timeline,
        modality=modality,
        values=values,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        support=support,
        source_id=f"{timeline}:{modality}",
    )


def test_sample_centres_and_event_support_are_physical_half_open() -> None:
    timestamps = sample_center_timestamps(1.0, 4)
    assert timestamps.tolist() == pytest.approx([1.25, 1.75, 2.25, 2.75])
    support = support_mask_from_events(
        timestamps,
        [
            {"timeline": "run-1", "type": "Video", "start": 1.5, "duration": 1.0},
            {"timeline": "run-2", "type": "Video", "start": 0.0, "duration": 10.0},
        ],
        timeline_id="run-1",
        event_types=("Video",),
    )
    assert support.tolist() == [False, True, True, False]


def test_neuralset_batch_conversion_uses_stimulus_axis_contract() -> None:
    batch = np.zeros((1, 2, MODALITY_DIMS["audio"], 3), dtype=np.float32)
    batch[0, 1, 3, 2] = 7.0
    chunk = neuralset_batch_to_chunk(
        "run-1",
        "audio",
        {"audio": batch},
        start_seconds=0.0,
        support_intervals=((0.0, 1.0),),
    )
    assert chunk.values.shape == (3, 2, 1024)
    assert chunk.timestamps.tolist() == pytest.approx([0.25, 0.75, 1.25])
    assert chunk.values[2, 1, 3] == 7.0
    assert chunk.support.tolist() == [True, True, False]


def test_export_import_round_trip_and_native_pooling(tmp_path: Path) -> None:
    timeline_id = "HCP7T/100610/task-movie/run-01"
    chunks = [
        _chunk(timeline_id, "video", _values("video", 2, 1.0), [0.25, 0.75]),
        _chunk(timeline_id, "audio", _values("audio", 2, 2.0), [0.25, 0.75]),
    ]
    index = export_raw_chunks(chunks, tmp_path / "raw")
    records = read_chunk_index(index)
    assert len(records) == 2
    assert {record.modality for record in records} == {"video", "audio"}
    loaded = {record.modality: record.load() for record in records}
    np.testing.assert_array_equal(loaded["video"].values, chunks[0].values)

    spec = NativeTimelineSpec(
        timeline_id=timeline_id,
        dataset="hcp",
        origin_seconds=0.0,
        native_tr_seconds=1.0,
        n_bins=2,
    )
    [(returned_spec, timeline)] = list(iter_indexed_timelines(index, [spec]))
    assert returned_spec == spec
    assert timeline.modalities["video"].values.shape == (2, 2, 1408)
    assert timeline.provenance["offline_normalization"] is False


@pytest.mark.parametrize(
    ("dataset", "wrong_tr"),
    (("algonauts", 1.0), ("hcp_movies", 1.75)),
)
def test_native_timeline_spec_rejects_wrong_dataset_clock(
    dataset: str, wrong_tr: float
) -> None:
    with pytest.raises(RawCacheError, match="native TR|TR"):
        NativeTimelineSpec(
            timeline_id="run-1",
            dataset=dataset,
            origin_seconds=0.0,
            native_tr_seconds=wrong_tr,
            n_bins=2,
        )


def test_tampered_artifact_is_rejected_before_use(tmp_path: Path) -> None:
    timeline_id = "run-1"
    index = export_raw_chunks(
        [
            _chunk(timeline_id, "video", _values("video", 1, 1.0), [0.25]),
            _chunk(timeline_id, "audio", _values("audio", 1, 2.0), [0.25]),
        ],
        tmp_path / "raw",
    )
    record = read_chunk_index(index)[0]
    values_path = index.parent / str(record.values["path"])
    with values_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RawCacheIntegrityError, match="byte-size mismatch"):
        read_chunk_index(index)


def test_dataset_text_rule_and_missing_timing_fail_closed(tmp_path: Path) -> None:
    # Specs are validated before opening an index, so a malformed/empty timing
    # selection is deterministic even when the index path is absent.
    with pytest.raises(RawCacheError, match="At least one native timeline specification"):
        iter_indexed_timelines(tmp_path / "missing.jsonl", [])

    # The native stimulus validator rejects text for non-Algonauts datasets;
    # this adapter intentionally does not silently drop it.
    timeline_id = "run-1"
    index = export_raw_chunks(
        [
            _chunk(timeline_id, "video", _values("video", 1), [0.25]),
            _chunk(timeline_id, "audio", _values("audio", 1), [0.25]),
            _chunk(timeline_id, "text", _values("text", 1), [0.25]),
        ],
        tmp_path / "raw",
    )
    with pytest.raises(ValueError, match="does not permit"):
        list(
            iter_indexed_timelines(
                index,
                [
                NativeTimelineSpec(
                    timeline_id=timeline_id,
                    dataset="hcp",
                    origin_seconds=0.0,
                    native_tr_seconds=1.0,
                    n_bins=1,
                )
                ],
            )
        )




def test_streaming_extractor_is_lazy_and_source_provenance_survives(tmp_path: Path) -> None:
    observed: list[int] = []
    source_path = tmp_path / "original-cache.bin"
    source_path.write_bytes(b"original")

    def extractor():
        for index in range(2):
            observed.append(index)
            yield RawCacheChunk(
                timeline_id="run-1",
                modality="video",
                values=_values("video", 1, float(index)),
                timestamps=[index + 0.25],
                source_id="upstream-key",
                source_path=source_path,
            )

    chunks = chunks_from_explicit_extractor(extractor)
    assert observed == []
    first = next(chunks)
    assert observed == [0]
    second = next(chunks)
    assert observed == [0, 1]
    assert first.source_path == str(source_path)

    index = export_raw_chunks((chunk for chunk in (first, second)), tmp_path / "raw")
    record = read_chunk_index(index)[0]
    assert record.source_id == "upstream-key"
    assert record.source_path == str(source_path.resolve())
    loaded = record.load()
    assert loaded.source_path == str(source_path.resolve())


def test_indexed_timeline_exact_mode_and_explicit_subset_mode(tmp_path: Path) -> None:
    chunks = []
    for timeline_id, value in (("run-a", 1.0), ("run-b", 2.0)):
        chunks.extend(
            [
                _chunk(timeline_id, "video", _values("video", 1, value), [0.25]),
                _chunk(timeline_id, "audio", _values("audio", 1, value), [0.25]),
            ]
        )
    index = export_raw_chunks(chunks, tmp_path / "raw")
    spec = NativeTimelineSpec(
        timeline_id="run-a",
        dataset="hcp",
        origin_seconds=0.0,
        native_tr_seconds=1.0,
        n_bins=1,
    )
    with pytest.raises(RawCacheError, match="without explicit native timing"):
        list(iter_indexed_timelines(index, [spec]))
    selected = list(iter_indexed_timelines(index, [spec], allow_unrequested=True))
    assert [item[0].timeline_id for item in selected] == ["run-a"]
    assert selected[0][1].modalities["video"].values.shape == (1, 2, 1408)


def test_tribev2_production_constructor_unpacks_data_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0},
        ]
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    config = {
        "data": {
            "features_to_use": ["text", "audio", "video"],
            "layers_to_use": [0.5, 0.75, 1.0],
            "layer_aggregation": "group_mean",
            "frequency": 2.0,
            "batch_size": 8,
            "num_workers": 4,
            "shuffle_train": True,
            "shuffle_val": True,
            "video_feature": {
                "infra": {"folder": "old-video", "cluster": "old"},
                "image": {"cache_n_layers": 20},
            },
            "audio_feature": {
                "infra": {"folder": "old-audio", "cluster": "old"},
                "audio": {"cache_n_layers": 20},
            },
            # Text is intentionally present in the checkpoint config, but HCP
            # must remove it from features_to_use before Data construction.
            "text_feature": {
                "infra": {"folder": "old-text", "cluster": "old"},
                "text": {"cache_n_layers": 20},
            },
            "image_feature": {"image": {"layer_aggregation": "mean"}},
            "study": {"path": "old-study", "infra_timelines": "stale"},
            "neuro": {"infra": {"folder": "stale-neuro"}},
            "subject_id": {"predefined_mapping": {"s1": 1}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    monkeypatch.setattr(raw_cache_module, "_load_tribev2_config", lambda _: config)
    captured: dict[str, object] = {}
    dims = {"video": 1408, "audio": 1024}

    class FakeData:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.frequency = kwargs["frequency"]
            self.batch_size = kwargs["batch_size"]
            self.num_workers = kwargs["num_workers"]
            self.shuffle_train = kwargs["shuffle_train"]
            self.shuffle_val = kwargs["shuffle_val"]

        def get_loaders(self, *, events, split_to_build):
            assert split_to_build == "all"
            batch = SimpleNamespace(
                data={
                    modality: np.ones((1, 2, dimension, 2), dtype=np.float32)
                    for modality, dimension in dims.items()
                },
                segments=[SimpleNamespace(start=0.0, duration=1.0)],
            )
            return {"all": [batch]}

    tribe_package = ModuleType("tribev2")
    tribe_main = ModuleType("tribev2.main")
    tribe_main.Data = FakeData
    tribe_package.main = tribe_main
    monkeypatch.setitem(sys.modules, "tribev2", tribe_package)
    monkeypatch.setitem(sys.modules, "tribev2.main", tribe_main)

    chunks = list(
        raw_cache_module.iter_tribev2_raw_chunks(
            events_path=events,
            config_path=config_path,
            cache_dir=cache_root,
            dataset="hcp",
        )
    )
    assert [chunk.modality for chunk in chunks] == ["video", "audio"]
    assert captured["features_to_use"] == ["video", "audio"]
    assert captured["batch_size"] == 1
    assert captured["num_workers"] == 0
    assert captured["shuffle_train"] is False
    assert captured["shuffle_val"] is False
    assert captured["video_feature"]["infra"] == {"folder": str(cache_root), "cluster": None}
    assert captured["audio_feature"]["infra"] == {"folder": str(cache_root), "cluster": None}
    assert captured["text_feature"]["infra"] == {"folder": str(cache_root), "cluster": None}
    assert captured["study"] == {"path": "."}
    assert captured["neuro"] == {}
    assert captured["subject_id"] == {"predefined_mapping": None}


def test_tribev2_data_only_iterator_uses_raw_batches_and_event_support() -> None:
    pd = pytest.importorskip("pandas")
    rows = [
        {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
        {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
        {"timeline": "run-1", "subject": "s1", "type": "Word", "start": 0.0, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
        # NeuralSet's audited Sentence enclosure creates the -1 us raw start
        # that the adapter must shift back to the physical 0 s clock.
        {"timeline": "run-1", "subject": "s1", "type": "Sentence", "start": -1e-6, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
    ]
    events = pd.DataFrame(rows)
    dims = {"video": 1408, "audio": 1024, "text": 3072}

    class FakeData:
        frequency = 2.0
        batch_size = 1
        num_workers = 0
        shuffle_train = False
        shuffle_val = False
        layers_to_use = [0.5, 0.75, 1.0]
        layer_aggregation = "group_mean"

        def get_loaders(self, *, events, split_to_build):
            assert split_to_build == "all"
            batch_data = {
                modality: np.ones((1, 2, dimension, 2), dtype=np.float32)
                for modality, dimension in dims.items()
            }
            batch = SimpleNamespace(
                data=batch_data,
                segments=[SimpleNamespace(start=-1e-6, duration=1.0)],
            )
            return {"all": [batch]}

    chunks = list(
        raw_cache_module.iter_tribev2_raw_chunks(
            events_path=events,
            dataset="algonauts",
            selected_timeline_ids=["run-1"],
            data=FakeData(),
        )
    )
    assert [chunk.modality for chunk in chunks] == ["video", "audio", "text"]
    assert all(chunk.timestamps.tolist() == pytest.approx([0.25, 0.75]) for chunk in chunks)
    assert all(chunk.support.tolist() == [True, True] for chunk in chunks)
    assert all(chunk.values.shape[0:] == (2, 2, dims[chunk.modality]) for chunk in chunks)



def test_tribev2_algonauts_transcript_free_timeline_has_unsupported_text() -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0},
        ]
    )
    dims = {"video": 1408, "audio": 1024, "text": 3072}

    class FakeData:
        frequency = 2.0
        batch_size = 1
        num_workers = 0
        shuffle_train = False
        shuffle_val = False
        layers_to_use = [0.5, 0.75, 1.0]
        layer_aggregation = "group_mean"

        def get_loaders(self, *, events, split_to_build):
            assert split_to_build == "all"
            return {
                "all": [
                    SimpleNamespace(
                        data={
                            modality: np.ones((1, 2, dimension, 2), dtype=np.float32)
                            for modality, dimension in dims.items()
                            if modality != "text"
                        },
                        segments=[SimpleNamespace(start=0.0, duration=1.0)],
                    )
                ]
            }

    chunks = list(
        raw_cache_module.iter_tribev2_raw_chunks(
            events_path=events,
            dataset="algonauts",
            data=FakeData(),
        )
    )
    by_modality = {chunk.modality: chunk for chunk in chunks}
    assert set(by_modality) == set(dims)
    assert by_modality["video"].support.tolist() == [True, True]
    assert by_modality["audio"].support.tolist() == [True, True]
    assert by_modality["text"].support.tolist() == [False, False]
    assert not by_modality["text"].values.any()
    assert by_modality["text"].metadata["support_rows_present"] is False
    assert by_modality["text"].metadata["values_synthesized"] is True


@pytest.mark.parametrize("missing_modality", ["video", "audio"])
def test_tribev2_still_requires_physical_support_rows(missing_modality: str) -> None:
    pd = pytest.importorskip("pandas")
    event_type = "Audio" if missing_modality == "video" else "Video"
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": event_type, "start": 0.0, "stop": 1.0},
        ]
    )
    data = SimpleNamespace(
        frequency=2.0,
        batch_size=1,
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        layers_to_use=[0.5, 0.75, 1.0],
        layer_aggregation="group_mean",
        get_loaders=lambda **kwargs: {"all": []},
    )

    with pytest.raises(ValueError, match=f"lacks {missing_modality.title()} events"):
        list(
            raw_cache_module.iter_tribev2_raw_chunks(
                events_path=events,
                dataset="algonauts",
                data=data,
            )
        )


@pytest.mark.parametrize(
    ("timeline_stop", "segment_starts", "match"),
    [
        (2.0, (0.0,), "segments do not cover timeline"),
        (1.0, (5.0,), "first segment must start"),
        (2.0, (0.0, 1.1), "segments must be contiguous"),
        (2.0, (0.0, 0.9), "segments must be contiguous"),
    ],
)
def test_tribev2_raw_segments_cover_timeline_without_gaps_or_overlaps(
    timeline_stop: float,
    segment_starts: tuple[float, ...],
    match: str,
) -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": timeline_stop},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": timeline_stop},
        ]
    )

    class FakeData:
        frequency = 2.0
        batch_size = 1
        num_workers = 0
        shuffle_train = False
        shuffle_val = False
        layers_to_use = [0.5, 0.75, 1.0]
        layer_aggregation = "group_mean"

        def get_loaders(self, *, events, split_to_build):
            assert split_to_build == "all"
            batches = []
            for start in segment_starts:
                batches.append(
                    SimpleNamespace(
                        data={
                            "video": np.ones((1, 2, 1408, 2), dtype=np.float32),
                            "audio": np.ones((1, 2, 1024, 2), dtype=np.float32),
                        },
                        segments=[SimpleNamespace(start=start, duration=1.0)],
                    )
                )
            return {"all": batches}

    with pytest.raises(RawCacheError, match=match):
        list(
            raw_cache_module.iter_tribev2_raw_chunks(
                events_path=events,
                dataset="hcp",
                data=FakeData(),
            )
        )


@pytest.mark.parametrize(
    ("layers_to_use", "layer_aggregation", "match"),
    [
        (None, "group_mean", "Data.layers_to_use is missing"),
        ([0.5, 0.75, 0.9], "group_mean", "must be exactly"),
        ([0.5, 0.75, 1.0], None, "Data.layer_aggregation is missing"),
        ([0.5, 0.75, 1.0], "mean", "must be exactly"),
    ],
)
def test_tribev2_data_only_iterator_requires_exact_layer_contract(
    layers_to_use: list[float] | None,
    layer_aggregation: str | None,
    match: str,
) -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0},
        ]
    )
    data = SimpleNamespace(
        frequency=2.0,
        batch_size=1,
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        layers_to_use=layers_to_use,
        layer_aggregation=layer_aggregation,
        get_loaders=lambda **kwargs: {"all": []},
    )

    with pytest.raises(RawCacheError, match=match):
        list(
            raw_cache_module.iter_tribev2_raw_chunks(
                events_path=events,
                dataset="hcp",
                data=data,
            )
        )

def test_tribev2_loader_closes_when_chunk_generator_is_closed_early() -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        [
            {"timeline": "run-1", "subject": "s1", "type": "Video", "start": 0.0, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
            {"timeline": "run-1", "subject": "s1", "type": "Audio", "start": 0.0, "stop": 1.0, "task": "task", "movie": "movie", "chunk": "1"},
        ]
    )

    class ClosingLoader:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            yield SimpleNamespace(
                data={
                    "video": np.ones((1, 2, 1408, 2), dtype=np.float32),
                    "audio": np.ones((1, 2, 1024, 2), dtype=np.float32),
                },
                segments=[SimpleNamespace(start=0.0, duration=1.0)],
            )

        def close(self) -> None:
            self.closed = True

    loader = ClosingLoader()

    class FakeData:
        frequency = 2.0
        batch_size = 1
        num_workers = 0
        shuffle_train = False
        shuffle_val = False
        layers_to_use = [0.5, 0.75, 1.0]
        layer_aggregation = "group_mean"

        def get_loaders(self, *, events, split_to_build):
            return {"all": loader}

    chunks = raw_cache_module.iter_tribev2_raw_chunks(
        events_path=events,
        dataset="hcp",
        data=FakeData(),
    )
    first = next(chunks)
    assert first.modality == "video"
    chunks.close()
    assert loader.closed is True
