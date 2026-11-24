"""Contract tests for native-clock stimulus preparation."""

from __future__ import annotations

import numpy as np
import pytest

from state_of_mind.nbs_data.stimuli.timing import (
    ALGONAUTS_MODALITIES,
    MODALITY_DIMS,
    ModalityChunk,
    StitchedModality,
    StimulusValidationError,
    build_native_timeline,
    extract_batch_data,
    pool_modality_to_bold,
    stitch_modality_chunks,
    validate_dataset_modalities,
)


def _values(modality: str, n: int, value: float = 0.0) -> np.ndarray:
    return np.full((n, 2, MODALITY_DIMS[modality]), value, dtype=np.float32)


def _chunk(
    modality: str,
    timestamps: list[float],
    value: float,
    *,
    support: list[bool] | None = None,
    source_id: str = "",
) -> ModalityChunk:
    return ModalityChunk(
        modality=modality,
        values=_values(modality, len(timestamps), value),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        support=support,
        source_id=source_id,
    )


@pytest.mark.parametrize("dataset", ["hcp", "hcp_movies"])
def test_video_audio_are_required_and_text_is_non_algonauts_only(dataset: str) -> None:
    assert validate_dataset_modalities({"video": (), "audio": ()}, dataset) == (
        "video",
        "audio",
    )
    with pytest.raises(StimulusValidationError, match="missing required"):
        validate_dataset_modalities({"video": ()}, dataset)
    with pytest.raises(StimulusValidationError, match="does not permit"):
        validate_dataset_modalities({"video": (), "audio": (), "text": ()}, dataset)

    assert validate_dataset_modalities(
        {name: () for name in ALGONAUTS_MODALITIES}, "algonauts"
    ) == ALGONAUTS_MODALITIES
    with pytest.raises(StimulusValidationError, match="Strict Algonauts"):
        validate_dataset_modalities({"video": (), "audio": ()}, "algonauts", require_text=True)


def test_batch_data_converts_neuralset_axes_without_upstream_import() -> None:
    raw = np.zeros((3, 2, MODALITY_DIMS["video"], 4), dtype=np.float32)
    raw[1, 0, 0, 2] = 7.0
    extracted = extract_batch_data({"video": raw}, "video", batch_index=1)
    assert extracted.shape == (4, 2, MODALITY_DIMS["video"])
    assert extracted.dtype == np.float32
    assert extracted[2, 0, 0] == 7.0

    unbatched = np.zeros((2, MODALITY_DIMS["audio"], 3), dtype=np.float64)
    assert extract_batch_data(unbatched, "audio").shape == (
        3,
        2,
        MODALITY_DIMS["audio"],
    )


def test_post_transformer_hidden_features_are_rejected_even_when_batch_is_two() -> None:
    hidden = np.zeros((2, 5, 1152), dtype=np.float32)
    with pytest.raises(StimulusValidationError, match="post-Transformer|1152"):
        extract_batch_data({"video": hidden}, "video")


@pytest.mark.parametrize("modality", ["video", "audio", "text"])
def test_chunk_contract_rejects_nonfinite_wrong_dtype_and_wrong_layers(modality: str) -> None:
    with pytest.raises(StimulusValidationError, match="floating dtype"):
        ModalityChunk(modality, np.zeros((1, 2, MODALITY_DIMS[modality]), dtype=np.int32), [0.0])
    bad = _values(modality, 1)
    bad[0, 0, 0] = np.nan
    with pytest.raises(StimulusValidationError, match="non-finite"):
        ModalityChunk(modality, bad, [0.0])
    with pytest.raises(StimulusValidationError, match="layer axis"):
        ModalityChunk(
            modality,
            np.zeros((1, 1, MODALITY_DIMS[modality]), dtype=np.float32),
            [0.0],
        )
    with pytest.raises(StimulusValidationError, match="support"):
        ModalityChunk(modality, _values(modality, 1), [0.0], support=[2])


def test_chunk_contract_requires_regular_two_hz_sample_centres() -> None:
    with pytest.raises(StimulusValidationError, match="regular sample centres"):
        _chunk("audio", [0.25, 1.0], 0.0)


@pytest.mark.parametrize("tr", [1.0, 1.49, 1.75])
def test_pool_uses_sample_centres_and_half_open_native_bins(tr: float) -> None:
    timestamps = np.asarray([0.25 * tr, 0.5 * tr, tr, 1.5 * tr], dtype=np.float64)
    values = _values("audio", len(timestamps))
    values[:, :, 0] = np.asarray([1.0, 3.0, 5.0, 7.0])[:, None]
    # Pooling itself is defined for arbitrary, strictly ordered sample centres;
    # production ModalityChunk inputs are separately constrained to regular
    # 2-Hz clocks above.
    stitched = StitchedModality(
        modality="audio",
        values=values,
        timestamps=timestamps,
        support=np.ones(len(timestamps), dtype=np.bool_),
        source_ids=tuple("clock" for _ in timestamps),
    )
    pooled = pool_modality_to_bold(
        "audio", stitched, origin_seconds=0.0, native_tr_seconds=tr, n_bins=3
    )
    assert pooled.values.shape == (3, 2, MODALITY_DIMS["audio"])
    assert pooled.values.dtype == np.float32
    assert pooled.values[:, 0, 0].tolist() == pytest.approx([2.0, 6.0, 0.0])
    assert pooled.support.tolist() == [True, True, False]
    assert pooled.source_counts.tolist() == [2, 2, 0]


def test_overlap_stitching_prefers_supported_earliest_source_without_averaging() -> None:
    first = _chunk("video", [0.0, 0.5, 1.0], 1.0, source_id="a")
    second = _chunk("video", [0.5, 1.0, 1.5], 9.0, source_id="b")
    stitched = stitch_modality_chunks("video", [second, first])
    assert stitched.timestamps.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert stitched.values[:, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 9.0])
    assert stitched.support.tolist() == [True, True, True, True]

    padded = _chunk(
        "video", [1.5, 2.0], 100.0, support=[False, False], source_id="padding"
    )
    valid = _chunk("video", [1.5, 2.0], 4.0, source_id="valid")
    stitched = stitch_modality_chunks("video", [padded, valid])
    assert stitched.values[:, 0, 0].tolist() == pytest.approx([4.0, 4.0])
    assert stitched.support.tolist() == [True, True]


def test_missing_bins_are_finite_zero_and_unsupported() -> None:
    stitched = StitchedModality(
        modality="audio",
        values=_values("audio", 2, 2.0),
        timestamps=np.asarray([0.1, 2.1], dtype=np.float64),
        support=np.ones(2, dtype=np.bool_),
    )
    pooled = pool_modality_to_bold(
        "audio", stitched, origin_seconds=0.0, native_tr_seconds=1.0, n_bins=4
    )
    assert pooled.support.tolist() == [True, False, True, False]
    assert pooled.source_counts.tolist() == [1, 0, 1, 0]
    assert np.all(pooled.values[1] == 0)
    assert np.all(np.isfinite(pooled.values))


def test_build_native_timeline_is_in_memory_and_semantic() -> None:
    chunks = {
        "video": [_chunk("video", [0.25, 0.75], 1.0, source_id="v")],
        "audio": [_chunk("audio", [0.25, 0.75], 2.0, source_id="a")],
        "text": [_chunk("text", [0.25, 0.75], 3.0, source_id="t")],
    }
    timeline = build_native_timeline(
        dataset="algonauts",
        timeline_id="sub-01/run-01",
        chunks=chunks,
        origin_seconds=0.0,
        native_tr_seconds=1.49,
        n_bins=2,
        require_text=True,
    )
    assert timeline.modalities["video"].values.shape == (2, 2, 1408)
    assert timeline.modalities["text"].values.dtype == np.float32
    assert timeline.provenance["offline_normalization"] is False
    assert timeline.provenance["hemodynamic_lag_seconds"] is None
    np.testing.assert_array_equal(
        timeline.modalities["audio"].values[0],
        np.full((2, MODALITY_DIMS["audio"]), 2.0, dtype=np.float32),
    )


def test_duplicate_tolerance_uses_distance_not_rounding_buckets() -> None:
    first = _chunk("audio", [0.49], 1.0, source_id="a")
    second = _chunk("audio", [0.51], 2.0, source_id="b")
    stitched = stitch_modality_chunks(
        "audio", [second, first], duplicate_tolerance_seconds=1.0
    )
    assert stitched.n_samples == 1
    assert stitched.values[0, 0, 0] == 1.0


@pytest.mark.parametrize("n_bins", [True, 1.5, 0])
def test_pool_rejects_non_positive_or_non_integral_bin_counts(n_bins: object) -> None:
    stitched = StitchedModality(
        modality="audio",
        values=_values("audio", 1),
        timestamps=np.asarray([0.25]),
        support=np.asarray([True]),
    )
    with pytest.raises(StimulusValidationError, match="positive integer"):
        pool_modality_to_bold(
            "audio",
            stitched,
            origin_seconds=0.0,
            native_tr_seconds=1.0,
            n_bins=n_bins,
        )


def test_build_native_timeline_rejects_wrong_dataset_clock() -> None:
    chunks = {
        "video": [_chunk("video", [0.25, 0.75], 1.0)],
        "audio": [_chunk("audio", [0.25, 0.75], 2.0)],
    }
    with pytest.raises(StimulusValidationError, match="native TR|TR"):
        build_native_timeline(
            dataset="hcp",
            timeline_id="run-1",
            chunks=chunks,
            origin_seconds=0.0,
            native_tr_seconds=1.75,
            n_bins=2,
        )
