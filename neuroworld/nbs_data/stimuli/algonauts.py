"""Algonauts stimulus-cache contract."""

from __future__ import annotations

from typing import Any

from .common import ALGONAUTS_MODALITIES, DATASET_TR_SECONDS, build_native_timeline
from .raw_cache import NativeTimelineSpec, iter_tribev2_raw_chunks


DATASET = "algonauts"
TRIBE_DATASET = "algonauts"
NATIVE_TR_SECONDS = DATASET_TR_SECONDS[DATASET]
MODALITIES = ALGONAUTS_MODALITIES
EXPECTED_RUN_COUNT = 1_392
EXPECTED_TIMELINE_COUNT = 1_392


def iter_raw_chunks(**kwargs: Any):
    """Stream Algonauts pre-projector chunks from the shared TRIBEv2 cache."""

    supplied = kwargs.pop("dataset", DATASET)
    if str(supplied).strip().lower().replace("-", "_") not in {
        "algonauts",
        "algonaut",
    }:
        raise ValueError(f"Algonauts extractor received dataset={supplied!r}")
    return iter_tribev2_raw_chunks(dataset=TRIBE_DATASET, **kwargs)


def timeline_spec(
    timeline_id: str,
    *,
    origin_seconds: float,
    n_bins: int,
    require_text: bool = True,
) -> NativeTimelineSpec:
    """Create an Algonauts native-clock timing specification."""

    return NativeTimelineSpec(
        timeline_id=timeline_id,
        dataset=DATASET,
        origin_seconds=origin_seconds,
        native_tr_seconds=NATIVE_TR_SECONDS,
        n_bins=n_bins,
        require_text=require_text,
    )


__all__ = [
    "DATASET",
    "EXPECTED_RUN_COUNT",
    "EXPECTED_TIMELINE_COUNT",
    "MODALITIES",
    "NATIVE_TR_SECONDS",
    "TRIBE_DATASET",
    "build_native_timeline",
    "iter_raw_chunks",
    "timeline_spec",
]
