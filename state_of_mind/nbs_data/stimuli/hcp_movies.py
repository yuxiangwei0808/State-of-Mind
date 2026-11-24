"""HCP 7T movie stimulus-cache contract."""

from __future__ import annotations

from typing import Any

from .common import DATASET_TR_SECONDS, NON_ALGONAUTS_MODALITIES, build_native_timeline
from .raw_cache import NativeTimelineSpec, iter_tribev2_raw_chunks


DATASET = "hcp"
CLI_DATASET = "hcp_movies"
TRIBE_DATASET = "hcp"
NATIVE_TR_SECONDS = DATASET_TR_SECONDS[DATASET]
MODALITIES = NON_ALGONAUTS_MODALITIES
EXPECTED_RUN_COUNT = 725
EXPECTED_TIMELINE_COUNT = 8


def iter_raw_chunks(**kwargs: Any):
    """Stream HCP movie pre-projector chunks from the shared TRIBEv2 cache."""

    supplied = kwargs.pop("dataset", CLI_DATASET)
    normalized = str(supplied).strip().lower().replace("-", "_")
    if normalized not in {"hcp", "hcp_movies", "hcp_movie", "hcp_7t"}:
        raise ValueError(f"HCP movie extractor received dataset={supplied!r}")
    return iter_tribev2_raw_chunks(dataset=TRIBE_DATASET, **kwargs)


def timeline_spec(
    timeline_id: str,
    *,
    origin_seconds: float,
    n_bins: int,
) -> NativeTimelineSpec:
    """Create an HCP movie native-clock timing specification."""

    return NativeTimelineSpec(
        timeline_id=timeline_id,
        dataset=DATASET,
        origin_seconds=origin_seconds,
        native_tr_seconds=NATIVE_TR_SECONDS,
        n_bins=n_bins,
    )


__all__ = [
    "CLI_DATASET",
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
