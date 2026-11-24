from types import SimpleNamespace

import pytest

from state_of_mind.nbs_data import pipeline
from state_of_mind.nbs_data.stimuli import algonauts, hcp_movies
from state_of_mind.nbs_data.stimuli.raw_cache import NativeTimelineSpec


@pytest.mark.parametrize(
    ("dataset", "module", "tr_seconds", "spec_dataset", "require_text"),
    (
        ("algonauts", algonauts, 1.49, "algonauts", True),
        ("hcp_movies", hcp_movies, 1.0, "hcp", False),
    ),
)
def test_pipeline_timeline_specs_use_dataset_stimulus_policy(
    monkeypatch,
    dataset,
    module,
    tr_seconds,
    spec_dataset,
    require_text,
):
    source = SimpleNamespace(
        timeline_id=f"{dataset}/timeline",
        tr_seconds=tr_seconds,
        n_volumes=7,
    )
    calls = []
    original = module.timeline_spec

    def wrapped(timeline_id, **kwargs):
        calls.append((timeline_id, kwargs))
        return original(timeline_id, **kwargs)

    monkeypatch.setattr(module, "timeline_spec", wrapped)

    [spec] = pipeline._timeline_specs([source], dataset)

    assert isinstance(spec, NativeTimelineSpec)
    assert spec.dataset == spec_dataset
    assert spec.native_tr_seconds == tr_seconds
    assert spec.n_bins == 7
    assert spec.require_text is require_text
    assert calls == [
        (
            source.timeline_id,
            {"origin_seconds": 0.0, "n_bins": 7},
        )
    ]
