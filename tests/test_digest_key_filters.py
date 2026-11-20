from __future__ import annotations

import neuroworld.nbs_data.bundle as bundle
import neuroworld.nbs_data.fmri.atlas as atlas
import neuroworld.nbs_data.pipeline as pipeline
import neuroworld.nbs_data.stimuli.timing as timing


_RETIRED_KEYS = (
    "hash",
    "hashes",
    "sha256",
    "configSHA256",
    "sha256sum",
    "md5sum",
    "checksum",
    "digests",
)


def test_timing_scrubs_digest_spellings_but_preserves_shape() -> None:
    metadata = {
        "shape": [2, 3],
        "configSHA256": "old",
        "nested": {"md5sum": "old", "shape": "kept"},
    }

    assert timing._without_digest_fields(metadata) == {
        "shape": [2, 3],
        "nested": {"shape": "kept"},
    }
    assert timing._is_digest_key("shape") is False


def test_atlas_digest_filter_accepts_only_the_root_capability_name() -> None:
    assert atlas._digest_key("shape") is False
    assert all(atlas._digest_key(key) for key in _RETIRED_KEYS)
    assert atlas._digest_key("hashes_computed") is True


def test_bundle_digest_filter_handles_camel_case_and_sum_spellings() -> None:
    assert bundle._is_retired_hash_key("shape") is False
    assert all(bundle._is_retired_hash_key(key) for key in _RETIRED_KEYS)
    assert bundle._without_hash_fields(
        {"shape": [2, 3], "configSHA256": "old", "nested": {"md5sum": "old"}}
    ) == {"shape": [2, 3], "nested": {}}


def test_pipeline_digest_filter_handles_camel_case_and_sum_spellings() -> None:
    assert pipeline._is_retired_hash_key("shape") is False
    assert all(pipeline._is_retired_hash_key(key) for key in _RETIRED_KEYS)
    assert pipeline._without_hash_fields(
        {"shape": [2, 3], "configSHA256": "old", "nested": {"md5sum": "old"}}
    ) == {"shape": [2, 3], "nested": {}}
