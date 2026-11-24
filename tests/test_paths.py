"""Environment-variable roots and legacy ${NEUROWORLD_*} placeholder compatibility."""
from state_of_mind import paths


def test_root_prefers_state_of_mind_and_falls_back_to_legacy(monkeypatch, tmp_path):
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"

    monkeypatch.delenv("STATE_OF_MIND_DATA", raising=False)
    monkeypatch.delenv("NEUROWORLD_DATA", raising=False)
    assert paths._root("STATE_OF_MIND_DATA", "NEUROWORLD_DATA", "data") == (paths.REPO / "data").resolve()

    monkeypatch.setenv("NEUROWORLD_DATA", str(old_dir))
    assert paths._root("STATE_OF_MIND_DATA", "NEUROWORLD_DATA", "data") == old_dir.resolve()

    monkeypatch.setenv("STATE_OF_MIND_DATA", str(new_dir))
    assert paths._root("STATE_OF_MIND_DATA", "NEUROWORLD_DATA", "data") == new_dir.resolve()


def test_expand_supports_canonical_and_legacy_placeholders():
    payload = {
        "new": ["${STATE_OF_MIND_DATA}/a", "${STATE_OF_MIND_OUTPUT}/b"],
        "legacy": ["${NEUROWORLD_DATA}/a", "${NEUROWORLD_OUTPUT}/b"],
    }
    expanded = paths.expand(payload)
    assert expanded["new"] == [f"{paths.DATA}/a", f"{paths.OUTPUT}/b"]
    assert expanded["legacy"] == expanded["new"]
