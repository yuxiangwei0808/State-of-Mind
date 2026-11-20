from pathlib import Path
import numpy as np
import pytest
from neuroworld.data.bundle import (BundleValidationError, contained_path, atomic_write_npy,
                                    atomic_write_json, atomic_write_jsonl, read_jsonl)

def test_contained_path_rejects_traversal_absolute_and_symlink_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.npy"
    np.save(inside, np.zeros((1,), dtype=np.float32), allow_pickle=False)
    assert contained_path(root, "inside.npy", must_exist=True, kind="file") == inside

    with pytest.raises(BundleValidationError, match="contained"):
        contained_path(root, "../outside.npy")
    with pytest.raises(BundleValidationError, match="contained"):
        contained_path(root, tmp_path / "outside.npy")

    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "external.npy"
    np.save(external, np.zeros((1,), dtype=np.float32), allow_pickle=False)
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleValidationError, match="escapes"):
        contained_path(root, "link/external.npy", must_exist=True)


def test_atomic_array_and_json_writes_fail_closed(tmp_path: Path) -> None:
    array_path = tmp_path / "array.npy"
    source = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    atomic_write_npy(array_path, source, ndim=2)
    loaded = np.load(array_path, allow_pickle=False)
    assert loaded.dtype == np.float32
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, source.astype(np.float32))

    with pytest.raises(FileExistsError):
        atomic_write_npy(array_path, source)
    atomic_write_npy(array_path, np.ones((2, 4)), overwrite=True)
    assert np.load(array_path, allow_pickle=False).shape == (2, 4)

    json_path = tmp_path / "value.json"
    atomic_write_json(json_path, {"answer": 42})
    with pytest.raises(FileExistsError):
        atomic_write_json(json_path, {"answer": 43})
    with pytest.raises(BundleValidationError, match="strict JSON"):
        atomic_write_json(tmp_path / "nan.json", {"value": float("nan")})


def test_jsonl_validation_rejects_blank_or_missing_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    atomic_write_jsonl(path, [{"run_id": "a"}, {"run_id": "b"}])
    assert [record["run_id"] for record in read_jsonl(path, required_keys={"run_id"})] == [
        "a",
        "b",
    ]

    path.write_text('{"run_id": "a"}\n\n{"wrong": "b"}\n', encoding="utf-8")
    with pytest.raises(BundleValidationError, match="Blank JSONL"):
        read_jsonl(path, required_keys={"run_id"})
