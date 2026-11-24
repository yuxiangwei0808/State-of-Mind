"""data.history_fill: a long-history window may hold missing frames in its oldest part, which enter as zero."""
import json

import numpy as np
import pytest

from state_of_mind.data.schaefer_v2_dataset import SchaeferWindowDataset
from state_of_mind.utils.config import DEFAULTS, validate_config
from copy import deepcopy


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    n = 40
    rng = np.random.default_rng(0)

    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        return {"path": name, "shape": list(value.shape), "dtype": str(value.dtype)}

    brain = (rng.normal(0, 1, (n, 1000)) + 5).astype(np.float32)  # no frame is near zero
    motion = np.ones(n, dtype=bool)
    motion[[3, 12]] = False           # motion-censored: one deep in the history, one later
    support = np.ones(n, dtype=bool)
    support[5:8] = False              # a rest: real brain frames, no movie
    valid = motion & support
    run = {"run_id": "r1", "subject_id": "s1", "timeline_id": "t1", "n_timepoints": n, "tr_seconds": 1.0,
           "time_origin_seconds": 0.0, "brain_views": {"projected_native": put("r1/brain.npy", brain)},
           "masks": {"valid_time": put("r1/valid.npy", valid), "stimulus_support": put("r1/support.npy", support),
                     "motion_qc": put("r1/motion.npy", motion), "nonsteady": put("r1/nonsteady.npy", np.zeros(n, dtype=bool))}}
    (root / "runs").mkdir(parents=True)
    (root / "runs/runs.jsonl").write_text(json.dumps(run) + "\n")
    (root / "features.jsonl").write_text("")
    (root / "bundle.json").write_text(json.dumps({"layout_id": "neuroworld-schaefer1000-v2", "n_parcels": 1000,
                                                  "dataset_id": "hcp_movies", "artifacts": {"features": {"path": "features.jsonl"}}}))
    return root, brain


def windows(root, fill, ranges=None):
    return SchaeferWindowDataset(root, ["r1"], tr_seconds=1.0, state_window_trs=10, num_states=4, include_stimulus=False,
                                 frame_ranges=ranges, history_fill_frames=fill)


def test_fill_keeps_windows_whose_recent_frames_are_valid_and_zeros_missing_history(bundle):
    root, brain = bundle
    strict, filled = windows(root, 0), windows(root, 7)
    # strict: every frame of the 13-frame window valid, so no window holds frames 3, 5-7 or 12
    assert [int(s) for _, s in strict.windows] == list(range(13, 28))
    # filled: frames start+7 .. start+12 valid (none of 3, 5-7, 12), so the first start is 6; those frames may be older
    starts = [int(s) for _, s in filled.windows]
    assert starts == list(range(6, 28))
    item = filled[0]["brain"].numpy()                                  # frames 6 .. 18
    np.testing.assert_array_equal(item[:2], brain[6:8])                # rest frames pass motion QC: kept
    assert (item[12 - 6] == 0).all()                                   # motion-censored history frame -> zero
    np.testing.assert_array_equal(item[7:], brain[13:19])              # the recent part is untouched
    assert (np.abs(np.delete(item, 12 - 6, axis=0)) > 0).all()


def test_fill_zeros_history_outside_the_split(bundle):
    root, brain = bundle
    ds = windows(root, 7, ranges={"r1": [(20, 40)]})
    starts = [int(s) for _, s in ds.windows]
    assert min(starts) == 13                                           # recent part from frame 20 on
    item = ds[0]["brain"].numpy()
    assert (item[:7] == 0).all()                                       # frames 13-19 lie outside the split
    np.testing.assert_array_equal(item[7:], brain[20:26])


def test_config_validates_history_fill():
    cfg = deepcopy(DEFAULTS)
    cfg["data"].update(bundle_path="x", train_runs=["a"], val_runs=["b"], history_fill="yes")
    with pytest.raises(ValueError, match="history_fill"):
        validate_config(cfg)
    cfg["data"]["history_fill"] = True
    with pytest.raises(ValueError, match="long-history"):
        validate_config(cfg)
