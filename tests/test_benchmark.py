from pathlib import Path

import numpy as np
import pytest

from state_of_mind.benchmark import grid as G, queries as Q
from state_of_mind.benchmark.data import BenchmarkData, Run, segments
from state_of_mind.benchmark.interface import Persistence, stimulus_source
from state_of_mind.benchmark.metrics import high_pass_basis, step_scores
from state_of_mind.benchmark.report import Units


def make_run(n=200, invalid=(), gap=None, allowed=None, run_id="r1", subject="s1", parcels=1000):
    brain_ok = np.ones(n, dtype=bool)
    brain_ok[list(invalid)] = False
    support = np.ones(n, dtype=bool)
    if gap:
        support[gap[0]:gap[1]] = False
    mask = np.ones(n, dtype=bool) if allowed is None else allowed
    run = Run(run_id=run_id, subject_id=subject, timeline_id="t", split="val", content="c", group="c", n=n,
              brain_path=Path("unused"), brain_ok=brain_ok, support=support, allowed=mask)
    # Each frame holds its own index, so any leak of a future frame is visible.
    run.__dict__["brain"] = np.repeat(np.arange(n, dtype=np.float32)[:, None], parcels, axis=1)
    return run


def steps(band, tr):
    s = G.steps_in(band, tr)
    return s.start, s.stop - 1


def test_band_steps_match_the_specification():
    assert [steps(b, 1.49) for b in ("B1", "B2", "B3")] == [(1, 5), (6, 20), (21, 40)]
    assert [steps(b, 1.0) for b in ("B1", "B2", "B3")] == [(1, 7), (8, 30), (31, 60)]
    assert [steps(b, 1.5) for b in ("B1", "B2", "B3")] == [(1, 5), (6, 20), (21, 40)]
    assert [steps(b, 1.75) for b in ("B1", "B2", "B3")] == [(1, 4), (5, 17), (18, 34)]
    assert G.GRIDS["algonauts"].prefixes() == {"P30": 20}
    assert G.GRIDS["hcp_movies"].prefixes() == {"P30": 30}
    assert G.GRIDS["algonauts"].history_frames() == 80 and G.GRIDS["hcp_movies"].history_frames() == 120
    assert G.GRIDS["algonauts"].horizon_steps == 40 and G.GRIDS["hcp_movies"].horizon_steps == 60  # the next minute


def test_band_truncation_rule():
    hcp = G.DatasetGrid("hcp", 1.0)
    assert "B3" in hcp.bands(56)  # 4 s short of 60 s: within 15% of the 30-s width
    assert "B3" not in hcp.bands(50)
    assert set(G.GRIDS["algonauts"].bands()) == {"B1", "B2", "B3"}


def test_forecast_origins_need_complete_prefix_and_one_supported_segment():
    grid = G.DatasetGrid("toy", 1.0, max_horizon_s=20, stride_s=1)
    run = make_run(n=120, invalid=[40], gap=(80, 85))
    origins = Q.forecast_origins(run, grid)
    assert len(origins)
    for t0 in origins:
        assert run.brain_ok[t0 - 30:t0].all()  # complete 30-s prefix
        assert run.support[t0:t0 + 20].all() and t0 + 20 <= run.n
    assert not any(t0 - 30 <= 40 < t0 for t0 in origins)
    assert not any(t0 < 85 and t0 + 20 > 80 for t0 in origins)


def test_targets_restricted_to_split_frames():
    allowed = np.zeros(150, dtype=bool)
    allowed[60:140] = True
    run = make_run(n=150, allowed=allowed)
    grid = G.DatasetGrid("toy", 1.0, max_horizon_s=20, stride_s=1)
    origins = Q.forecast_origins(run, grid)
    assert origins.min() >= 60 and origins.max() + 20 <= 140
    # The prefix may come from frames outside the split.
    assert origins.min() == 60


def test_batches_never_contain_the_origin_or_later_frames():
    run = make_run(n=100, invalid=[50])
    batch = Q.batch("toy", 1.0, run, [40, 60], 5, horizon=3)
    assert batch.prefix.shape == (2, 5, 1000)
    assert np.array_equal(batch.prefix[:, :, 0], [[35, 36, 37, 38, 39], [55, 56, 57, 58, 59]])
    late = Q.batch("toy", 1.0, run, [55], 10, horizon=3)
    assert not late.prefix_valid[0, 5]  # frame 50 is invalid
    assert np.isnan(late.prefix[0, 5]).all()


def test_targets_mask_invalid_frames_and_segment_end():
    run = make_run(n=100, invalid=[45])
    observed, ok = Q.targets(run, [40], 10, stop=[48])
    assert observed[0, :, 0].tolist() == list(range(40, 50))
    assert ok[0].tolist() == [True] * 5 + [False] + [True] * 2 + [False] * 2


def test_persistence_repeats_last_valid_frame():
    run = make_run(n=100, invalid=[59])
    batch = Q.batch("toy", 1.0, run, [60], 5, horizon=4)
    assert np.all(Persistence().predict(batch) == 58)
    empty = Q.batch("toy", 1.0, run, [60], 0, horizon=4)
    assert np.all(Persistence().predict(empty) == 0)


def test_step_scores_match_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4, 30)).astype(np.float32)
    y = (x + rng.normal(size=x.shape)).astype(np.float32)
    ok = np.ones((80, 4), dtype=bool)
    ok[3, 2] = False
    origins = np.arange(80) * 3 + 30
    out = step_scores(x, y, ok, origins, 1.0)
    rows = ok[:, 2]
    direct = np.mean([np.corrcoef(x[rows, 2, p], y[rows, 2, p])[0, 1] for p in range(30)])
    assert np.isclose(out["pearson"][2], direct)
    basis = high_pass_basis(origins[rows] + 2, 1.0)
    q, _ = np.linalg.qr(basis)
    xr = x[rows, 2] - q @ (q.T @ x[rows, 2])
    yr = y[rows, 2] - q @ (q.T @ y[rows, 2])
    assert np.isclose(out["pearson_hp"][2], np.mean([np.corrcoef(xr[:, p], yr[:, p])[0, 1] for p in range(30)]))
    assert out["rows"].tolist() == [80, 80, 79, 80]




def test_unit_bootstrap_ratio_and_paired_zero():
    units = Units(["a", "b", "c"], draws=200)
    num, den = units.collect([("a", 1.0, 1.0), ("b", 3.0, 1.0), ("c", 2.0, 2.0)])
    estimate, samples = units.ratio(num, den)
    assert np.isclose(estimate, 6 / 4) and samples.shape == (200,)
    diff = Units.interval(estimate - estimate, samples - samples)
    assert diff["value"] == 0 and diff["low"] == 0 and diff["high"] == 0


class _Data:
    """Minimal BenchmarkData stand-in."""
    dataset = "toy"

    def __init__(self, grid):
        self.grid = grid

    def repeated_segments(self, run):
        return []

    def bootstrap_unit(self, run):
        return run.run_id


def test_stimulus_source_switches_at_origin():
    run = make_run(n=100)
    batch = Q.batch("toy", 1.0, run, [40, 50], 5, 10, splices=[("other", 7), None])
    timelines, frames = stimulus_source(batch, 0, np.array([38, 39, 40, 45]))
    assert timelines.tolist() == ["t", "t", "other", "other"] and frames.tolist() == [38, 39, 7, 12]
    timelines, frames = stimulus_source(batch, 1, np.array([55]))
    assert timelines.tolist() == ["t"] and frames.tolist() == [55]


def test_test_splits_are_scored_like_any_other(monkeypatch):
    """No split is locked: the test split and new participants' test runs (heldout-test) are ordinary test data."""
    data = BenchmarkData.__new__(BenchmarkData)
    data.cfg = {"data": {"train_runs": ["*/run-0[1-3]/x"], "val_runs": [], "dev_runs": [], "test_runs": ["*/run-04/x"],
                         "heldout_subjects": ["b"], "split_frames": {}}}
    data.records = {f"{s}/run-0{i}/x": {"run_id": f"{s}/run-0{i}/x", "subject_id": s} for s in "ab" for i in range(1, 5)}
    monkeypatch.setattr(BenchmarkData, "_run", lambda self, record, split: record["run_id"])
    assert data.runs("test") == ["a/run-04/x"]
    assert data.runs("heldout-test") == ["b/run-04/x"]
    assert data.runs("heldout-fit") == ["b/run-01/x"]
    assert sorted(data.runs("heldout-adapt")) == ["b/run-02/x", "b/run-03/x"]


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_targets_and_batches_accept_no_origins():
    run = make_run(n=50)
    observed, ok = Q.targets(run, np.array([], dtype=np.int64), 10)
    assert observed.shape == (0, 10, 1000) and ok.shape == (0, 10)


def test_parallel_map_matches_serial():
    from state_of_mind.benchmark.forecast import _map
    items = list(range(6))
    square = lambda offset, x: offset + x * x
    assert _map(square, items, (1,), workers=3) == _map(square, items, (1,), workers=1) == [1, 2, 5, 10, 17, 26]




class _RepeatData:
    """Two viewings of one Movie10-like part by one participant."""
    dataset = "toy"

    def __init__(self):
        self.grid = G.DatasetGrid("toy", 1.0)

    def bootstrap_unit(self, run):
        return run.content


class _PrefixLevel:
    """Forecasts the mean of the last prefix frame for every step (uses the past, ignores the stimulus)."""
    name = "prefix level"

    def predict(self, batch):
        last = np.nan_to_num(batch.prefix[:, -1])
        return np.repeat(last[:, None], batch.horizon, axis=1).astype(np.float32)


class _IgnoresPast:
    name = "ignores past"

    def predict(self, batch):
        return np.zeros((len(batch), batch.horizon, 1000), dtype=np.float32)


def test_state_pairs_and_scores():
    from state_of_mind.benchmark import state as S
    data = _RepeatData()

    def viewings():  # fresh runs each time: scoring releases each run's brain array
        rng = np.random.default_rng(0)
        runs = []
        for view in ("", "_run-2"):
            run = make_run(n=200, run_id=f"s1{view}", subject="s1")
            run.timeline_id, run.content = f"s1:movie10:life01{view}", "movie10:life01"
            offset = rng.normal(size=(1, 1000)).astype(np.float32)  # a viewing-specific level the past reveals
            run.__dict__["brain"] = (offset + rng.normal(size=(200, 1000))).astype(np.float32)
            runs.append(run)
        return runs

    pairs = S.pairs(data, viewings())
    assert len(pairs) == 1 and np.array_equal(pairs[0][1], pairs[0][3])
    blind = S.summarize(S.collect(data, _IgnoresPast(), viewings()), 1.0)
    assert all(abs(v["advantage"]["value"]) < 1e-12 and abs(v["difference skill"]["value"]) < 1e-12
               for v in blind["bands"].values())
    level = S.summarize(S.collect(data, _PrefixLevel(), viewings()), 1.0)
    assert level["bands"]["B2"]["advantage"]["value"] > 0  # the own past predicts the own level better
    assert level["bands"]["B2"]["difference skill"]["value"] > 0
    # Several models scored on the same pairs (a latent model's two readouts) give each one's own sums.
    together = S.collect(data, [_IgnoresPast(), _PrefixLevel()], viewings())
    for model in (_IgnoresPast(), _PrefixLevel()):
        alone = S.collect(data, model, viewings())
        assert together[model.name].keys() == alone.keys()
        for unit in alone:
            for k in alone[unit]:
                np.testing.assert_array_equal(together[model.name][unit][k], alone[unit][k])


def test_spliced_stimulus_can_start_after_the_origin():
    from state_of_mind.benchmark.interface import QueryBatch
    batch = QueryBatch(dataset="toy", tr=1.0, run_ids=["r"], subject_ids=["s"], timeline_ids=["a"],
                       origins=np.array([10]), prefix=np.zeros((1, 0, 1000), np.float32),
                       prefix_valid=np.zeros((1, 0), bool), horizon=10, splices=[("b", 50, 3)])
    timelines, frames = stimulus_source(batch, 0, np.arange(8, 16))
    assert list(timelines) == ["a"] * 5 + ["b"] * 3 and frames.tolist() == [8, 9, 10, 11, 12, 50, 51, 52]


def test_updating_schedules_and_scoring_before_reveal():
    from state_of_mind.benchmark import updating as U
    assert U.schedule(40, 0) == [] and U.schedule(40, "all") == list(range(40))
    assert U.schedule(40, 4) == [7, 15, 23, 31] and len(U.schedule(40, 16)) == 16

    class Data:
        dataset, grid = "toy", G.DatasetGrid("toy", 1.0)

    class LastFrame:
        """Repeats the newest input frame: perfect on a constant brain once that level is revealed."""
        name = "last frame"

        def predict(self, batch):
            last = np.nan_to_num(batch.prefix[:, -1])
            return np.repeat(last[:, None], batch.horizon, axis=1).astype(np.float32)

    run = make_run(n=200)
    run.__dict__["brain"] = np.r_[np.zeros((100, 1000)), np.ones((100, 1000))].astype(np.float32)
    o = np.array([100])  # the level jumps exactly at the origin, so the prefix says nothing about it
    closed, y, ok = U.rollout(Data(), LastFrame().predict, [(run, o)], 0)
    assert np.allclose(closed, 0)  # open loop never learns the new level
    revealed, _, _ = U.rollout(Data(), LastFrame().predict, [(run, o)], "all")
    assert np.allclose(revealed[0, 0], 0) and np.allclose(revealed[0, 1:], 1)  # frame 0 scored before its reveal


def test_updating_own_fill_feeds_the_forecast_back_for_frames_not_shown():
    from state_of_mind.benchmark import updating as U

    class Data:
        dataset, grid = "toy", G.DatasetGrid("toy", 1.0)

    class CountsMissing:
        """Predicts, for every step, the number of invalid frames in its input window (as a constant map)."""
        name = "counts missing"

        def predict(self, batch):
            missing = (~batch.prefix_valid).sum(1).astype(np.float32)
            return np.repeat(missing[:, None, None], batch.horizon, axis=1) * np.ones((1, 1, 1000), np.float32)

    run = make_run(n=200)
    run.__dict__["brain"] = np.zeros((200, 1000), np.float32)
    o = np.array([100])
    plain, _, _ = U.rollout(Data(), CountsMissing().predict, [(run, o)], 4)
    own, _, _ = U.rollout(Data(), CountsMissing().predict, [(run, o)], 4, own_fill=True)
    first = U.schedule(plain.shape[1], 4)[0]
    assert np.allclose(plain[0, :first + 1], own[0, :first + 1])  # identical until the first reveal
    assert plain[0, -1, 0] > 0 and np.allclose(own[0, first + 1:], plain[0, 0, 0])  # own fill: no frame stays missing




def test_state_centering_removes_differences_every_participant_shares():
    from state_of_mind.benchmark import state as S
    shared = np.random.default_rng(5).normal(size=(2, 4, 10)).astype(np.float32)
    records = [{"keys": [("run-1", "run-2", 0), ("run-1", "run-2", 3)], "dp": shared.copy(),
                "dy": shared + np.float32(k)} for k in range(3)]
    S._center_differences(records)
    assert all(np.allclose(r["dp"], 0.0) for r in records)  # a stimulus-only prediction: the same for everyone
    assert np.allclose(sum(r["dy"] for r in records), 0.0, atol=1e-5)


def test_queries_mark_history_before_the_run_start_invalid():
    run = make_run(n=50)
    b = Q.batch("toy", 1.0, run, np.array([5, 30]), 20, 3)
    assert not b.prefix_valid[0, :15].any() and b.prefix_valid[0, 15:].all()
    assert np.isnan(b.prefix[0, :15]).all() and np.allclose(b.prefix[0, 15:, 0], np.arange(5))
    assert b.prefix_valid[1].all() and np.allclose(b.prefix[1, :, 0], np.arange(10, 30))


def test_pareto_marks():
    from state_of_mind.benchmark import profile as P
    rows = {"a": {"forecast": {"value": 10}, "state": {"value": 55}, "updating": {"value": 20}},
            "b": {"forecast": {"value": 12}, "state": {"value": 55}, "updating": {"value": 21}},
            "c": {"forecast": {"value": 5}, "state": {"value": 60}, "updating": {"value": 10}},
            "d": {"forecast": {"value": 11}, "state": None, "updating": None}}
    assert P.pareto(rows) == {"a": False, "b": True, "c": True, "d": False}


def toy_adapter(rollout, frames=3):
    """A RolloutAdapter around a toy rollout(batch) -> (fMRI, states)."""
    from collections import OrderedDict
    from state_of_mind.benchmark.adapters import RolloutAdapter
    adapter = RolloutAdapter.__new__(RolloutAdapter)
    adapter.frames, adapter._memory, adapter._memory_bytes, adapter._rollout = frames, OrderedDict(), 0, rollout
    adapter.name = "toy"
    return adapter


def test_world_model_gap_filling_does_not_depend_on_the_rest_of_the_batch():
    from dataclasses import replace
    from state_of_mind.benchmark.interface import QueryBatch
    adapter = toy_adapter(lambda part: (np.repeat(part.prefix[:, -1:], part.horizon, axis=1), None))  # persistence
    prefix = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
    valid = np.ones((2, 8), dtype=bool)
    valid[:, 4:] = False
    valid[0, 5] = True  # a revealed frame: valid in query A, failed QC in query B
    batch = QueryBatch(dataset="toy", tr=1.0, run_ids=["r", "r"], subject_ids=["s", "s"], timeline_ids=["a", "a"],
                       origins=np.array([20, 30]), prefix=prefix, prefix_valid=valid, horizon=5)
    alone = replace(batch, run_ids=["r"], subject_ids=["s"], timeline_ids=["a"], origins=batch.origins[:1],
                    prefix=prefix[:1], prefix_valid=valid[:1])
    together, single = adapter._fill_gaps(batch), adapter._fill_gaps(alone)
    np.testing.assert_array_equal(together.prefix[0, -4:], single.prefix[0, -4:])
    np.testing.assert_array_equal(single.prefix[0, -2], prefix[0, 5])  # the gap after the measurement starts from it


def test_world_model_gap_filling_handles_a_prefix_shorter_than_the_model_reads():
    from state_of_mind.benchmark.interface import QueryBatch
    adapter = toy_adapter(lambda part: (np.repeat(part.prefix[:, -1:], part.horizon, axis=1), None), frames=12)
    prefix = np.arange(8 * 2, dtype=np.float32).reshape(1, 8, 2)
    valid = np.ones((1, 8), dtype=bool)
    valid[:, [5, 6]] = False  # the model reads 12 frames; the prefix holds 8, two of them not revealed
    batch = QueryBatch(dataset="toy", tr=1.0, run_ids=["r"], subject_ids=["s"], timeline_ids=["a"],
                       origins=np.array([20]), prefix=prefix, prefix_valid=valid, horizon=5)
    filled = adapter._fill_gaps(batch)
    assert filled.prefix_valid.all()
    np.testing.assert_array_equal(filled.prefix[0, 5], prefix[0, 4])  # filled from the frames before the gap
    np.testing.assert_array_equal(filled.prefix[0, [0, 4, 7]], prefix[0, [0, 4, 7]])
    # Frames before a run start have nothing to be filled from: they stay unmeasured (zero), the later gap is filled.
    valid[:, :2] = False
    filled = adapter._fill_gaps(batch)
    np.testing.assert_array_equal(filled.prefix_valid[0], [False, False, True, True, True, True, True, True])
    np.testing.assert_array_equal(filled.prefix[0, :2], 0)
    np.testing.assert_array_equal(filled.prefix[0, 5], prefix[0, 4])


def test_world_model_updating_reuses_rollouts_and_batches_runs_without_changing_forecasts(monkeypatch):
    from state_of_mind.benchmark import adapters as A, updating as U

    class Data:
        dataset, grid = "toy", G.DatasetGrid("toy", 1.0)

    def adapter():
        def rollout(part):  # causal: step k depends only on the frames read, the origin and k
            model.steps += part.horizon * len(part)
            k = np.arange(1, part.horizon + 1)[None, :, None]
            fmri = model._window(part).mean(1)[:, None] * 0.8 ** k + 0.001 * part.origins[:, None, None] + 0.01 * k
            return fmri.astype(np.float32), fmri[..., :4].astype(np.float32)
        model = toy_adapter(rollout)
        model.steps = 0
        return model

    runs = [make_run(n=300, invalid=(170,), run_id="r1", subject="s1"), make_run(n=260, run_id="r2", subject="s2")]
    episodes = [(run, U.origins(Data(), run)) for run in runs]
    assert all(len(o) > 1 for _, o in episodes)
    for budget in (16, "gap", "all"):
        remembering = adapter()
        together = U.rollout(Data(), remembering.predict, episodes, budget)[0]
        separate = np.concatenate([U.rollout(Data(), remembering.predict, [e], budget)[0] for e in episodes])
        with monkeypatch.context() as m:
            m.setattr(A, "MEMORY_BYTES", 0)  # nothing kept: every gap rolled out again at every reveal
            plain = adapter()
            alone = U.rollout(Data(), plain.predict, episodes, budget)[0]
        np.testing.assert_array_equal(together, alone)
        np.testing.assert_array_equal(separate, alone)
        if budget != "all":
            assert remembering.steps * 3 < plain.steps


def test_card_curve_steps_and_paired_differences():
    from state_of_mind.benchmark import profile as P
    # The first curve point is the first step on both grids (1.49 s, 1 s), never a later one.
    assert P._steps("algonauts", (1.5, 6.0, 60.0)) == [0, 3, 39]
    assert P._steps("hcp_movies", (1.5, 6.0, 60.0)) == [0, 5, 59]
    # Skill a = 1 - 2/10 = 80%, skill b = 1 - 5/10 = 50%: a - b = +30 points; identical units give a zero-width interval.
    a = {"u1": (1.0, 5.0), "u2": (1.0, 5.0)}
    b = {"u1": (2.5, 5.0), "u2": (2.5, 5.0)}
    cell = P._paired(a, b)
    assert cell["value"] == pytest.approx(30.0) and cell["low"] == pytest.approx(30.0) and cell["high"] == pytest.approx(30.0)
    assert P._paired(a, {"other": (1.0, 1.0)}) is None
    # Several code versions of one protocol are listed without a warning; several protocols warn.
    one = {("2026-10-23", "abc"): 3, ("2026-10-23", "def"): 1}
    assert "Warning" not in P._stamp_line(one, "Core results")
    assert "Warning" in P._stamp_line({("2026-10-23", "abc"): 3, ("2026-10-01", "abc"): 1}, "Core results")




def test_forecast_skill_is_relative_to_the_run_mean_on_the_same_targets():
    from state_of_mind.benchmark.metrics import step_scores
    rng = np.random.default_rng(7)
    y = rng.normal(size=(30, 4, 6)) + 2.0
    mean = np.full(6, 2.0)
    ok = np.ones((30, 4), dtype=bool)
    average = step_scores(np.broadcast_to(mean, y.shape).copy(), y, ok, np.arange(30) * 3, 1.0, mean=mean)
    assert np.allclose(average["err"], average["ref"])  # predicting the run's average: skill exactly 0
    perfect = step_scores(y, y, ok, np.arange(30) * 3, 1.0, mean=mean)
    assert np.allclose(perfect["err"], 0.0) and (perfect["ref"] > 0).all()
    assert np.isnan(step_scores(y, y, ok, np.arange(30) * 3, 1.0)["err"]).all()
    two = step_scores(y[:2] + 1.0, y[:2], ok[:2], np.arange(2) * 3, 1.0, mean=mean)  # fewer than min_rows origins
    assert np.allclose(two["err"], 2 * 6) and np.isnan(two["nmse"]).all()  # squared error kept, nmse needs more


def test_state_difference_predicted_and_seconds_it_helps():
    from state_of_mind.benchmark import state as S
    rng = np.random.default_rng(8)
    dy = rng.normal(size=(40, 6, 20))
    dp = np.zeros_like(dy)
    dp[:, :2] = dy[:, :2]  # predicts the first two steps exactly, nothing after
    both = np.ones((40, 6), dtype=bool)
    sums = {f"u{i}": S._difference(dp[i * 10:(i + 1) * 10], dy[i * 10:(i + 1) * 10], both[i * 10:(i + 1) * 10])
            for i in range(4)}
    out = S.difference_summary(sums, tr=1.5)
    assert abs(out["skill"]["value"] - 2 / 6) < 0.05  # two of six equally noisy steps explained
    assert out["seconds_helping"] == 3.0  # two steps of 1.5 s
    assert np.allclose([c["value"] for c in out["by_step"][2:]], 0.0)  # predicting no difference: 0



def test_updating_gap_schedule_keeps_the_budget_and_leaves_a_gap():
    from state_of_mind.benchmark import updating as U
    hidden = 40
    a, b = U.gap_window(hidden)
    gap = U.schedule(hidden, "gap")
    assert len(gap) == 16 and not any(a <= k < b for k in gap) and max(gap) < hidden
    assert U.schedule(hidden, 16) == sorted({int(round((j + 1) * hidden / 17)) - 1 for j in range(16)})


def test_queries_hide_the_run_and_share_a_stimulus_handle_across_viewings():
    from dataclasses import replace
    base = make_run(n=60, run_id="sub-01/movie10/figures01/run-2")
    run = replace(base, timeline_id="s1:figures01_run-2", stimulus="s1:figures01")
    run.__dict__["brain"] = base.brain
    b = Q.batch("toy", 1.0, run, np.array([40]), 20, 3)
    assert b.timeline_ids == ["s1:figures01"]
    assert "run" not in b.run_ids[0] and "figures" not in b.run_ids[0]


def test_latent_probe_moments_match_a_direct_ridge():
    from state_of_mind.benchmark import latent as L
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 6)) * np.array([1, 10, 0.1, 3, 1, 5]) + 2.0
    y = z @ rng.normal(size=(6, 4)) + rng.normal(size=(300, 4))
    m, held = L._moments(), L._moments()
    L._add(m, z[:200], y[:200])
    L._add(held, z[200:], y[200:])
    # The ridge on standardized inputs, solved directly and mapped back to the original coordinates.
    zc, yc = z[:200] - z[:200].mean(0), y[:200] - y[:200].mean(0)
    sd = zc.std(0)
    zs = zc / sd
    lam = 200 * 0.1  # the standardized Gram matrix's mean eigenvalue is the sample count
    W = np.linalg.solve(zs.T @ zs + lam * np.eye(6), zs.T @ yc) / sd[:, None]
    b = y[:200].mean(0) - z[:200].mean(0) @ W
    (W2, b2), = L._solutions(m, [0.1])
    np.testing.assert_allclose(W2, W, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(b2, b, rtol=1e-8, atol=1e-10)
    # The held-out error from moments equals the direct error.
    assert L._error(held, W2, b2) == pytest.approx(((z[200:] @ W + b - y[200:]) ** 2).sum(), rel=1e-8)


def test_card_flags_a_row_that_mixes_probes(tmp_path, monkeypatch):
    import json
    from state_of_mind.benchmark import profile as P
    monkeypatch.setattr(P, "ROOT", tmp_path)

    def result(path, probe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stamp": {"readout": {"weights sha256": probe}} if probe else {}}))

    result(tmp_path / "algonauts" / "val" / "m (latent)" / "results.json", "aaa")
    result(tmp_path / "algonauts" / "updating" / "m (latent).json", "aaa")
    assert P._probe_mix("algonauts", "m (latent)") is None
    result(tmp_path / "algonauts" / "dev" / "m (latent)" / "results.json", "bbb")
    assert P._probe_mix("algonauts", "m (latent)") == {"aaa", "bbb"}
    result(tmp_path / "algonauts" / "dev" / "m (latent)" / "results.json", None)  # scored before probes were stamped
    assert P._probe_mix("algonauts", "m (latent)") == {"aaa", None}
    result(tmp_path / "algonauts" / "val" / "plain" / "results.json", None)  # no probe at all: nothing to flag
    assert P._probe_mix("algonauts", "plain") is None


def test_latent_probe_ignores_coordinate_scales_and_constant_coordinates():
    from state_of_mind.benchmark import latent as L
    rng = np.random.default_rng(1)
    z = rng.normal(size=(400, 5))
    y = z[:, :2] @ rng.normal(size=(2, 3)) + rng.normal(size=(400, 3))
    folds = np.arange(400) % 4

    def fit(x):
        total, parts = L._moments(), [L._moments() for _ in range(4)]
        for f in range(4):
            L._add(total, x[folds == f], y[folds == f])
            L._add(parts[f], x[folds == f], y[folds == f])
        alpha, _ = L._select([(total, parts)])
        (W, b), = L._solutions(total, [alpha])
        return alpha, x @ W + b

    alpha, plain = fit(z)
    # Rescaling coordinates (an irrelevant one by 10^4) or adding a constant one changes neither the penalty nor
    # the predictions.
    rescaled = np.c_[z * np.array([1e-3, 1.0, 1e4, 10.0, 1e2]), np.full(400, 5.0)]
    alpha2, scaled = fit(rescaled)
    assert alpha2 == alpha
    np.testing.assert_allclose(scaled, plain, rtol=1e-6, atol=1e-8)


