from pathlib import Path

import numpy as np
import pytest

from neuroworld.benchmark import grid as G, queries as Q
from neuroworld.benchmark.data import BenchmarkData, Run, segments
from neuroworld.benchmark.interface import Persistence, stimulus_source
from neuroworld.benchmark.metrics import high_pass_basis, step_scores
from neuroworld.benchmark.report import Units


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


def test_spliced_stimulus_rows_for_the_ridge():
    from neuroworld.benchmark.references import StimulusRows
    rows = StimulusRows.__new__(StimulusRows)
    rows.delays, rows.pad = [0, 2], 2
    rows.rows = {"a": np.concatenate([np.zeros((2, 1)), np.arange(10)[:, None] + 100.0]).astype(np.float32),
                 "b": np.concatenate([np.zeros((2, 1)), np.arange(10)[:, None] + 200.0]).astype(np.float32)}
    run = make_run(n=10)
    run.timeline_id = "a"
    batch = Q.batch("toy", 1.0, run, [5], 1, 3, splices=[("b", 1)])
    x = rows.lagged("a", np.array([5, 6, 7]), lambda f, m: stimulus_source(batch, 0, f, m))
    # Delay 0 reads b from frame 1 on; delay 2 still reads a before the splice (frames 3, 4).
    assert x[:, 0].tolist() == [201, 202, 203] and x[:, 1].tolist() == [103, 104, 201]


def test_per_modality_history_replaces_only_that_block_before_the_origin():
    from neuroworld.benchmark.references import StimulusRows
    rows = StimulusRows.__new__(StimulusRows)
    rows.delays, rows.pad = [2], 2
    a = np.stack([np.arange(10) + 100.0, np.arange(10) + 110.0, np.arange(10) + 120.0], 1)
    b = a + 100.0
    rows.rows = {k: np.concatenate([np.zeros((2, 3)), v]).astype(np.float32) for k, v in (("a", a), ("b", b))}
    rows.blocks = {"audio": slice(0, 1), "text": slice(1, 2), "video": slice(2, 3)}
    run = make_run(n=10)
    run.timeline_id = "a"
    batch = Q.batch("toy", 1.0, run, [5], 1, 3, histories=[{"text": ("b", 5)}])
    x = rows.lagged("a", np.array([5, 6, 7]), lambda f, m: stimulus_source(batch, 0, f, m))
    # Frames 3, 4 lie before the origin: only the text column comes from b; frame 5 is the query's own.
    assert x[:, 0].tolist() == [103, 104, 105]
    assert x[:, 1].tolist() == [213, 214, 115]
    assert x[:, 2].tolist() == [123, 124, 125]


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
    from neuroworld.benchmark.forecast import _map
    items = list(range(6))
    square = lambda offset, x: offset + x * x
    assert _map(square, items, (1,), workers=3) == _map(square, items, (1,), workers=1) == [1, 2, 5, 10, 17, 26]




def test_ipsid_filter_and_roll_matches_the_state_space_recursion():
    from neuroworld.benchmark.baselines import IPSIDBaseline
    rng = np.random.default_rng(0)
    nx, ny, nu = 3, 4, 2
    p = {"A": 0.5 * np.eye(nx), "B": rng.normal(size=(nx, nu)), "C": rng.normal(size=(ny, nx)), "D": np.zeros((ny, nu)),
         "K": 0.1 * rng.normal(size=(nx, ny)), "y_mean": np.zeros(ny), "u_mean": np.zeros(nu)}
    y_pre, u_pre, u_fut = rng.normal(size=(1, 5, ny)), rng.normal(size=(1, 5, nu)), rng.normal(size=(1, 3, nu))
    out, states, origin = IPSIDBaseline._filter_and_roll(p, y_pre, u_pre, u_fut)
    x = np.zeros(nx)
    for t in range(5):
        x = p["A"] @ x + p["B"] @ u_pre[0, t] + p["K"] @ (y_pre[0, t] - p["C"] @ x)
    assert np.allclose(origin[0], x)
    for k in range(3):
        assert np.allclose(out[0, k], p["C"] @ x) and np.allclose(states[0, k], x)
        x = p["A"] @ x + p["B"] @ u_fut[0, k]


def test_gain_is_the_slope_on_the_detrended_reference():
    from neuroworld.benchmark.gain import window_gains
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 20, 1000)).astype(np.float32)
    trend = np.linspace(0, 5, 20)[None, :, None]
    ok = np.ones((3, 20), dtype=bool)
    sets = [np.arange(100), np.arange(500, 600)]
    g = window_gains(2.0 * ref + 7.0 + trend, ref, ok, sets)
    assert np.allclose(window_gains(2.0 * ref, ref, ok, sets), 2.0 * np.ones((3, 2)), atol=0.05)
    assert np.all(np.isfinite(g))
    assert np.allclose(window_gains(np.ones_like(ref) * 3.0 + trend, ref, ok, sets), 0.0, atol=1e-6)  # persistence-like
    ok[0, :5] = False
    assert np.isnan(window_gains(ref, ref, ok, sets)[0]).all()  # fewer than 80% valid frames
    flat = np.ones_like(ref) + 1e-9 * ref  # no stimulus: a flat reference has no gain under a floor
    assert np.isnan(window_gains(ref, flat, np.ones((3, 20), bool), sets, floor=np.array([1e-6, 1e-6]))).all()


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
    from neuroworld.benchmark import state as S
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
    from neuroworld.benchmark.interface import QueryBatch
    batch = QueryBatch(dataset="toy", tr=1.0, run_ids=["r"], subject_ids=["s"], timeline_ids=["a"],
                       origins=np.array([10]), prefix=np.zeros((1, 0, 1000), np.float32),
                       prefix_valid=np.zeros((1, 0), bool), horizon=10, splices=[("b", 50, 3)])
    timelines, frames = stimulus_source(batch, 0, np.arange(8, 16))
    assert list(timelines) == ["a"] * 5 + ["b"] * 3 and frames.tolist() == [8, 9, 10, 11, 12, 50, 51, 52]


def test_updating_schedules_and_scoring_before_reveal():
    from neuroworld.benchmark import updating as U
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
    from neuroworld.benchmark import updating as U

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




def test_ipsid_skips_the_update_at_missing_frames():
    from neuroworld.benchmark.baselines import IPSIDBaseline
    rng = np.random.default_rng(1)
    p = {"A": 0.9 * np.eye(3), "B": rng.normal(size=(3, 2)), "C": rng.normal(size=(4, 3)), "D": np.zeros((4, 2)),
         "K": 0.3 * rng.normal(size=(3, 4)), "y_mean": np.zeros(4), "u_mean": np.zeros(2)}
    y, u, fut = rng.normal(size=(2, 6, 4)), rng.normal(size=(2, 6, 2)), rng.normal(size=(2, 3, 2))
    valid = np.ones((2, 6)); valid[:, 3:] = 0
    masked = IPSIDBaseline._filter_and_roll(p, y, u, fut, valid=valid)[0]
    y_other = y.copy(); y_other[:, 3:] += 100.0  # values at missing frames must not matter
    assert np.allclose(masked, IPSIDBaseline._filter_and_roll(p, y_other, u, fut, valid=valid)[0])


def test_state_gain_difference_compares_two_viewings():
    from neuroworld.benchmark import state as S
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(4, 20, 1000)).astype(np.float32)

    class Reference:
        floor, sets = None, {"stimulus-driven": np.arange(200)}

        def predict(self, subject, timeline, frames):
            return ref.reshape(-1, 1000)[: len(frames)]

    runs = [make_run(run_id="a"), make_run(run_id="b")]
    o = np.arange(4) * 20
    ok = np.ones((4, 20), dtype=bool)
    d = S._gain_difference(Reference(), [np.arange(200)], runs, (o, o), (2.0 * ref, ref), (ok, ok))
    assert np.allclose(d, 1.0, atol=1e-4)  # gain 2 in viewing a, 1 in viewing b


def test_state_gain_reads_both_viewings_against_one_reference():
    from neuroworld.benchmark import state as S
    ref = np.random.default_rng(1).normal(size=(4, 20, 1000)).astype(np.float32)

    class Reference:  # the two runs' features differ slightly in scale over the same clip
        floor = None

        def predict(self, subject, timeline, frames):
            return (1.1 if timeline.endswith("a") else 1.0) * ref.reshape(-1, 1000)[: len(frames)]

    from dataclasses import replace
    runs = [replace(make_run(run_id=r), timeline_id=f"t-{r}") for r in ("a", "b")]
    o, ok = np.arange(4) * 20, np.ones((4, 20), dtype=bool)
    args = (Reference(), [np.arange(200)], runs, (o, o), (ref, ref), (ok, ok))
    assert np.abs(S._gain_difference(*args)).min() > 0.05  # own references: the same response looks different
    assert np.allclose(S._gain_difference(*args, common=True), 0.0, atol=1e-6)


def test_state_gain_centering_removes_what_participants_share():
    from neuroworld.benchmark import state as S
    keys = [("run-2", "run-3", 0), ("run-2", "run-3", 10)]
    records = [{"unit": u, "keys": keys, "p": np.array([np.nan, np.nan]),
                **{f"d {w}": np.array([d, d + 1.0]) for w in S.GAIN_WINDOWS},
                **{f"q {w}": np.array([0.5, -0.2]) for w in S.GAIN_WINDOWS}}  # a stimulus-only model: the same q
               for u, d in (("p1", 0.1), ("p2", 0.4), ("p3", -0.2))]
    S._center_by_origin(records)
    assert all(np.allclose(r["q 30-60 s"], 0.0) for r in records)
    assert np.allclose(sum(r["d 30-60 s"] for r in records), 0.0)
    assert all(np.isnan(r["p"]).all() for r in records)


def test_state_gain_correlation_of_a_constant_prediction_is_marked():
    from neuroworld.benchmark import state as S
    from neuroworld.benchmark.report import Units
    sums = lambda a, y: np.array([a.size, a.sum(), y.sum(), (a * y).sum(), (a * a).sum(), (y * y).sum()])
    y = np.random.default_rng(2).normal(size=(3, 10))
    units = Units(["u0", "u1", "u2"])
    flat = S._correlation(np.stack([sums(np.zeros(10), y[i]) for i in range(3)]), units)
    assert flat["value"] is None and flat.get("constant")
    moving = S._correlation(np.stack([sums(y[i] + 0.1, y[i]) for i in range(3)]), units)
    assert moving["value"] > 0.99 and not moving.get("constant")
    empty = S._correlation(np.zeros((3, 6)), units)  # nothing scored: n/a, not constant
    assert empty["value"] is None and not empty.get("constant")


def test_state_paired_gain_is_zero_for_a_model_against_itself():
    from neuroworld.benchmark import state as S
    rng = np.random.default_rng(3)
    tr, horizon = 1.49, 40

    def unit():
        s = {"se_own": rng.random(horizon), "se_other": rng.random(horizon), "count": np.full(horizon, 5.0)}
        for b in S.BANDS:
            s.update({f"{b}|err": rng.random(1), f"{b}|energy": rng.random(1) + 1, f"{b}|corr": rng.random(6)})
        for w in S.GAIN_WINDOWS:
            a, y = rng.normal(size=8), rng.normal(size=8)
            s.update({f"gain {w}|corr": np.array([8, a.sum(), y.sum(), (a * y).sum(), (a * a).sum(), (y * y).sum()]),
                      f"gain {w}|prefix corr": np.zeros(6), f"gain {w}|err": rng.random(1), f"gain {w}|energy": rng.random(1) + 1})
        return s

    sums = {f"u{i}": unit() for i in range(4)}
    d = S.paired(sums, sums, tr)
    for w in S.GAIN_WINDOWS:
        assert abs(d[f"gain {w}"]["correlation"]["value"]) < 1e-12 and abs(d[f"gain {w}"]["skill"]["value"]) < 1e-12


def test_state_centering_removes_differences_every_participant_shares():
    from neuroworld.benchmark import state as S
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
    from neuroworld.benchmark import profile as P
    rows = {"a": {"forecast": {"value": 10}, "state": {"value": 55}, "updating": {"value": 20}},
            "b": {"forecast": {"value": 12}, "state": {"value": 55}, "updating": {"value": 21}},
            "c": {"forecast": {"value": 5}, "state": {"value": 60}, "updating": {"value": 10}},
            "d": {"forecast": {"value": 11}, "state": None, "updating": None}}
    assert P.pareto(rows) == {"a": False, "b": True, "c": True, "d": False}


def toy_adapter(rollout, frames=3):
    """A RolloutAdapter around a toy rollout(batch) -> (fMRI, states)."""
    from collections import OrderedDict
    from neuroworld.benchmark.adapters import RolloutAdapter
    adapter = RolloutAdapter.__new__(RolloutAdapter)
    adapter.frames, adapter._memory, adapter._memory_bytes, adapter._rollout = frames, OrderedDict(), 0, rollout
    adapter.name = "toy"
    return adapter


def test_world_model_gap_filling_does_not_depend_on_the_rest_of_the_batch():
    from dataclasses import replace
    from neuroworld.benchmark.interface import QueryBatch
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
    from neuroworld.benchmark.interface import QueryBatch
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
    from neuroworld.benchmark import adapters as A, updating as U

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
    from neuroworld.benchmark import profile as P
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
    from neuroworld.benchmark.metrics import step_scores
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
    from neuroworld.benchmark import state as S
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
    from neuroworld.benchmark import updating as U
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


def test_linear_forecaster_solves_each_steps_own_ridge_when_targets_are_missing():
    from neuroworld.benchmark.references import _accumulate, _moments, _solve
    rng = np.random.default_rng(9)
    h = rng.normal(size=(60, 5))
    r = rng.normal(size=(60, 3, 2))
    ok = np.ones((60, 3), dtype=bool)
    ok[50:, 2] = False  # the last anchors miss the step-3 target
    ok[[3, 17], 1] = False
    r = r * ok[..., None]
    g = _moments(5, 3, 2)
    _accumulate(g, h[:25], r[:25], ok[:25])  # two runs
    _accumulate(g, h[25:], r[25:], ok[25:])
    lam, V, VC, mean_h, mean_r = _solve(g)
    beta = 0.5
    for k in range(3):
        W = V[k] @ (VC[k] / (lam[k] + beta)[:, None])
        x, y = h[ok[:, k]], r[ok[:, k], k]
        xc, yc = x - x.mean(0), y - y.mean(0)
        direct = np.linalg.solve(xc.T @ xc + beta * np.eye(5), xc.T @ yc)
        assert np.allclose(W, direct, atol=1e-8)
        assert np.allclose(mean_h[k], x.mean(0)) and np.allclose(mean_r[k], y.mean(0))






def test_ipsid_fit_settings_change_with_the_stimulus_timing():
    from types import SimpleNamespace
    from neuroworld.benchmark.baselines import IPSIDBaseline
    model = IPSIDBaseline.__new__(IPSIDBaseline)
    model.features = "projections"
    settings = []
    for delay in (3.0, 6.0):
        cfg = {"data": {"bundle_path": "/b", "hrf_delay_seconds": delay, "train_runs": ["r"], "split_frames": {}}}
        model.data = SimpleNamespace(cfg=cfg, dataset="hcp_movies", grid=SimpleNamespace(tr=1.0))
        settings.append(model._settings())
    assert settings[0] != settings[1] and '"lag": 3' in settings[0] and '"lag": 6' in settings[1]


def test_latent_probe_moments_match_a_direct_ridge():
    from neuroworld.benchmark import latent as L
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
    from neuroworld.benchmark import profile as P
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


def test_calibration_reaches_the_model_behind_a_latent_row(monkeypatch):
    from types import SimpleNamespace
    from neuroworld.benchmark import adapt as A
    from neuroworld.benchmark.latent import LatentProbe
    calibrated = []
    monkeypatch.setattr(A, "adapt", lambda model, data, runs, log=None: calibrated.append(model))
    adapter = SimpleNamespace(state_frames=3)  # a model with participant heads
    row = LatentProbe(adapter, SimpleNamespace(provenance=None), "toy (latent)")
    data = SimpleNamespace(runs=lambda split: [])
    A.calibrate(row, data, "heldout-adapt", log=None)
    A.calibrate(row, data, "val", log=None)  # not a held-out split: nothing to calibrate
    assert calibrated == [adapter]


def test_latent_probe_ignores_coordinate_scales_and_constant_coordinates():
    from neuroworld.benchmark import latent as L
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


def test_kalman_batch_of_viewers_with_different_state_widths():
    """IPSID identifies one model per viewer, with its own state width: a batch mixing viewers pads latent outputs to the widest."""
    from neuroworld.benchmark.baselines import IPSIDBaseline

    class Batch:
        def __init__(self, subjects):
            self.subject_ids = list(subjects)

        def take(self, rows):
            return Batch([self.subject_ids[i] for i in rows])

        def __len__(self):
            return len(self.subject_ids)

    class Fake(IPSIDBaseline):
        def __init__(self):
            pass

        def _group(self, subject):
            return subject

        def _run(self, batch, use_input=True):
            if len(set(batch.subject_ids)) > 1:
                return IPSIDBaseline._run(self, batch, use_input)
            n, width = len(batch.subject_ids), 16 if batch.subject_ids[0] == "a" else 32
            return np.full((n, 3, 5), 1.0 if width == 16 else 2.0), np.full((n, 3, width), float(width))

    fmri, states = Fake()._run(Batch(["a", "b", "a"]))
    assert fmri.shape == (3, 3, 5) and states.shape == (3, 3, 32)
    assert (fmri[[0, 2]] == 1).all() and (fmri[1] == 2).all()
    assert (states[0, :, :16] == 16).all() and (states[0, :, 16:] == 0).all() and (states[1] == 32).all()
