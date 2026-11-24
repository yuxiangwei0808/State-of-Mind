"""Updating task: can a few new measurements keep the state on track?

After a fully revealed history (up to two minutes, the last 30 s complete), the model runs through the next minute
of the recorded movie with fMRI hidden. A schedule reveals B frames, evenly spaced. Between reveals the model
forecasts the next segment from everything before it: the history and the hidden period so far, with a mask that
marks only the revealed frames as valid. Episodes start every 30 s (each needs up to 60 sequential forecasts).
Each model decides how to bridge the gaps (the linear forecaster enters a missing frame as the training mean; a
latent world model rolls its state through them; a Kalman filter skips the update). With own_fill (--own-fill), every
frame the model is not shown is instead filled with its own earlier forecast of it, which is how the paper scores the
linear forecaster (models that already fill gaps with their own forecasts are unchanged). Every frame is scored before
it is revealed. Hiding recorded frames changes only what the model knows, so every schedule has a valid target.
  budgets   B = 0 (open loop), 2, 4, 8, 16, and every frame (one-step forecasting)
  metrics   nMSE over the minute (each parcel normalized by its measured variance in the run) and the pooled
            Pearson r over all scored frames, averaged over parcels; the headline is the skill with B = 4 (one
            measurement every ~15 s), 1 - the summed error over the run-mean reference's
Pooled per bootstrap unit (content parts on Algonauts, participants on HCP).

    python -m state_of_mind.benchmark updating --model run:<dir> --device cuda [--own-fill]
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from state_of_mind.benchmark import queries as Q
from state_of_mind.benchmark.forecast import predict_batch
from state_of_mind.benchmark.grid import LONGEST_PREFIX, frames_within
from state_of_mind.benchmark.metrics import pooled_correlation
from state_of_mind.benchmark.report import Units

HIDDEN_SECONDS = 60.0
STRIDE_SECONDS = 30.0
GAP_SECONDS = (20.0, 45.0)  # the "gap" schedule: 16 measurements, none in this window
BUDGETS = (0, 2, 4, 8, 16, "gap", "all")
RUNS = 16  # runs whose episodes are asked together (larger batches, fewer calls)


def _even(start, stop, count):
    return [start + int(round((j + 1) * (stop - start) / (count + 1))) - 1 for j in range(count)]


def gap_window(hidden):
    """(first, stop) steps of the gap: no measurement under the "gap" schedule."""
    return tuple(int(round(hidden * s / HIDDEN_SECONDS)) for s in GAP_SECONDS)


def schedule(hidden, budget):
    """Sorted 0-based positions (within the hidden period) of the revealed frames: B evenly spaced, every frame, or
    "gap" (16 measurements, 8 before and 8 after a 25-s stretch without any)."""
    if budget == "all":
        return list(range(hidden))
    if budget == "gap":
        a, b = gap_window(hidden)
        return sorted(set(_even(0, a, 8) + _even(b, hidden, 8)))
    return sorted(set(_even(0, hidden, int(budget))))


def origins(data, run):
    grid = data.grid
    prefix, hidden = grid.prefixes()[LONGEST_PREFIX], frames_within(HIDDEN_SECONDS, grid.tr)
    stride = frames_within(STRIDE_SECONDS, grid.tr)
    return np.array([t for t in range(prefix, run.n - hidden + 1, stride)
                     if run.prefix_complete(t, prefix) and run.target_span_ok(t, hidden)], dtype=np.int64)


def rollout(data, predict, episodes, budget, own_fill=False):
    """([n, hidden, ...] forecasts of the hidden period under one reveal schedule, [n, hidden, 1000] targets,
    [n, hidden] mask) for episodes = [(run, origins)] of one or more runs, asked together; predict(QueryBatch) gives
    the forecasts. own_fill: the frames not shown enter the history as the model's own forecast of them."""
    grid = data.grid
    prefix, hidden = grid.history_frames(), frames_within(HIDDEN_SECONDS, grid.tr)
    start = Q.joint_batch(data.dataset, grid.tr, episodes, prefix, hidden)
    y, ok = Q.joint_targets(episodes, hidden)
    n = len(start)
    history = np.concatenate([start.prefix, np.full((n, hidden, 1000), np.nan, np.float32)], axis=1)
    valid = np.concatenate([start.prefix_valid, np.zeros((n, hidden), bool)], axis=1)
    reveal = schedule(hidden, budget)
    forecast, s = None, 0
    for e in sorted(set(reveal) | {hidden - 1}):
        batch = replace(start, origins=start.origins + s, prefix=history[:, :prefix + s].copy(),
                        prefix_valid=valid[:, :prefix + s].copy(), horizon=e - s + 1)
        part = predict(batch)
        if forecast is None:
            forecast = np.zeros((n, hidden, part.shape[-1]), np.float32)
        forecast[:, s:e + 1] = part
        if own_fill:  # the model's own forecast stands in for the frames it is not shown
            history[:, prefix + s:prefix + e + 1] = part
            valid[:, prefix + s:prefix + e + 1] = True
        if e in reveal:  # scored above, now revealed
            if own_fill:  # an invalid measured frame keeps the forecast
                history[:, prefix + e] = np.where(ok[:, e, None], y[:, e], part[:, -1])
            else:  # an invalid measured frame stays missing
                history[:, prefix + e] = y[:, e]
                valid[:, prefix + e] = ok[:, e]
        s = e + 1
    return forecast, y, ok


def _add(s, forecast, y, ok, mean):
    """Adds one run's episodes to its unit's sums."""
    err = np.where(ok, ((forecast.astype(np.float64) - y) ** 2).sum(-1), 0.0)  # [origins, hidden]
    ref = np.where(ok, ((y.astype(np.float64) - mean) ** 2).sum(-1), 0.0)
    s["step se"] += err.sum(0)
    s["step ref"] += ref.sum(0)
    s["se"] += float(err.sum())
    s["ref"] += float(ref.sum())
    x, t = forecast[ok].astype(np.float64), y[ok].astype(np.float64)
    s["count"] += int(ok.sum())
    s["corr"] += np.stack([np.full(1000, len(x), float), x.sum(0), t.sum(0), (x * t).sum(0), (x * x).sum(0),
                           (t * t).sum(0)])


def collect(data, models, runs, log=None, own_fill=False):
    """{model name: {budget: {unit: sums}}} for models scored on the same episodes: squared error and the run-mean
    reference's squared error (in total and per step of the hidden period), frame count and per-parcel Pearson sums
    over the scored frames. The episodes of RUNS runs are asked together; the models one after the other, so two
    readouts of one latent model share its rollouts through the adapter's memory."""
    sums = {m.name: {str(b): {} for b in BUDGETS} for m in models}
    hidden = frames_within(HIDDEN_SECONDS, data.grid.tr)
    for first in range(0, len(runs), RUNS):
        group = runs[first:first + RUNS]
        episodes = Q.episodes(group, lambda run: origins(data, run))
        means = [Q.run_mean(run) for run, _ in episodes]
        for b in BUDGETS if episodes else ():
            for model in models:
                forecast, y, ok = rollout(data, lambda batch: predict_batch(model, batch), episodes, b, own_fill)
                for (run, rows), mean in zip(Q.spans(episodes), means):
                    s = sums[model.name][str(b)].setdefault(str(data.bootstrap_unit(run)), {
                        "se": 0.0, "ref": 0.0, "count": 0, "corr": np.zeros((6, 1000)), "step se": np.zeros(hidden),
                        "step ref": np.zeros(hidden)})
                    _add(s, forecast[rows], y[rows], ok[rows], mean)
        for run in group:
            run.__dict__.pop("brain", None)
        if log and (first + len(group)) // 50 > first // 50:
            log(f"updating {', '.join(m.name for m in models)}: {first + len(group)}/{len(runs)} runs")
    return sums


def summarize(sums):
    """Per budget: nMSE (squared error over the run-mean reference's, on the same frames), pooled r, and the nMSE in
    the gap window and after it (the "gap" schedule's 25 s without measurements, and the 15 s once they resume)."""
    out = {}
    for b, by_unit in sums.items():
        units = Units(list(by_unit))
        stack = lambda k: np.array([by_unit[u][k] for u in units.units], dtype=np.float64)
        cnt = stack("count")
        corr = np.stack([by_unit[u]["corr"] for u in units.units])
        r_samples = np.array([pooled_correlation(np.tensordot(w, corr, axes=1)) for w in units.counts[:500]])
        out[b] = {"nmse": Units.interval(*units.ratio(stack("se"), stack("ref"))),
                  "r": Units.interval(pooled_correlation(corr.sum(0)), r_samples), "frames": int(cnt.sum())}
        if "step se" in by_unit[units.units[0]]:
            se, ref = stack("step se"), stack("step ref")
            a, c = gap_window(se.shape[1])
            out[b]["windows"] = {name: Units.interval(*units.ratio(se[:, lo:hi].sum(1), ref[:, lo:hi].sum(1)))
                                 for name, (lo, hi) in (("gap", (a, c)), ("after gap", (c, se.shape[1])))}
    return out


def paired(sums_a, sums_b):
    """nMSE of model a minus model b per budget, on common units."""
    out = {}
    for b in sums_a:
        common = sorted(set(sums_a[b]) & set(sums_b[b]))
        units = Units(common)
        ra = units.ratio(np.array([sums_a[b][u]["se"] for u in common]), np.array([sums_a[b][u]["ref"] for u in common], float))
        rb = units.ratio(np.array([sums_b[b][u]["se"] for u in common]), np.array([sums_b[b][u]["ref"] for u in common], float))
        out[b] = Units.interval(ra[0] - rb[0], ra[1] - rb[1])
    return out


def save(folder, name, results, sums):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(results, indent=1) + "\n")
    arrays = {}
    for split, by_budget in sums.items():
        for b, by_unit in by_budget.items():
            for u, s in by_unit.items():
                arrays.update({f"{split}|{b}|{u}|{k}": np.asarray(v) for k, v in s.items()})
    np.savez_compressed(folder / f"{name}.npz", **arrays)


def load(folder, name):
    out = {}
    with np.load(Path(folder) / f"{name}.npz") as f:
        for key in f.files:
            split, b, u, k = key.split("|", 3)
            value = f[key]
            out.setdefault(split, {}).setdefault(b, {}).setdefault(u, {})[k] = value if value.ndim else value.item()
    return out


def run(data, model, splits=("val", "dev"), output="artifacts/benchmark", log=print, own_fill=False):
    """Score `model` (or a list of models, on the same episodes); returns the results (a list for a list)."""
    from state_of_mind.benchmark.grid import stamp
    models = model if isinstance(model, list) else [model]
    results = {m.name: {"model": m.name, "dataset": data.dataset, "hidden_seconds": HIDDEN_SECONDS, "stamp": stamp(m),
                        "splits": {}, **({"own fill": True} if own_fill else {})} for m in models}
    sums = {m.name: {} for m in models}
    for split in splits:
        runs = data.runs(split)
        by_model = collect(data, models, runs, log, own_fill)
        for m in models:
            sums[m.name][split] = by_model[m.name]
            budgets = summarize(by_model[m.name])
            results[m.name]["splits"][split] = {"exposure": runs[0].exposure if runs else None, "budgets": budgets}
            if log:
                log(f"updating {m.name} {split}: " + ", ".join(f"B={b} nMSE {v['nmse']['value']:.3f}"
                                                             for b, v in budgets.items()))
    for m in models:
        save(Path(output) / data.dataset / "updating", m.name, results[m.name], sums[m.name])
    out = [results[m.name] for m in models]
    return out if isinstance(model, list) else out[0]


def table(folder, reference):
    """Markdown: per split, nMSE by budget for each model and its paired difference from the reference."""
    from state_of_mind.benchmark.state import _fmt
    folder = Path(folder)
    names = sorted(p.stem for p in folder.glob("*.json"))
    results = {n: json.loads((folder / f"{n}.json").read_text()) for n in names}
    sums = {n: load(folder, n) for n in names}
    lines = []
    for split in sorted({s for r in results.values() for s in r["splits"]}):
        budgets = [str(b) for b in BUDGETS]
        lines += [f"**{split}** (nMSE; r in parentheses)", "", "| Model | " + " | ".join(f"B = {b}" for b in budgets) + " |",
                  "|---|" + "---|" * len(budgets)]
        for n in names:
            entry = results[n]["splits"].get(split)
            if entry:
                cells = [f"{entry['budgets'][b]['nmse']['value']:.3f} ({entry['budgets'][b]['r']['value']:.3f})" for b in budgets]
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
        if reference in sums and split in sums[reference]:
            for n in names:
                if n != reference and split in sums[n]:
                    d = paired(sums[n][split], sums[reference][split])
                    lines.append(f"| {n} − {reference} | " + " | ".join(_fmt(d[b], True) for b in budgets) + " |")
        lines.append("")
    return "\n".join(lines)
