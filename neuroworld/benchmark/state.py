"""State task: does the observed past improve the forecast of this particular future?

Two viewings of the same scene by the same participant share the future movie but not the brain state:
  Algonauts  the two viewings of each Movie10 part (development split; the same past movie)
  HCP        the viewings of the 84-s clip that ends every run (different preceding clips), in three groups:
             new participants' runs 2-3 (held out), their pairs with run 1 (calibration), training participants
At origins every 3 s inside the repeated material, each viewing is forecast for the next minute from its own past
and from the other viewing's past (its fMRI history and its stimulus history); the future movie is the same.
  difference predicted      the headline: the share of the measured difference between the viewings that the
                            predicted difference explains over the minute, 1 - sum (dy_hat - dy)^2 / sum dy^2 with
                            dy = viewing 1 - viewing 2 (0% = predicting no difference; below 0 = a wrong or
                            too large difference), per second of the horizon, and the seconds for which every
                            second's 95% interval stays above 0. On HCP both differences are first centered across
                            participants at each origin (_center_differences)
Details, per forecast step (normalized by each parcel's measured variance in the run):
  correct-past advantage    nMSE with the other viewing's past minus nMSE with the own past; 0 for a model that
                            ignores the past
  viewing-difference skill  1 - sum (dy_hat - dy)^2 / sum dy^2 with dy = viewing 1 - viewing 2, each forecast from
                            its own past; 0 for a model that predicts no difference
  gain readout              the same comparison for gain (gain.py: how strongly a series follows the movie), in the
                            0-30 s and 30-60 s windows: skill and correlation of the predicted with the measured
                            gain difference between the viewings, both viewings read against one reference (a
                            forecast's gain over the whole window, the measured gain over its valid frames); the
                            measured gain difference in the two 30-s prefixes is the model-free reference
                            (correlation). On HCP every pair is the same two runs, so gain differences are centered
                            across participants at each origin: a stimulus-only model then scores 0
All are pooled over pairs and origins per bootstrap unit (Movie10 parts, HCP participants).

    python -m neuroworld.benchmark state --model run:<dir> --device cuda
"""
from itertools import combinations
import json
import warnings
from pathlib import Path

import numpy as np

from neuroworld.benchmark import gain as G, queries as Q
from neuroworld.benchmark.forecast import predict_batch
from neuroworld.benchmark.grid import GRIDS, LONGEST_PREFIX, frames_within, window_steps
from neuroworld.benchmark.report import Units

HORIZON_SECONDS = 60.0
STRIDE_SECONDS = 3.0  # origins, as in every task
BANDS = {"B1": (0.0, 7.5), "B2": (7.5, 30.0), "B3": (30.0, 60.0)}
MODALITIES = ("video", "audio", "text")


def _with_run1(a, b):
    return "/run-01/" in a.run_id or "/run-01/" in b.run_id


# Friends has no repeated viewings, so Algonauts uses Movie10 (development). HCP: group -> (split, pair filter,
# exposure). Run 1 calibrates new participants' readout heads, so their pairs with run 1 are tagged.
GROUPS = {"algonauts": {"dev": ("dev", None, "seen in selection")},
          "hcp_movies": {"heldout-adapt": ("heldout", lambda a, b: not _with_run1(a, b), "unseen"),
                         "heldout-calibration": ("heldout", _with_run1, "adaptation"),
                         "train": ("train", None, "seen in fitting")}}
SPLITS = {dataset: tuple(groups) for dataset, groups in GROUPS.items()}


def _viewing(run):
    return 2 if run.timeline_id.endswith("_run-2") else 1


def pairs(data, runs, keep=None):
    """[(run_a, origins_a, run_b, origins_b)]: the same participant's viewings of the same material, with aligned
    origins every 3 s (frame k of a's list and of b's list are the same point of the movie); `keep(run_a, run_b)`
    filters HCP pairs."""
    tr, prefix = data.grid.tr, data.grid.prefixes()[LONGEST_PREFIX]
    horizon, stride = frames_within(HORIZON_SECONDS, tr), frames_within(STRIDE_SECONDS, tr)
    out = []
    if data.dataset == "hcp_movies":
        by_subject = {}
        for run in runs:
            for _, start, stop in data.repeated_segments(run):
                by_subject.setdefault(run.subject_id, []).append((run, start, stop))
        for views in by_subject.values():
            for (ra, sa, ea), (rb, sb, eb) in combinations(sorted(views, key=lambda v: v[0].run_id), 2):
                if keep is not None and not keep(ra, rb):
                    continue
                offsets = [k for k in range(0, min(ea - sa, eb - sb) - horizon + 1, stride)
                           if ra.prefix_complete(sa + k, prefix) and rb.prefix_complete(sb + k, prefix)]
                if offsets:
                    out.append((ra, np.array([sa + k for k in offsets]), rb, np.array([sb + k for k in offsets])))
        return out
    groups = {}
    for run in runs:
        groups.setdefault((run.subject_id, run.content), {})[_viewing(run)] = run
    for views in groups.values():
        if set(views) != {1, 2}:
            continue
        ra, rb = views[1], views[2]
        n = min(ra.n, rb.n)
        o = np.array([t for t in range(prefix, n - horizon + 1, stride)
                      if ra.prefix_complete(t, prefix) and rb.prefix_complete(t, prefix)], dtype=np.int64)
        if len(o):
            out.append((ra, o, rb, o.copy()))
    return out


def _queries(data, run, origins, other=None, other_origins=None):
    """Forecast queries for `run` at `origins`, from its own past or from `other`'s past at `other_origins` (the
    full history: up to two minutes)."""
    tr, prefix = data.grid.tr, data.grid.history_frames()
    horizon = frames_within(HORIZON_SECONDS, tr)
    if other is None:
        return Q.batch(data.dataset, tr, run, origins, prefix, horizon)
    histories = [{m: (other.stimulus_id, int(t)) for m in MODALITIES + ("all",)} for t in other_origins]
    batch = Q.batch(data.dataset, tr, run, origins, prefix, horizon, histories=histories)
    donor = Q.batch(data.dataset, tr, other, other_origins, prefix, horizon)
    batch.prefix, batch.prefix_valid = donor.prefix, donor.prefix_valid
    return batch


_scale = Q.run_variance  # [1000] the nMSE normalizer of every task


def _band_steps(tr, horizon):
    return {b: np.arange(s.start - 1, s.stop - 1) for b, (lo, hi) in BANDS.items()
            for s in [window_steps(lo, hi, tr, horizon)]}


GAIN_WINDOWS = ("0-30 s", "30-60 s")


def _gain_difference(reference, sets, runs, origins, series, oks, offset=0, common=False):
    """[n] gain of series a minus series b over their common valid frames, each against its own run's reference or,
    with common=True (the same movie on the same frames), both against the mean of the two runs' references."""
    frames = series[0].shape[1]
    ok = oks[0] & oks[1]
    refs = [reference.predict(run.subject_id, run.timeline_id, (o[:, None] + offset + np.arange(frames)).reshape(-1))
            .reshape(len(o), frames, -1) for run, o in zip(runs, origins)]
    if common:
        refs = [0.5 * (refs[0] + refs[1])] * 2
    a, b = (G.window_gains(y, ref, ok, sets, reference.floor)[:, 0] for y, ref in zip(series, refs))
    return a - b


def _center_by_origin(records):
    """Subtract, per (runs, origin), the mean over participants of each gain difference (in place); NaNs ignored."""
    groups = {}
    for r in records:
        for i, key in enumerate(r["keys"]):
            groups.setdefault(key, []).append((r, i))
    for members in groups.values():
        for name in ("p", *(f"{x} {w}" for w in GAIN_WINDOWS for x in ("d", "q"))):
            values = np.array([r[name][i] for r, i in members])
            if np.isfinite(values).any():
                mean = np.nanmean(values)
                for r, i in members:
                    r[name][i] -= mean


def _add_gain(s, r):
    """Add one pair's gain differences to its unit's sums."""
    for w in GAIN_WINDOWS:
        d, q = r[f"d {w}"], r[f"q {w}"]
        for key, x in (("corr", q), ("prefix corr", r["p"])):
            m = np.isfinite(d) & np.isfinite(x)
            a, y = x[m], d[m]
            s[f"gain {w}|{key}"] += np.array([a.size, a.sum(), y.sum(), (a * y).sum(), (a * a).sum(), (y * y).sum()])
        m = np.isfinite(d) & np.isfinite(q)
        s[f"gain {w}|err"] += ((q[m] - d[m]) ** 2).sum()
        s[f"gain {w}|energy"] += (d[m] ** 2).sum()


def _difference(dp, dy, both):
    """Viewing-difference sums per step over trials (pairs and origins) and parcels: squared error of the predicted
    difference and energy of the measured one, at the valid steps."""
    err = np.where(both, ((dp - dy) ** 2).sum(-1), 0.0).sum(0)
    energy = np.where(both, (dy ** 2).sum(-1), 0.0).sum(0)
    return {"difference step|err": err, "difference step|energy": energy}


def _center_differences(records):
    """Subtract, per (runs, origin) and step, the mean over participants of the raw predicted and measured viewing
    differences (in place): what every participant shares (the movie, the session order) is not scored. The mean is
    taken in float64, so a prediction that is the same for every participant becomes exactly 0 (a tie)."""
    groups = {}
    for r in records:
        for i, key in enumerate(r["keys"]):
            groups.setdefault(key, []).append((r, i))
    for members in groups.values():
        for name in ("dp", "dy"):
            with warnings.catch_warnings():  # a step no member could score stays missing
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(np.stack([r[name][i] for r, i in members]).astype(np.float64), 0)
            for r, i in members:
                r[name][i] = (r[name][i].astype(np.float64) - mean).astype(r[name].dtype)


def collect(data, model, runs, reference=None, log=None, keep=None):
    """{unit: sums} over all pairs: the difference predicted (per step), per-step squared errors (own and other
    past), per-band difference sums and, with a gain reference, the gain readout's sums per window. For a list of
    models: {model name: that}, the models one after the other on each pair (two readouts of one latent model then
    share its rollouts through the adapter's memory)."""
    several = isinstance(model, list)
    models = model if several else [model]
    tr = data.grid.tr
    horizon = frames_within(HORIZON_SECONDS, tr)
    bands = _band_steps(tr, horizon)
    window, prefix = frames_within(G.WINDOW_SECONDS, tr), data.grid.prefixes()[LONGEST_PREFIX]
    sets = [reference.sets["stimulus-driven"]] if reference is not None else None
    sums, records = {m.name: {} for m in models}, {m.name: [] for m in models}
    center = data.dataset == "hcp_movies"  # every pair is the same two runs (see _center_differences)
    items = pairs(data, runs, keep)
    for n, (ra, oa, rb, ob) in enumerate(items):
        unit = str(data.bootstrap_unit(ra))
        for model in models:
            s = sums[model.name].setdefault(unit, {
                "se_own": np.zeros(horizon), "se_other": np.zeros(horizon), "count": np.zeros(horizon),
                **{f"{b}|{k}": np.zeros(6 if k == "corr" else 1) for b in bands for k in ("err", "energy", "corr")},
                **{f"gain {w}|{k}": np.zeros(6 if "corr" in k else 1)
                   for w in GAIN_WINDOWS for k in ("err", "energy", "corr", "prefix corr")},
                "difference step|err": np.zeros(horizon), "difference step|energy": np.zeros(horizon)})
            forecast = {("a", "own"): predict_batch(model, _queries(data, ra, oa)),
                        ("a", "other"): predict_batch(model, _queries(data, ra, oa, rb, ob)),
                        ("b", "own"): predict_batch(model, _queries(data, rb, ob)),
                        ("b", "other"): predict_batch(model, _queries(data, rb, ob, ra, oa))}
            measured = {"a": Q.targets(ra, oa, horizon), "b": Q.targets(rb, ob, horizon)}
            scale = {"a": _scale(ra), "b": _scale(rb)}
            for v in ("a", "b"):
                y, ok = measured[v]
                for past in ("own", "other"):
                    err = (((forecast[(v, past)] - y) ** 2) / scale[v]).mean(-1)  # [origins, horizon]
                    s[f"se_{past}"] += np.where(ok, err, 0.0).sum(0)
                s["count"] += ok.sum(0)
            both = measured["a"][1] & measured["b"][1]
            # Raw viewing differences: every scored frame and parcel weighs the same.
            dy = measured["a"][0] - measured["b"][0]
            dp = forecast[("a", "own")] - forecast[("b", "own")]
            for b, steps in bands.items():
                m = both[:, steps]
                x, y = dp[:, steps][m], dy[:, steps][m]
                s[f"{b}|err"] += ((x - y) ** 2).sum()
                s[f"{b}|energy"] += (y ** 2).sum()
                s[f"{b}|corr"] += np.array([x.size, x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])
            r = {"unit": unit, "keys": [(ra.timeline_id, rb.timeline_id, int(o)) for o in oa]}
            if center:  # centered across participants, below
                raw = {"dp": forecast[("a", "own")] - forecast[("b", "own")], "dy": measured["a"][0] - measured["b"][0]}
                r.update({k: np.where(both[..., None], v, np.nan).astype(np.float32) for k, v in raw.items()},
                         both=both)
            else:
                for k, v in _difference(dp, dy, both).items():
                    s[k] += v
            if reference is not None:
                before = {"a": Q.batch(data.dataset, tr, ra, oa, prefix, 0),
                          "b": Q.batch(data.dataset, tr, rb, ob, prefix, 0)}
                supported = [r.support[o[:, None] + np.arange(-prefix, 0)] for r, o in ((ra, oa), (rb, ob))]
                # The prefixes are different material on HCP, so each is read against its own run's reference.
                r["p"] = _gain_difference(reference, sets, (ra, rb), (oa, ob),
                                          (np.nan_to_num(before["a"].prefix), np.nan_to_num(before["b"].prefix)),
                                          (before["a"].prefix_valid & supported[0],
                                           before["b"].prefix_valid & supported[1]),
                                          offset=-prefix)
                for k, w in enumerate(GAIN_WINDOWS):
                    span = slice(k * window, (k + 1) * window)
                    measured_ok = (measured["a"][1][:, span], measured["b"][1][:, span])
                    every = np.ones_like(measured_ok[0])  # a forecast exists at every frame
                    for x, ya, yb, oks in (("d", measured["a"][0], measured["b"][0], measured_ok),
                                           ("q", forecast[("a", "own")], forecast[("b", "own")], (every, every))):
                        r[f"{x} {w}"] = _gain_difference(reference, sets, (ra, rb), (oa + k * window, ob + k * window),
                                                         (ya[:, span], yb[:, span]), oks, common=True)
            records[model.name].append(r)
        for run in (ra, rb):
            run.__dict__.pop("brain", None)
        if log and (n + 1) % 25 == 0:
            log(f"state {', '.join(m.name for m in models)}: {n + 1}/{len(items)} pairs")
    for name, recs in records.items():
        if center:  # every pair is the same two runs: keep only what is specific to the person
            _center_differences(recs)
            if reference is not None:
                _center_by_origin(recs)
        for r in recs:
            if center:
                dp, dy = (np.nan_to_num(r[k]) for k in ("dp", "dy"))
                for k, v in _difference(dp, dy, r["both"]).items():
                    sums[name][r["unit"]][k] += v
            if reference is not None:
                _add_gain(sums[name][r["unit"]], r)
    return sums if several else sums[models[0].name]


def _pearson(s):
    n, a, b, ab, aa, bb = s.T if s.ndim == 2 else s
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ab - a * b / n) / np.sqrt((aa - a * a / n) * (bb - b * b / n))


def summarize(sums, tr, units=None):
    """Advantage per step and per band, viewing-difference skill and correlation per band, with 95% intervals."""
    units = units or Units(list(sums))
    idx = [u for u in units.units]
    stack = lambda key: np.stack([sums[u][key] for u in idx])
    horizon = len(next(iter(sums.values()))["count"])
    bands = _band_steps(tr, horizon)
    adv, cnt = stack("se_other") - stack("se_own"), stack("count")
    out = {"units": len(idx), "pairs_steps": int(cnt.sum()), "advantage_by_step": [], "bands": {}}
    if "difference step|err" in sums[idx[0]]:
        out["difference"] = difference_summary({u: sums[u] for u in idx}, tr, units)
    for h in range(horizon):
        out["advantage_by_step"].append(Units.interval(*units.ratio(adv[:, h], cnt[:, h])))
    for b, steps in bands.items():
        entry = {"advantage": Units.interval(*units.ratio(adv[:, steps].sum(1), cnt[:, steps].sum(1)))}
        err, energy = stack(f"{b}|err")[:, 0], stack(f"{b}|energy")[:, 0]
        ratio, samples = units.ratio(err, energy)
        entry["difference skill"] = Units.interval(1 - ratio, 1 - samples)
        corr = stack(f"{b}|corr")
        estimate = _pearson(corr.sum(0))
        entry["difference correlation"] = (Units.interval(estimate, _pearson(units.counts @ corr))
                                           if np.isfinite(estimate) and corr[:, 4].sum() > 0 else
                                           {"value": None, "low": None, "high": None})
        out["bands"][b] = entry
    if f"gain {GAIN_WINDOWS[0]}|corr" in sums[idx[0]]:
        out["gain"] = {}
        for w in GAIN_WINDOWS:
            err, energy = stack(f"gain {w}|err")[:, 0], stack(f"gain {w}|energy")[:, 0]
            ratio, samples = units.ratio(err, energy)
            out["gain"][w] = {"skill": Units.interval(1 - ratio, 1 - samples),
                              "correlation": _correlation(stack(f"gain {w}|corr"), units),
                              "measured prefix correlation": _correlation(stack(f"gain {w}|prefix corr"), units)}
    return out


def difference_summary(sums, tr, units=None):
    """The share of the viewing difference predicted over the minute and per step (95% intervals), and the seconds
    ahead for which every step's interval stays above 0."""
    units = units or Units(list(sums))
    stack = lambda key: np.stack([sums[u][key] for u in units.units])
    err, energy = stack("difference step|err"), stack("difference step|energy")
    ratio, samples = units.ratio(err.sum(1), energy.sum(1))
    steps = []
    for k in range(err.shape[1]):
        r, s = units.ratio(err[:, k], energy[:, k])
        steps.append(Units.interval(1 - r, 1 - s))
    helps = 0
    for cell in steps:
        if cell["low"] is None or cell["low"] <= 0:
            break
        helps += 1
    return {"skill": Units.interval(1 - ratio, 1 - samples), "by_step": steps, "seconds_helping": helps * tr}


def _correlation(corr, units):
    """Pooled Pearson from per-unit sums; undefined, and marked constant, when the prediction does not vary."""
    estimate = _pearson(corr.sum(0))
    if corr[:, 0].sum() == 0:  # nothing to score (the HCP prefix reference)
        return {"value": None, "low": None, "high": None}
    if corr[:, 4].sum() - corr[:, 1].sum() ** 2 / corr[:, 0].sum() <= 1e-12:
        return {"value": None, "low": None, "high": None, "constant": True}
    if not np.isfinite(estimate):
        return {"value": None, "low": None, "high": None}
    return Units.interval(estimate, _pearson(units.counts @ corr))


def paired(sums_a, sums_b, tr):
    """Model a minus model b on their common units: band advantage and difference skill, and the gain readout's
    correlation and skill per window (n/a when either prediction is constant)."""
    common = sorted(set(sums_a) & set(sums_b))
    units = Units(common)
    horizon = len(sums_a[common[0]]["count"])
    out = {}
    for b, steps in _band_steps(tr, horizon).items():
        values = {}
        for name, sums in (("a", sums_a), ("b", sums_b)):
            adv = np.stack([sums[u]["se_other"][steps].sum() - sums[u]["se_own"][steps].sum() for u in units.units])
            cnt = np.stack([sums[u]["count"][steps].sum() for u in units.units])
            err = np.stack([sums[u][f"{b}|err"][0] for u in units.units])
            energy = np.stack([sums[u][f"{b}|energy"][0] for u in units.units])
            values[name] = (units.ratio(adv, cnt), units.ratio(err, energy))
        (ea, sa), (eb, sb) = values["a"][0], values["b"][0]
        (ra, qa), (rb, qb) = values["a"][1], values["b"][1]
        out[b] = {"advantage": Units.interval(ea - eb, sa - sb), "difference skill": Units.interval(rb - ra, qb - qa)}
    if "difference step|err" in sums_a[common[0]] and "difference step|err" in sums_b[common[0]]:
        (ea, sa), (eb, sb) = (units.ratio(np.stack([s[u]["difference step|err"].sum() for u in units.units]),
                                          np.stack([s[u]["difference step|energy"].sum() for u in units.units]))
                              for s in (sums_a, sums_b))
        out["difference"] = Units.interval(eb - ea, sb - sa)  # skill of a minus skill of b
    if f"gain {GAIN_WINDOWS[0]}|corr" in sums_a[common[0]]:
        for w in GAIN_WINDOWS:
            stack = {k: {n: np.stack([s[u][f"gain {w}|{k}"] for u in units.units]) for n, s in (("a", sums_a), ("b", sums_b))}
                     for k in ("corr", "err", "energy")}
            corr = {n: _correlation(c, units) for n, c in stack["corr"].items()}
            if corr["a"]["value"] is None or corr["b"]["value"] is None:
                r = {"value": None, "low": None, "high": None}
            else:
                pa, pb = (_pearson(units.counts @ stack["corr"][n]) for n in ("a", "b"))
                r = Units.interval(corr["a"]["value"] - corr["b"]["value"], pa - pb)
            (ra, qa), (rb, qb) = (units.ratio(stack["err"][n][:, 0], stack["energy"][n][:, 0]) for n in ("a", "b"))
            out[f"gain {w}"] = {"correlation": r, "skill": Units.interval(rb - ra, qb - qa)}
    return out


def save(folder, name, results, sums):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(results, indent=1) + "\n")
    np.savez_compressed(folder / f"{name}.npz", **{f"{split}|{u}|{k}": v for split, by_unit in sums.items()
                                                    for u, s in by_unit.items() for k, v in s.items()})


def load(folder, name):
    """{split: {unit: {key: array}}}"""
    out = {}
    with np.load(Path(folder) / f"{name}.npz") as f:
        for key in f.files:
            split, unit, k = key.split("|", 2)
            out.setdefault(split, {}).setdefault(unit, {})[k] = f[key]
    return out


def run(data, model, splits=None, output="artifacts/benchmark", log=print):
    """Score `model` (or a list of models, on the same pairs); returns the results (a list for a list)."""
    from neuroworld.benchmark.adapt import calibrate
    from neuroworld.benchmark.grid import stamp
    splits = splits or SPLITS[data.dataset]
    models = model if isinstance(model, list) else [model]
    results = {m.name: {"model": m.name, "dataset": data.dataset, "horizon_seconds": HORIZON_SECONDS,
                        "stamp": stamp(m), "splits": {}} for m in models}
    sums = {m.name: {} for m in models}
    reference = G.Reference(data, output, log)
    adapted = False
    for split in splits:
        source, keep, exposure = GROUPS[data.dataset][split]
        if source == "heldout" and not adapted:
            for m in models:
                calibrate(m, data, "heldout-adapt", log=log)
            adapted = True
        by_model = collect(data, models, data.runs(source), reference, log, keep)
        for m in models:
            sums[m.name][split] = by_model[m.name]
            if not by_model[m.name]:
                continue
            entry = {"exposure": exposure, **summarize(by_model[m.name], data.grid.tr)}
            results[m.name]["splits"][split] = entry
            if log:
                bands = "; ".join(f"{b} difference skill {v['difference skill']['value']:.4f}"
                                  for b, v in entry["bands"].items())
                log(f"state {m.name} {split}: difference predicted {100 * entry['difference']['skill']['value']:.1f}%, "
                    f"helps for {entry['difference']['seconds_helping']:.1f} s; {bands}")
    for m in models:
        save(Path(output) / data.dataset / "state", m.name, results[m.name], sums[m.name])
    out = [results[m.name] for m in models]
    return out if isinstance(model, list) else out[0]


def _fmt(cell, signed=False):
    if cell is None or cell.get("value") is None:
        return "—"
    f = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
    return f(cell["value"]) + (f" [{f(cell['low'])}, {f(cell['high'])}]" if cell.get("low") is not None else "")


def table(folder, reference):
    """Markdown: per split and model, band advantage and difference skill, and the paired difference from the
    reference (the linear forecaster)."""
    folder = Path(folder)
    names = sorted(p.stem for p in folder.glob("*.json"))
    results = {n: json.loads((folder / f"{n}.json").read_text()) for n in names}
    sums = {n: load(folder, n) for n in names}
    lines = []
    for split in sorted({s for r in results.values() for s in r["splits"]}):
        lines += [f"**{split}**", "", "| Model | " + " | ".join(f"{b} advantage | {b} difference skill" for b in BANDS) + " |",
                  "|---|" + "---|" * (2 * len(BANDS))]
        for n in names:
            entry = results[n]["splits"].get(split)
            if entry:
                lines.append(f"| {n} | " + " | ".join(f"{_fmt(entry['bands'][b]['advantage'])} | "
                                                       f"{_fmt(entry['bands'][b]['difference skill'])}" for b in BANDS) + " |")
        if reference in sums and split in sums[reference]:
            for n in names:
                if n != reference and split in sums[n]:
                    d = paired(sums[n][split], sums[reference][split], GRIDS[results[n]["dataset"]].tr)
                    lines.append(f"| {n} − {reference} | " + " | ".join(
                        f"{_fmt(d[b]['advantage'], True)} | {_fmt(d[b]['difference skill'], True)}" for b in BANDS) + " |")
        lines.append("")
        gains = [n for n in names if "gain" in results[n]["splits"].get(split, {})]
        if gains:  # the secondary gain readout
            cell = lambda c, signed=False: "n/a (constant)" if c.get("constant") else _fmt(c, signed)
            lines += [f"**{split}, gain readout** (secondary)", "",
                      "| Model | " + " | ".join(f"r {w} | skill {w}" for w in GAIN_WINDOWS) + " |",
                      "|---|" + "---|" * (2 * len(GAIN_WINDOWS))]
            for n in gains:
                g = results[n]["splits"][split]["gain"]
                lines.append(f"| {n} | " + " | ".join(f"{cell(g[w]['correlation'])} | {cell(g[w]['skill'])}"
                                                       for w in GAIN_WINDOWS) + " |")
            prefix = results[gains[0]]["splits"][split]["gain"]
            lines.append("| measured prefix (no model) | " + " | ".join(
                f"{cell(prefix[w]['measured prefix correlation'])} | —" for w in GAIN_WINDOWS) + " |")
            if reference in gains:
                for n in gains:
                    if n != reference:
                        d = paired(sums[n][split], sums[reference][split], GRIDS[results[n]["dataset"]].tr)
                        lines.append(f"| {n} − {reference} | " + " | ".join(
                            f"{_fmt(d[f'gain {w}']['correlation'], True)} | {_fmt(d[f'gain {w}']['skill'], True)}"
                            for w in GAIN_WINDOWS) + " |")
            lines.append("")
    return "\n".join(lines)
