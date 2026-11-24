"""History-length study: how much of the supplied fMRI history do models use, and does any of them gain beyond
30 s or beyond the benchmark's two-minute allowance?

Every model is scored on the benchmark's own origins and targets (Forecast, State and Updating on the held-out
panels), with each query's history cut to its last L seconds, L in SECONDS; queries carry 240 s.
  linear   the direct linear forecaster refitted for each L, not masked. Up to 30 s it reads every frame (at 30 s it
           is the benchmark baseline); beyond, the last 30 s frame by frame and the older history as 15-s averages,
           each standardized on training anchors. Training anchors need the last min(L, 30 s) complete, like
           evaluation origins; older missing frames enter as the training mean. L = 0 is its stimulus-only ridge.
  linear-matched  the same at 7.5 and 15 s, but with the 30-s model's spatial components per frame (102 on Algonauts,
           68 on HCP; the baseline gives shorter windows more, up to 256, within one feature budget)
  ipsid    the benchmark's Kalman filter over the last L s (a filter takes any length; L = 0 runs it open loop)
  mean     each parcel's average over the last L s, at every step (how much a longer history tells about the level)
Forecast is also scored on late origins, with at least 240 s of the run before them, where every L gets its full
history. Results are written in the benchmark's formats under OUTPUT/<dataset>/.

    python -m state_of_mind.experiments.history_length run --dataset algonauts --model linear --seconds 60
    python -m state_of_mind.experiments.history_length run --dataset algonauts --model linear --seconds 30 120 \
        --features context --benchmark-root <benchmark store> --output <history store>
                                                    (the linear arm on other stimulus rows, named "linear [context] L s")
    python -m state_of_mind.experiments.history_length report
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from state_of_mind.benchmark import forecast as F, grid as G, queries as Q, state as S, updating as U
from state_of_mind.benchmark.grid import GRIDS, frames_within
from state_of_mind.benchmark.references import FIT_STRIDE, HISTORY_WIDTH, LinearForecaster, SpatialBasis, StimulusRidge
from state_of_mind.benchmark.report import Units

OUTPUT = Path("artifacts/history-length")
BENCHMARK = Path("artifacts/benchmark")  # stimulus-row caches, read only
CONFIGS = {"algonauts": "state_of_mind/configs/algonauts/base.yaml", "hcp_movies": "state_of_mind/configs/hcp_movies/base.yaml"}
SECONDS = (0, 7.5, 15, 30, 60, 120, 240)
LONGEST, RECENT, BLOCK = 240.0, 30.0, 15.0
PANELS = {"algonauts": [("Algonauts: new content (Friends S6 and Movie10)", ("val", "dev"), ("dev",))],
          "hcp_movies": [("HCP: new content (held-out clips)", ("val",), ()),
                         ("HCP: new participants (runs 2-3)", ("heldout-adapt",), ("heldout-adapt",))]}
BUDGETS = (0, 4, 16, "gap")


class Window:
    """A model whose queries are cut to their last `seconds` of history."""

    def __init__(self, model, seconds, name):
        self.model, self.seconds, self.name = model, seconds, name

    def predict(self, batch):
        keep, width = frames_within(self.seconds, batch.tr), batch.prefix.shape[1]
        if keep < width:
            batch = replace(batch, prefix=batch.prefix[:, width - keep:], prefix_valid=batch.prefix_valid[:, width - keep:])
        return self.model.predict(batch)


class WindowLinear(LinearForecaster):
    """The direct linear forecaster on an L-s window (module docstring)."""

    def __init__(self, data, stimulus, seconds, basis, log=print, matched=False):
        tr = data.grid.tr
        self.frames = frames_within(seconds, tr)
        self.recent, self.block = min(self.frames, frames_within(RECENT, tr)), frames_within(BLOCK, tr)
        # Feature width of the recent frames: the baseline's budget, or the 30-s model's components per frame.
        self.width = HISTORY_WIDTH // frames_within(RECENT, tr) * self.recent if matched else HISTORY_WIDTH
        self.blocks = (self.frames - self.recent) // self.block
        self.basis, self.scale = basis, None
        if self.blocks:
            self.frames = self.recent + self.blocks * self.block  # whole blocks only
            sample = []
            for run in data.runs("train")[::4]:
                anchors, prefix = self._prefixes(run, self.frames)
                if len(anchors):
                    sample.append(self._older(prefix[::4]))
                run.__dict__.pop("brain", None)
            self.scale = np.concatenate(sample).std(0) + 1e-6
        super().__init__(data, stimulus, basis, log, windows={f"P{seconds:g}": self.frames})

    def _prefixes(self, run, frames):
        if frames == self.recent:
            return super()._prefixes(run, frames)
        ok = run.brain_ok & run.allowed
        bad = np.r_[0, np.cumsum(~ok)]
        anchors = np.arange(self.recent, run.n, FIT_STRIDE)
        anchors = anchors[bad[anchors] == bad[anchors - self.recent]]
        index = anchors[:, None] + np.arange(-frames, 0)
        valid = (index >= 0) & ok[np.maximum(index, 0)]
        brain = np.asarray(run.brain[:], dtype=np.float32)
        return anchors, np.where(valid[..., None], brain[np.maximum(index, 0)], self.basis.mean.astype(np.float32))

    def _older(self, prefix):
        """[B, blocks * C] averages of whole 15-s blocks before the last 30 s, newest first, in the recent frames'
        spatial components."""
        c = min(self.basis.projection.shape[1], HISTORY_WIDTH // self.recent)
        old = prefix[:, prefix.shape[1] - self.recent - self.blocks * self.block:prefix.shape[1] - self.recent]
        means = old.reshape(len(old), self.blocks, self.block, -1).mean(2)[:, ::-1]
        return ((means - self.basis.mean.astype(np.float32)) @ self.basis.projection[:, :c]).reshape(len(old), -1)

    def _features(self, prefix):
        recent = self.basis.features(prefix[:, -self.recent:], self.width)
        if not self.blocks:
            return recent
        return np.concatenate([recent, self._older(prefix) / self.scale.astype(np.float32)], axis=1)


class HistoryMean:
    """Each parcel's average over the valid history frames (0, the training mean, without any)."""

    def predict(self, batch):
        valid = batch.prefix_valid[..., None]
        total = np.where(valid, np.nan_to_num(batch.prefix), 0.0).sum(1)
        mean = total / np.maximum(valid.sum(1), 1)
        return np.repeat(mean[:, None].astype(np.float32), batch.horizon, axis=1)


@contextmanager
def late_origins(data):
    """Forecast origins with at least LONGEST seconds of the run before them."""
    first, original = frames_within(LONGEST, data.grid.tr), Q.forecast_origins
    Q.forecast_origins = lambda run, grid: (lambda o: o[o >= first])(original(run, grid))
    try:
        yield
    finally:
        Q.forecast_origins = original


def _name(kind, seconds):
    return f"{kind} {seconds:g} s"


def models(kind, data, seconds, device, log, features="projections", root=BENCHMARK):
    """(L, model) for every requested history length, built in turn (a fitted model is released after scoring)."""
    if kind in ("linear", "linear-matched"):
        stimulus = StimulusRidge(data, cache=Path(root) / data.dataset / f"stimulus-rows-{features}.npz", log=log,
                                 features=features)
        basis = SpatialBasis(data)
        tag = "" if features == "projections" else f" [{features}]"
        for s in seconds:
            fitted = WindowLinear(data, stimulus, s, basis, log, matched=kind == "linear-matched") if s else stimulus
            yield s, Window(fitted, s, _name(kind + tag, s))
    elif kind == "ipsid":
        from state_of_mind.benchmark.baselines import IPSIDBaseline
        model = IPSIDBaseline(data, "projections", log=log, output=str(BENCHMARK))
        for s in seconds:
            yield s, Window(model, s, _name(kind, s))
    elif kind == "mean":
        for s in seconds:
            yield s, Window(HistoryMean(), s, _name(kind, s))
    else:
        raise ValueError(kind)


def run(args):
    from state_of_mind.benchmark.data import BenchmarkData
    log = lambda text: print(text, flush=True)
    G.HISTORY_SECONDS = LONGEST  # every query carries 240 s; each model reads its last L
    U.BUDGETS = BUDGETS
    data = BenchmarkData(CONFIGS[args.dataset])
    root = Path(args.output)
    tasks = [s for _, splits, _ in PANELS[args.dataset] for s in splits]
    groups = [g for _, _, state in PANELS[args.dataset] for g in state]
    for seconds, model in models(args.model, data, args.seconds, args.device, log, args.features, args.benchmark_root):
        started = time.time()
        for split in tasks if "forecast" in args.tasks else ():
            F.run(data, model, split, output=root, log=log)
            with late_origins(data):
                F.run(data, model, split, output=root / "late", log=log)
        sums = {}
        for group in groups if "state" in args.tasks else ():
            source, keep, _ = S.GROUPS[data.dataset][group]
            sums[group] = S.collect(data, model, data.runs(source), None, log, keep)
        if sums:
            S.save(root / data.dataset / "state", model.name, {"model": model.name, "stamp": G.stamp()}, sums)
        if "updating" in args.tasks:
            U.run(data, model, splits=tasks, output=root, log=log)
        log(f"{model.name}: scored in {time.time() - started:.0f}s")


# ----------------------------------------------------------------------------------------------- report

def _skill(units, err, ref):
    """{value, low, high} of 1 - sum(err) / sum(ref) (per-unit arrays), and its bootstrap samples."""
    estimate, samples = units.ratio(err, ref)
    return Units.interval(1 - estimate, 1 - samples), 1 - samples


@lru_cache(maxsize=None)
def _origins(root, dataset, splits, name):
    """Number of scored Forecast origins."""
    paths = [root / dataset / split / name / "results.json" for split in splits]
    return sum(row["origins"] for p in paths if p.is_file() for row in json.loads(p.read_text())["forecast"])


@lru_cache(maxsize=None)
def _forecast(root, dataset, splits, name):
    """{unit: (err [H], ref [H])} pooled over the splits, or None."""
    out = {}
    for split in splits:
        path = root / dataset / split / name / "results.json"
        if not path.is_file():
            return None
        for row in json.loads(path.read_text())["forecast"]:
            s = row["scores"]["P30"]
            e, r = np.nan_to_num(np.asarray(s["err"], float)), np.nan_to_num(np.asarray(s["ref"], float))
            a, b = out.get(row["unit"], (0.0, 0.0))
            out[row["unit"]] = (a + e, b + r)
    return out


def _windows(dataset):
    grid = GRIDS[dataset]
    out = {"minute": np.arange(grid.horizon_steps)}
    out.update({f"{lo:g}-{hi:g} s": np.arange(frames_within(lo, grid.tr), frames_within(hi, grid.tr))
                for lo, hi in G.BANDS.values()})
    return out


def _cells(by_unit, reference, windows):
    """{window: (cell, paired cell vs the reference)} in percent."""
    common = sorted(set(by_unit) & set(reference or by_unit))
    units = Units(common)
    out = {}
    for label, steps in windows.items():
        err = np.array([by_unit[u][0][steps].sum() for u in common])
        ref = np.array([by_unit[u][1][steps].sum() for u in common])
        cell, samples = _skill(units, err, ref)
        diff = None
        if reference is not None:
            base, base_samples = _skill(units, np.array([reference[u][0][steps].sum() for u in common]),
                                        np.array([reference[u][1][steps].sum() for u in common]))
            diff = Units.interval(cell["value"] - base["value"], samples - base_samples)
        out[label] = (cell, diff)
    return out


def _fmt(cell, diff=None):
    if cell is None or cell["value"] is None:
        return "n/a"
    text = f"{100 * cell['value']:.1f}"
    if diff is not None and diff["low"] is not None:
        text += f" ({100 * diff['value']:+.1f} [{100 * diff['low']:+.1f}, {100 * diff['high']:+.1f}])"
    return text


@lru_cache(maxsize=None)
def _updating(root, dataset, splits, name):
    """{budget: {unit: (se [hidden], ref [hidden])}} pooled over the splits, or None."""
    path = root / dataset / "updating" / f"{name}.npz"
    if not path.is_file():
        return None
    sums = U.load(path.parent, name)
    out = {}
    for split in splits:
        for b, by_unit in sums.get(split, {}).items():
            for u, s in by_unit.items():
                out.setdefault(b, {})[u] = (np.asarray(s["step se"], float), np.asarray(s["step ref"], float))
    return out or None


@lru_cache(maxsize=None)
def _state(root, dataset, groups, name):
    path = root / dataset / "state" / f"{name}.npz"
    if not groups or not path.is_file():
        return None
    sums = S.load(path.parent, name)
    return {u: (s["difference step|err"].astype(float), s["difference step|energy"].astype(float))
            for g in groups for u, s in sums.get(g, {}).items()}


def _summary(root):
    """Compact tables: each model's skill over the minute by history length, and the paired gains from 30 to 120 s and
    from 120 to 240 s. Updating only for models that handle missing frames themselves (the linear forecaster's
    windows would be filled with the training mean)."""
    tasks = [("Forecast", lambda d, splits, groups, n: _forecast(root, d, splits, n),
              ("linear", "linear-matched", "ipsid", "mean")),
             ("State", lambda d, splits, groups, n: _state(root, d, groups, n), ("linear", "linear-matched", "ipsid")),
             ("Updating, 4 measurements", lambda d, splits, groups, n: (_updating(root, d, splits, n) or {}).get("4"),
              ("ipsid",))]
    lines = []
    for title, load, kinds in tasks:
        lines += [f"**{title}** (% over the minute; gains in percentage points, paired 95% intervals)", "",
                  "| Model | Panel | " + " | ".join(f"{s:g} s" for s in SECONDS) + " | 120 − 30 s | 240 − 120 s |",
                  "|---|---|" + "---:|" * len(SECONDS) + "---|---|"]
        for kind in kinds:
            for dataset, panels in PANELS.items():
                minute = {"minute": np.arange(frames_within(60.0, GRIDS[dataset].tr))}
                for label, splits, groups in panels:
                    found = {s: load(dataset, splits, groups, _name(kind, s)) for s in SECONDS
                             if s or not title.startswith("Updating")}  # see report()
                    found = {s: found.get(s) for s in SECONDS}
                    if not any(found.values()):
                        continue
                    cells = [_fmt(_cells(found[s], None, minute)["minute"][0]) if found[s] else "" for s in SECONDS]
                    gains = []
                    for a, b in ((120, 30), (240, 120)):
                        if found[a] and found[b]:
                            d = _cells(found[a], found[b], minute)["minute"][1]
                            gains.append(f"{100 * d['value']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]")
                        else:
                            gains.append("")
                    lines.append(f"| {kind} | {label.split(': ', 1)[0]}: {label.split(': ', 1)[1].split(' (')[0]} | "
                                 + " | ".join(cells + gains) + " |")
        lines.append("")
    return lines


def report(args):
    root = Path(args.output)
    lines = ["# History-length study", "", *_summary(root), "Each cell: skill in % (Forecast and Updating: % of measured BOLD variance "
             "predicted; State: % of the viewing difference predicted), then the paired difference from the same model "
             "with 30 s of history (percentage points, 95% bootstrap interval).", ""]
    for dataset, panels in PANELS.items():
        windows = _windows(dataset)
        hidden = frames_within(U.HIDDEN_SECONDS, GRIDS[dataset].tr)
        a, c = U.gap_window(hidden)
        for label, splits, groups in panels:
            sections = [("Forecast, all origins", lambda n: _forecast(root, dataset, splits, n), windows),
                        ("Forecast, origins with at least 240 s of the run before them",
                         lambda n: _forecast(root / "late", dataset, splits, n), windows),
                        ("State", lambda n: _state(root, dataset, groups, n), windows)]
            for budget in ("0", "4", "16"):
                sections.append((f"Updating, {budget} measurements", lambda n, b=budget: (_updating(root, dataset, splits, n) or {}).get(b),
                                 {"minute": np.arange(hidden)}))
            sections.append(("Updating, gap schedule: the 25-s gap", lambda n: (_updating(root, dataset, splits, n) or {}).get("gap"),
                             {"gap": np.arange(a, c), "last 5 s of gap": np.arange(c - frames_within(5, GRIDS[dataset].tr), c)}))
            counts = [_origins(r, dataset, splits, _name("linear", 0)) for r in (root, root / "late")]
            lines += [f"## {label}", "", f"Forecast origins: {counts[0]} in all, {counts[1]} with at least 240 s of the run "
                      "before them.", ""]
            for title, load, cols in sections:
                rows = []
                for kind in ("linear", "linear-matched", "ipsid", "mean"):
                    reference = load(_name("linear" if kind == "linear-matched" else kind, 30))
                    for s in SECONDS:
                        if title.startswith("Updating") and s == 0 and kind != "linear":
                            continue  # with no history a query also loses the revealed frames; only stimulus-only is defined
                        found = load(_name(kind, s))
                        if found:
                            cells = _cells(found, reference if s != 30 else None, cols)
                            rows.append(f"| {kind} | {s:g} | " + " | ".join(_fmt(*cells[w]) for w in cols) + " |")
                if rows:
                    lines += [f"**{title}**", "", "| Model | History (s) | " + " | ".join(cols) + " |",
                              "|---|---:|" + "---|" * len(cols), *rows, ""]
    text = "\n".join(lines)
    (root / "report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    r.add_argument("--model", choices=("linear", "linear-matched", "ipsid", "mean"), required=True)
    r.add_argument("--seconds", type=float, nargs="+", default=list(SECONDS))
    r.add_argument("--device", default="cpu")
    r.add_argument("--tasks", nargs="+", choices=("forecast", "state", "updating"), default=["forecast", "state", "updating"])
    r.add_argument("--output", default=str(OUTPUT))
    r.add_argument("--features", default="projections", choices=("raw", "projections", "context"),
                   help="Stimulus rows of the linear arm")
    r.add_argument("--benchmark-root", default=str(BENCHMARK), help="Where the stimulus-row caches are")
    p = sub.add_parser("report")
    p.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    from state_of_mind.benchmark.__main__ import _raise_open_file_limit
    _raise_open_file_limit()
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
