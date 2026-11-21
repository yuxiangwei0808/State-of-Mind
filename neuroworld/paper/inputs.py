"""neuroworld/paper/inputs.py (paper, Sec. 5.3, Table 3): evaluating the fitted 180-s Spectator without refitting under
four test-time input conditions on all Algonauts evaluation data (Friends S6 and the four Movie10 films):

  full          Full inputs (Forecast 16.0, State 5.5, Updating 20.3, r 0.349)
  no-stimulus   Drop the stimulus term from every rollout step (Forecast 7.5, State 5.5, Updating 13.7, r 0.224)
  no-fmri       Zero every fMRI frame read by the encoder (Forecast 9.4, State 0.0, Updating 8.4, r 0.253)
  other-viewer  Feed another individual's fMRI from the same movie, shifts 1-3 pooled (Forecast 6.7, State -5.1, Updating 1.2, r 0.236)

    python -m neuroworld.paper.inputs run --condition all [--device cuda] [--out outputs/paper-analysis/inputs]
    python -m neuroworld.paper.inputs report [--out outputs/paper-analysis/inputs]
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from neuroworld.benchmark import forecast as F, grid as G, state as S, updating as U
from neuroworld.benchmark.data import BenchmarkData
from neuroworld.paper.collect import latent_row
from neuroworld.paths import OUTPUT

RELEASE = OUTPUT / "spectator-v1/algonauts/model"
DEFAULT_OUT = OUTPUT / "paper-analysis/inputs"
SPLITS = ("val", "dev", "test")
VIEWERS = ("Algonauts2025/01", "Algonauts2025/02", "Algonauts2025/03", "Algonauts2025/05")
CONDITIONS = ("full", "no-stimulus", "no-fmri", "other-viewer")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


class OtherViewer:
    """Wrap a row so every query's fMRI (history and Updating reveals) comes from another viewer of the same recording,
    shifted by ``shift`` places in VIEWERS. Stimulus, probe, and targets remain those of the predicted viewer."""

    def __init__(self, row, data, shift):
        self.row, self.shift = row, shift
        self.name = f"other-viewer+{shift}"
        runs = [r for split in SPLITS for r in data.runs(split)]
        self.by_query = {r.query_id: r for r in runs}
        self.by_run = {r.run_id: r for r in runs}
        self.missing = 0

    def partner(self, query_id, subject):
        other = VIEWERS[(VIEWERS.index(subject) + self.shift) % len(VIEWERS)]
        return self.by_run.get(self.by_query[query_id].run_id.replace(subject, other))

    def predict(self, batch):
        if self.shift == 0:
            return self.row.predict(batch)
        prefix = np.full_like(batch.prefix, np.nan)
        valid = np.zeros_like(batch.prefix_valid)
        width = batch.prefix.shape[1]
        for i in range(len(batch)):
            run = self.partner(batch.run_ids[i], batch.subject_ids[i])
            if run is None:
                self.missing += 1
                continue
            frames = int(batch.origins[i]) - width + np.arange(width)
            ok = batch.prefix_valid[i] & (frames >= 0) & (frames < run.n)
            ok[ok] &= run.brain_ok[frames[ok]]
            if ok.any():
                prefix[i, ok] = np.asarray(run.brain[frames[ok]], dtype=np.float32)
            valid[i] = ok
        return self.row.predict(replace(batch, prefix=prefix, prefix_valid=valid))


def _build_condition(condition, data, model_dir, probe_path, device):
    """Build (list of named model wrappers) for one input condition using the frozen Spectator and its probe."""
    store = OUTPUT / "benchmark-results"
    if condition == "other-viewer":
        _, _, row, _, _ = latent_row(model_dir, "spectator", 180.0, store, data, device=device, probe_path=probe_path)
        return [OtherViewer(row, data, shift) for shift in (1, 2, 3)]
    _, _, row, _, model = latent_row(model_dir, condition, 180.0, store, data, device=device, probe_path=probe_path)
    row.name = condition
    if condition == "no-stimulus":
        with torch.no_grad():
            model.predictor.condition.weight.zero_()
            model.predictor.condition.bias.zero_()
    elif condition == "no-fmri":
        model.zero_brain = True
    return [row]


def _score_model(data, model, out):
    for split in SPLITS:
        F.run(data, model, split, output=out, workers=1, log=log)
    source, keep, _ = S.GROUPS[data.dataset]["dev"]
    state_sums = {"dev": S.collect(data, [model], data.runs(source), None, log, keep)[model.name]}
    S.save(out / data.dataset / "state", model.name, {"model": model.name, "stamp": G.stamp()}, state_sums)
    old_budgets = U.BUDGETS
    U.BUDGETS = (4,)
    try:
        upd_sums = {split: U.collect(data, [model], data.runs(split), log)[model.name] for split in SPLITS}
    finally:
        U.BUDGETS = old_budgets
    U.save(out / data.dataset / "updating", model.name, {"model": model.name, "stamp": G.stamp()}, upd_sums)


def run(args):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or hard > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536, hard))
    G.HISTORY_SECONDS = 180.0
    data = BenchmarkData("neuroworld/configs/algonauts/base.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for cond in todo:
        log(f"running input condition: {cond}")
        for model in _build_condition(cond, data, args.model_dir, args.probe, args.device):
            _score_model(data, model, out)
            if isinstance(model, OtherViewer):
                log(f"{model.name}: queries without partner recording: {model.missing}")
    report(args)


def _add(acc, unit, a, b):
    x, y = acc.get(unit, (0.0, 0.0))
    acc[unit] = (x + a, y + b)


def _load_row(out, name):
    f, r, u4, st = {}, {}, {}, {}
    for split in SPLITS:
        path = out / "algonauts" / split / name
        if (path / "results.json").is_file():
            for row in F.load(path)["forecast"]:
                s = row["scores"]["P30"]
                _add(f, row["unit"], float(np.nansum(np.asarray(s["err"], float))), float(np.nansum(np.asarray(s["ref"], float))))
                val = float(np.nanmean(np.asarray(s["pearson"], float)[:40]))
                if np.isfinite(val):
                    _add(r, row["unit"], val, 1)
    upd_path = out / "algonauts" / "updating" / f"{name}.npz"
    if upd_path.is_file():
        loaded = U.load(upd_path.parent, name)
        for split in SPLITS:
            for unit, v in loaded.get(split, {}).get("4", {}).items():
                _add(u4, unit, float(v["se"]), float(v["ref"]))
    st_path = out / "algonauts" / "state" / f"{name}.npz"
    if st_path.is_file():
        for unit, v in S.load(st_path.parent, name).get("dev", {}).items():
            _add(st, unit, float(v["difference step|err"].sum()), float(v["difference step|energy"].sum()))
    return f, st, u4, r


def _pool(parts):
    out = {}
    for d in parts:
        for unit, (a, b) in d.items():
            _add(out, unit, a, b)
    return out


def report(args):
    out = Path(args.out)
    skill = lambda s: None if not s else round(100 * (1 - sum(e for e, _ in s.values()) / sum(r for _, r in s.values())), 2)
    mean_r = lambda s: None if not s else round(sum(a for a, _ in s.values()) / sum(n for _, n in s.values()), 4)
    labels = [("Full inputs", ["full"]),
              ("Without stimulus", ["no-stimulus"]),
              ("Without fMRI", ["no-fmri"]),
              ("Another individual's fMRI", [f"other-viewer+{k}" for k in (1, 2, 3)])]
    table = {}
    print("| Input condition | Forecast | State | Updating | r |")
    print("|---|---:|---:|---:|---:|")
    for label, names in labels:
        rows = [_load_row(out, n) for n in names]
        f, st, u4, r = (_pool([row[i] for row in rows]) for i in range(4))
        if not f and not st and not u4:
            continue
        entry = {"Forecast": skill(f), "State": skill(st), "Updating": skill(u4), "r": mean_r(r)}
        table[label] = entry
        fmt = lambda v, d=1: "--" if v is None else f"{v:.{d}f}"
        print(f"| {label} | {fmt(entry['Forecast'])} | {fmt(entry['State'])} | {fmt(entry['Updating'])} | {fmt(entry['r'], 3)} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results-inputs.json").write_text(json.dumps(table, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Evaluate Spectator under one or all input conditions")
    r.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    r.add_argument("--model-dir", default=str(RELEASE))
    r.add_argument("--probe", help="Optional probe .npz overriding <model-dir>/probe.npz")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--device", default="cuda")
    p = sub.add_parser("report", help="Summarize saved input-condition results into Table 3")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
