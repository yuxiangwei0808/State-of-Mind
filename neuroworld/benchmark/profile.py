"""The benchmark card and the scoring helpers the paper's scripts use, read from the saved results.

The card has one table per panel (Algonauts: Friends S6 and the four Movie10 films; HCP: known participants' held-out
clips and run 4; HCP: new participants after run-1 calibration) with Forecast, State and Updating for every scored row,
each with a small curve, Pareto marks (point estimates: no other row is at least as good on every capability and better
on one), paired differences from a chosen reference row, and the predictor nonanticipation check. Splits within a
panel are pooled by adding their sums. Recordings a model was fitted or calibrated on are listed separately.

    python -m neuroworld.benchmark profile [--reference <row>]    # writes <output>/card.md
"""
import json
from pathlib import Path

import numpy as np

from neuroworld.benchmark import forecast as F, report as R, state as S, updating as U
from neuroworld.benchmark.grid import GRIDS, frames_within

ROOT = Path("artifacts/benchmark")
# Panels: tasks are the Forecast and Updating splits; state the State groups.
PANELS = [
    {"dataset": "algonauts", "label": "Algonauts: Friends S6 and the four Movie10 films",
     "tasks": ("val", "dev", "test"), "state": ("dev",)},
    {"dataset": "hcp_movies", "label": "HCP: known participants, held-out clips and run 4",
     "tasks": ("val", "test"), "state": ()},
    {"dataset": "hcp_movies", "label": "HCP: new participants, after run-1 calibration (runs 2-4)",
     "tasks": ("heldout-adapt", "heldout-test"), "state": ("heldout-adapt",)},
]
SEEN = [{"dataset": "algonauts", "label": "Algonauts training recordings (Friends S1-5)", "tasks": ("train",), "state": ()},
        {"dataset": "hcp_movies", "label": "HCP training participants' training recordings", "tasks": ("train",),
         "state": ("train",)},
        {"dataset": "hcp_movies", "label": "HCP new participants' calibration run (run 1)", "tasks": ("heldout-fit",),
         "state": ("heldout-calibration",)}]
CAPABILITIES = ("forecast", "state", "updating")  # the Pareto comparison


def models(dataset):
    """Every row with Forecast results in one of the dataset's splits, sorted by name."""
    names = set()
    for split in {s for panel in PANELS + SEEN if panel["dataset"] == dataset for s in panel["tasks"]}:
        folder = ROOT / dataset / split
        if folder.is_dir():
            names |= {path.name for path in folder.iterdir() if (path / "results.json").is_file()}
    return sorted(names)


CURVE_SECONDS = (1.5, 6.0, 20.0, 60.0)
CURVE_BUDGETS = ("0", "4", "16", "all")
SINCE_SECONDS = (2.0, 5.0, 10.0, 20.0)  # gap curve: seconds since the last measurement


def _merge(parts):
    """{unit: {key: sum}} added over parts (a unit appearing in several parts adds up)."""
    out = {}
    for part in parts:
        for unit, sums in part.items():
            if unit in out:
                out[unit] = {k: out[unit][k] + np.asarray(v, dtype=np.float64) for k, v in sums.items()}
            else:
                out[unit] = {k: np.asarray(v, dtype=np.float64) for k, v in sums.items()}
    return out


def _steps(dataset, seconds):
    """0-based index of the last step at or before each time (at least the first step)."""
    tr = GRIDS[dataset].tr
    return [max(1, frames_within(s, tr)) - 1 for s in seconds]


def forecast_cells(dataset, splits, name):
    """% of measured BOLD variance predicted over the minute (100 x (1 - summed error / summed run-mean reference
    error), every scored frame weighing the same) and the same at CURVE_SECONDS."""
    err = ref = None
    for split in splits:
        path = ROOT / dataset / split / name
        if not (path / "results.json").is_file():
            return None
        for row in F.load(path)["forecast"]:
            s = row["scores"]["P30"]
            if "err" not in s:
                return None
            e, r = np.nan_to_num(np.asarray(s["err"], float)), np.nan_to_num(np.asarray(s["ref"], float))
            err, ref = (e, r) if err is None else (err + e, ref + r)
    if err is None:
        return None
    horizon = GRIDS[dataset].horizon_steps
    return {"value": 100 * (1 - err[:horizon].sum() / ref[:horizon].sum()),
            "curve": [100 * (1 - err[k] / ref[k]) for k in _steps(dataset, CURVE_SECONDS)]}


def state_cells(dataset, groups, name):
    """% of the viewing difference predicted over the minute, at CURVE_SECONDS, and the horizon of detected positive
    skill (the seconds ahead for which every step's 95% interval stays above 0)."""
    path = ROOT / dataset / "state" / f"{name}.npz"
    if not groups or not path.is_file():
        return None
    sums = S.load(path.parent, name)
    if not all(g in sums for g in groups):
        return None
    merged = _merge(sums[g] for g in groups)
    if not merged or any("difference step|err" not in s for s in merged.values()):
        return None
    out = S.difference_summary(merged, GRIDS[dataset].tr)
    steps = out["by_step"]
    return {"value": 100 * out["skill"]["value"], "seconds": out["seconds_helping"],
            "curve": [100 * steps[k]["value"] for k in _steps(dataset, CURVE_SECONDS) if k < len(steps)]}


HEADLINE_BUDGET = "4"  # Updating's headline: one measurement every 12-15 s, so a model must carry its state between
# them; with 16 (one every ~4 s) short-range extrapolation from the latest frame already does well


def updating_cells(dataset, splits, name):
    """% predicted with HEADLINE_BUDGET measurements in the minute (and with 16), at CURVE_BUDGETS, and in the 25-s gap
    of the "gap" schedule: over the gap window beside no measurements on the same window, and at SINCE_SECONDS after
    the last measurement."""
    path = ROOT / dataset / "updating" / f"{name}.npz"
    if not path.is_file():
        return None
    sums = U.load(path.parent, name)
    if not all(s in sums for s in splits) or "gap" not in sums[splits[0]]:
        return None
    total = lambda b, k: sum(v[k] for s in splits for v in sums[s][b].values())
    skill = lambda b: 100 * (1 - total(b, "se") / total(b, "ref"))
    se = lambda b: total(b, "step se")
    ref = lambda b: total(b, "step ref")
    hidden = len(se("gap"))
    a, c = U.gap_window(hidden)
    window = lambda b: 100 * (1 - se(b)[a:c].sum() / ref(b)[a:c].sum())
    last = max(k for k in U.schedule(hidden, "gap") if k < a)
    since = [last + int(round(s / GRIDS[dataset].tr)) for s in SINCE_SECONDS]
    scored = lambda b: all(b in sums[s] for s in splits)  # rows scored before a budget was added lack it
    return {"value": skill(HEADLINE_BUDGET), "16": skill("16"), "curve": [skill(b) if scored(b) else None for b in CURVE_BUDGETS],
            "gap": window("gap"),
            "gap none": window("0"), "since": [100 * (1 - se("gap")[k] / ref("gap")[k]) for k in since if k < c]}



def nonanticipation(dataset, name):
    path = ROOT / dataset / "nonanticipation" / f"{name}.json"
    return json.loads(path.read_text())["pass"] if path.is_file() else None


def pareto(rows):
    """{name: True} for models that no other model matches or beats on every capability and beats on one (point
    estimates). To dominate a model, another must have a score wherever it has one."""
    value = lambda cells, k: None if not cells.get(k) or cells[k].get("value") is None else cells[k]["value"]
    out = {}
    for a, ca in rows.items():
        keys = [k for k in CAPABILITIES if value(ca, k) is not None]
        dominated = False
        for b, cb in rows.items():
            if a == b or not keys or any(value(cb, k) is None for k in keys):
                continue
            if all(value(cb, k) >= value(ca, k) for k in keys) and any(value(cb, k) > value(ca, k) for k in keys):
                dominated = True
                break
        out[a] = not dominated
    return out


def _pct(value):
    return "n/a" if value is None else f"{value:.1f}%"


def _curve(values):
    return "n/a" if not values else " → ".join("–" if v is None else f"{v:.0f}" for v in values)



def _panel_table(panel, names):
    """Card lines: one panel's table for the rows names, with Pareto marks among the rows scored on every capability."""
    d, rows = panel["dataset"], {}
    for name in names:
        cells = {"forecast": forecast_cells(d, panel["tasks"], name), "state": state_cells(d, panel["state"], name),
                 "updating": updating_cells(d, panel["tasks"], name)}
        if any(cells.values()):
            rows[name] = cells
    if not rows:
        return []
    needed = [k for k in CAPABILITIES if k != "state" or panel["state"]]
    best = pareto({n: c for n, c in rows.items() if all(c.get(k) for k in needed)})
    lines = ["| Model | Forecast: % of BOLD variance predicted over the minute | curve | State: % of the viewing "
             "difference predicted | curve | Updating: % predicted, 4 measurements (16) | curve | 25-s gap (no "
             "measurements at all) | gap curve | Nonanticipation |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, c in rows.items():
        f, s, u = c["forecast"] or {}, c["state"] or {}, c["updating"] or {}
        gap = f"{u['gap']:.1f}% ({u['gap none']:.1f}%)" if u else "n/a"
        upd = f"{u['value']:.1f}% ({u['16']:.1f}%)" if u else "n/a"
        check = nonanticipation(d, name)
        lines.append(f"| {name}{' ★' if best.get(name) else ''} | {_pct(f.get('value'))} | {_curve(f.get('curve'))} "
                     f"| {_pct(s.get('value'))} | {_curve(s.get('curve'))} | {upd} | {_curve(u.get('curve'))} | {gap} | "
                     f"{_curve(u.get('since'))} | {'—' if check is None else 'pass' if check else 'FAIL'} |")
    return lines + [""]


def card(reference=None):
    lines = ["# Benchmark card", "",
             "Forecast and Updating: % of measured BOLD variance predicted (0% = predicting the run's average). State: % of "
             "the measured difference between two viewings predicted (0% = predicting no difference). Curves: Forecast and "
             "State at the first step (1.5 s on Algonauts, 1 s on HCP), "
             + ", ".join(f"{s:g}" for s in CURVE_SECONDS[1:]) + " s after the origin; Updating with "
             + ", ".join(CURVE_BUDGETS).replace("all", "every frame") + " measurements in the minute; the gap curve at "
             + ", ".join(f"{s:g}" for s in SINCE_SECONDS) + " s after the last measurement before a 25-s gap. A latent "
             "model is scored through a linear probe of its predicted states (its \"(latent)\" row). ★: on the Pareto "
             "frontier of point estimates; close values may not differ (paired intervals below).", "",
             _stamp_line(stamps(), "Results"), ""]
    mixed = [f"{name} ({dataset}): {sorted(str(i) for i in ids)}" for dataset in GRIDS for name in models(dataset)
             for ids in [_probe_mix(dataset, name)] if ids]
    if mixed:
        lines += ["**Warning: these rows combine results from different probes (or files without one); rescore them:** "
                  + "; ".join(mixed) + ".", ""]
    for panel in PANELS:
        lines += [f"## {panel['label']}", ""] + _panel_table(panel, models(panel["dataset"]))
    if reference:
        lines += paired_table(reference)
    lines += ["# Seen data", "",
              "Recordings used to fit the model (training) or to calibrate it (new participants' run 1); a model can "
              "memorize these.", "", "| Recordings | Model | Forecast | State | Updating, 4 measurements |",
              "|---|---|---:|---:|---:|"]
    for panel in SEEN:
        d = panel["dataset"]
        for name in models(d):
            f, s = forecast_cells(d, panel["tasks"], name), state_cells(d, panel["state"], name)
            u = updating_cells(d, panel["tasks"], name)
            if f or s or u:
                lines.append(f"| {panel['label']} | {name} | {_pct((f or {}).get('value'))} | "
                             f"{_pct((s or {}).get('value'))} | {_pct((u or {}).get('value'))} |")
    return "\n".join(lines) + "\n"


def _unit_sums(task, dataset, splits, name):
    """{unit: (error, reference error)} summed over a panel's splits: Forecast over the minute, State's viewing
    difference (error and energy), Updating with HEADLINE_BUDGET measurements; None when a result is missing."""
    out = {}
    def add(unit, e, r):
        a, b = out.get(unit, (0.0, 0.0))
        out[unit] = (a + float(e), b + float(r))
    if not splits:
        return None
    if task == "forecast":
        for split in splits:
            path = ROOT / dataset / split / name
            if not (path / "results.json").is_file():
                return None
            for row in F.load(path)["forecast"]:
                s = row["scores"]["P30"]
                if "err" not in s:
                    return None
                add(row["unit"], np.nansum(np.asarray(s["err"], float)), np.nansum(np.asarray(s["ref"], float)))
    else:
        path = ROOT / dataset / task / f"{name}.npz"
        if not path.is_file():
            return None
        sums = (S if task == "state" else U).load(path.parent, name)
        if not all(s in sums for s in splits):
            return None
        for split in splits:
            for unit, v in (sums[split] if task == "state" else sums[split].get(HEADLINE_BUDGET, {})).items():
                if task == "state":
                    if "difference step|err" not in v:
                        return None
                    add(unit, v["difference step|err"].sum(), v["difference step|energy"].sum())
                else:
                    add(unit, v["se"], v["ref"])
    return out or None


def _paired(a, b):
    """Skill of a minus skill of b in percentage points, with its paired 95% interval over their common units."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    units = R.Units(common)
    (ea, sa), (eb, sb) = (units.ratio(np.array([x[u][0] for u in common]), np.array([x[u][1] for u in common]))
                          for x in (a, b))
    return R.Units.interval(100 * (eb - ea), 100 * (sb - sa))



def paired_table(reference):
    """Card lines: each row minus the reference per capability and panel, with the approximate detectable difference
    (1.43 x the median paired interval half-width: about 80% power for typical comparisons)."""
    lines = ["# Paired differences", "",
             f"Each row minus {reference}, in percentage points (95% paired intervals over content units on Algonauts and "
             "participants on HCP). Detectable: roughly the smallest difference a typical comparison resolves in that panel.",
             "", "| Panel / model | Forecast | State | Updating |", "|---|---|---|---|"]
    fmt = lambda c: "n/a" if not c or c["low"] is None else f"{c['value']:+.1f} [{c['low']:+.1f}, {c['high']:+.1f}]"
    for panel in PANELS:
        d = panel["dataset"]
        splits = {"forecast": panel["tasks"], "state": panel["state"], "updating": panel["tasks"]}
        base = {k: _unit_sums(k, d, splits[k], reference) for k in CAPABILITIES}
        rows = {}
        for name in models(d):
            if name == reference:
                continue
            cells = {k: _paired(mine, base[k]) for k in CAPABILITIES
                     for mine in [_unit_sums(k, d, splits[k], name)] if mine and base[k]}
            if cells:
                rows[name] = cells
        if not rows:
            continue
        half = [[(c[k]["high"] - c[k]["low"]) / 2 for c in rows.values() if c.get(k) and c[k]["low"] is not None]
                for k in CAPABILITIES]
        detect = " / ".join("–" if not h else f"{1.43 * np.median(h):.1f}" for h in half)
        lines.append(f"| **{panel['label']}** (detectable {detect}) | | | |")
        for name, cells in rows.items():
            lines.append(f"| {name} | " + " | ".join(fmt(cells.get(k)) for k in CAPABILITIES) + " |")
    return lines + [""]


def stamps():
    """{(protocol, source and code): count} over the result files under ROOT; results written before stamps count as
    unstamped."""
    paths = list(ROOT.glob("*/*/*/results.json")) + list(ROOT.glob("*/state/*.json")) + list(ROOT.glob("*/updating/*.json"))
    out = {}
    for path in paths:
        s = json.loads(path.read_text()).get("stamp") or {"protocol": "unstamped", "source": "unknown"}
        if not isinstance(s, dict):
            continue
        key = (s["protocol"], s["source"] + (f", code {s['code']}" if s.get("code") else ""))
        out[key] = out.get(key, 0) + 1
    return out


def _probe_mix(dataset, name):
    """The probe weight digests in a model's result files (None for a file without one) when they differ, else None:
    a row must come from one frozen probe."""
    paths = [ROOT / dataset / split / name / "results.json" for split in ("train", "val", "dev", "test", "heldout-fit",
                                                                          "heldout-adapt", "heldout-test")]
    paths += [ROOT / dataset / task / f"{name}.json" for task in ("state", "updating", "nonanticipation")]
    ids = {(json.loads(p.read_text()).get("stamp") or {}).get("readout", {}).get("weights sha256")
           for p in paths if p.is_file()}
    return ids if len(ids) > 1 and any(ids) else None


def _stamp_line(found, label):
    """The protocol and code versions of the results; a warning only when the protocols differ (results from several
    code versions of one protocol are expected as models are added)."""
    if len(found) == 1:
        (protocol, source), = found
        return f"{label}: protocol {protocol}, source {source}."
    listing = "; ".join(f"{s} ({n} files)" for (p, s), n in sorted(found.items()))
    protocols = sorted({p for p, _ in found})
    if len(protocols) == 1:
        return f"{label}: protocol {protocols[0]}; sources: {listing}."
    return f"**Warning: {label} come from different protocols:** " + "; ".join(
        f"{p}, {s} ({n} files)" for (p, s), n in sorted(found.items())) + "."



def run(output=ROOT, reference=None):
    global ROOT
    ROOT = Path(output)  # read the results of this output directory, not the default one
    text = card(reference)
    (Path(output) / "card.md").write_text(text)
    return text
