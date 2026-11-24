"""state_of_mind/paper/history_table.py: the paper's main table by history length (7.5, 120 and 180 s; Appendix Fig. 5 adds 15, 30, 60
and 90 s), on all Algonauts test data:
Forecast (Friends S6 and all four Movie10 movies), State (the Movie10 repeats) and Updating with 4 new fMRI frames, for every model
at every length it runs at; and Spectator minus each model at the same length (paired, 95% interval over content units).
Also Pearson r of the forecast: per recording, each parcel's correlation between forecast and measured fMRI across starting
points at each frame ahead, averaged over parcels, the 40 frames and recordings (its steps 1-20 variant and other standard
metrics are in state_of_mind/paper/standard_metrics.py).
Writes $STATE_OF_MIND_OUTPUT/paper-analysis/results-history-table.json.

    python -m state_of_mind.paper.history_table
"""
import sys
import json

import numpy as np

from state_of_mind.benchmark import forecast as F, profile as P, report as R, updating as U
from state_of_mind.paper.analysis import A, ROOTS, SPLITS, UPDATING_ROOTS, find

LENGTHS = ("7.5 s", "120 s", "180 s")
CURVE = ("7.5 s", "15 s", "30 s", "60 s", "90 s", "120 s", "180 s")  # Appendix Fig. 5
# Benchmark rows of each model at each history length.
ROWS = {
    "Linear forecaster": {h: f"linear [context] {h}" for h in CURVE},
    "IPSID": {"7.5 s": "IPSID [context] 7.5 s", **{f"{s} s": f"IPSID [context] {s} s" for s in (15, 30, 60, 90)},
              "120 s": "IPSID [context]", "180 s": "IPSID [context] 180 s"},
    "NeuroWorld": {"7.5 s": "NeuroWorld-style k1 (latent)", **{h: f"NeuroWorld k1 {h[:-2]}s (latent)" for h in CURVE[1:]}},
    "TRIBE v2": {h: "TRIBEv2 zero-shot" for h in CURVE},  # reads no fMRI
    "Spectator": {h: f"ridge-jepa-{h[:-2]}s-std-partial (latent)" for h in CURVE},
}
TASKS = ("Forecast", "State", "Updating")


def forecast_units(name):
    out = {}
    for split in SPLITS:
        path = find(f"algonauts/{split}/{name}/results.json")
        if path is None:
            return None
        for row in F.load(path.parent)["forecast"]:
            s = row["scores"]["P30"]
            a, b = out.get(row["unit"], (0.0, 0.0))
            out[row["unit"]] = (a + np.nansum(np.asarray(s["err"], float)), b + np.nansum(np.asarray(s["ref"], float)))
    return out


def updating_units(name):
    out = {}
    for split in SPLITS:
        got = None
        for root in UPDATING_ROOTS:
            if (root / "algonauts/updating" / f"{name}.npz").is_file():
                loaded = U.load(root / "algonauts/updating", name)
                if split in loaded:
                    got = loaded[split]
                    break
        if got is None:
            return None
        for unit, v in got.get(P.HEADLINE_BUDGET, {}).items():
            a, b = out.get(unit, (0.0, 0.0))
            out[unit] = (a + float(v["se"]), b + float(v["ref"]))
    return out


def state_units(name):
    for root in ROOTS:
        if (root / "algonauts/state" / f"{name}.npz").is_file():
            P.ROOT = root
            u = P._unit_sums("state", "algonauts", ["dev"], name)
            if u:
                return u
    return None


def pearson_units(name, steps=40):
    """{content unit: (sum over its recordings of the forecast's Pearson r over the first <steps> frames, recordings)}."""
    out = {}
    for split in SPLITS:
        path = find(f"algonauts/{split}/{name}/results.json")
        if path is None:
            return None
        for row in F.load(path.parent)["forecast"]:
            r = np.nanmean(np.asarray(row["scores"]["P30"]["pearson"], float)[:steps])
            if np.isfinite(r):
                a, b = out.get(row["unit"], (0.0, 0))
                out[row["unit"]] = (a + r, b + 1)
    return out


def pearson(name, steps=40):
    """Pearson r of the forecast over the first <steps> frames ahead, averaged over recordings (None if a split is missing)."""
    u = pearson_units(name, steps)
    return None if not u else sum(v[0] for v in u.values()) / sum(v[1] for v in u.values())


def paired_r(a, b):
    """Mean r of a minus mean r of b, with its paired 95% interval over their common content units (as profile._paired)."""
    common = sorted(set(a) & set(b))
    units = R.Units(common)
    (ea, sa), (eb, sb) = (units.ratio(np.array([x[u][0] for u in common], float), np.array([x[u][1] for u in common], float))
                          for x in (a, b))
    return R.Units.interval(ea - eb, sa - sb)


def skill(u):
    return None if not u else 100 * (1 - sum(v[0] for v in u.values()) / sum(v[1] for v in u.values()))


def unit_sums(lengths):
    """{(model, length, task): per-unit (error, reference) sums, or None where the model does not run}."""
    fns = {"Forecast": forecast_units, "State": state_units, "Updating": updating_units}
    return {(label, h, task): None if h not in rows else fns[task](rows[h])
            for label, rows in ROWS.items() for h in lengths for task in TASKS}


def main():
    units = unit_sums(LENGTHS)
    table = {label: {h: {**{task: skill(units[label, h, task]) for task in TASKS},
                         "r": pearson(rows[h]) if h in rows else None} for h in LENGTHS} for label, rows in ROWS.items()}
    diffs = {}
    for label in ROWS:
        if label == "Spectator":
            continue
        for h in LENGTHS:
            for task in TASKS:
                mine, theirs = units["Spectator", h, task], units[label, h, task]
                if mine and theirs:
                    d = P._paired(mine, theirs)
                    if d and d["low"] is not None:
                        diffs.setdefault(label, {}).setdefault(h, {})[task] = f"{d['value']:+.1f} [{d['low']:+.1f}, {d['high']:+.1f}]"
    for h in LENGTHS:
        mine = pearson_units(ROWS["Spectator"][h])
        for label in ROWS:
            if label != "Spectator" and (theirs := pearson_units(ROWS[label][h])):
                d = paired_r(mine, theirs)
                diffs.setdefault(label, {}).setdefault(h, {})["r"] = f"{d['value']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
    A.mkdir(parents=True, exist_ok=True)
    (A / "results-history-table.json").write_text(json.dumps({"table": table, "spectator_minus": diffs}, indent=1))
    cell = lambda x, t: "--" if x is None else f"{x:.3f}" if t == "r" else f"{x:.1f}"
    print("| Model | " + " | ".join(f"{h} {t}" for h in LENGTHS for t in (*TASKS, "r")) + " |")
    for label, v in table.items():
        print(f"| {label} | " + " | ".join(cell(v[h][t], t) for h in LENGTHS for t in (*TASKS, "r")) + " |")
    print()
    for label, d in diffs.items():
        print(label, json.dumps(d))


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):  # this script parses no options: show its usage and run nothing
        sys.exit(print(__doc__))
    main()
