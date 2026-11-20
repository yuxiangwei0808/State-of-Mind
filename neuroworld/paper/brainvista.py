"""neuroworld/paper/brainvista.py (paper, Table 2 and its significance table): BrainVista's row.

BrainVista (Yin et al., arXiv 2602.04512) is the raw-features version, trained by a collaborator for each viewer and history
length with the published code and settings (checkpoints in $NEUROWORLD_EXTERNAL/baselines/brainvista/ckpt). As released, its forecasts are about 1.6 times as spread out as the measured fMRI
(its detokenizer's input LayerNorm rescales the shrunk predicted tokens), so its Forecast, State and Updating scores are about
-180%, -100% and -180% while its correlations are positive.

The paper's row rescales its forecasts: one gain and one offset per parcel, shared by the four viewers. The gains are cross-fitted
on the evaluation data, which favours BrainVista: the 66 content units (episodes and movie segments, the
bootstrap units) are split into two halves, and the gains fitted by least squares on the Forecast forecasts of one half are
applied to the other half, in Forecast, State and Updating. No forecast is scored with a gain fitted on its own unit. The paper
does not describe the rescaling; this file is its record.

Everything is computed from the stored forecasts and results, with no model run:
  Forecast   per-unit, per-parcel sums over the saved forecasts (pred/forecasts/brainvista_<L>_s); each unit's reference error
             is the stored one
  State      the saved forecasts of the two viewings of each Movie10 segment watched twice (the offset cancels)
  Updating   the stored per-parcel sums of the Updating task with 4 new fMRI frames (count, sum x, sum y, sum xy, sum xx, sum yy)
  r          unchanged by a positive gain per parcel (every fitted gain is positive), so the stored values
The scores without rescaling are recomputed the same way as a check: Forecast and Updating reproduce the stored results exactly,
State within 0.4 points (the pairs of starting points differ slightly). Differences from Spectator use the helpers of
neuroworld/paper/ablations.py, as for the other rows.

    python -m neuroworld.paper.brainvista 7.5 120 180      writes paper-analysis/results-brainvista.json

Result (Forecast / State / Updating / r): 7.5 s 5.0 / -0.4 / 8.4 / 0.184; 120 s 6.4 / 0.2 / 10.2 / 0.213; 180 s 6.6 / 0.2 / 10.3 /
0.220. With gains fitted on BrainVista's forecasts for the training recordings instead (two starting points per recording): 3.8 /
-1.6 / 8.5, 5.7 / -0.9 / 10.7 and 5.9 / -0.9 / 10.6. The scripts of that variant and all outputs are in
$NEUROWORLD_OUTPUT/paper-analysis/extra-nov14/bv/.
"""
import collections
import json
import sys

import numpy as np

from neuroworld.benchmark import updating as U
from neuroworld.paper import ablations as T
from neuroworld.paper.analysis import A
from neuroworld.paths import EXTERNAL

BV = EXTERNAL / "baselines/history-length-brainvista"  # BrainVista's benchmark results
FC = EXTERNAL / "baselines/pred/forecasts"             # its saved forecasts and targets
SPLITS = ("val", "dev", "test")
SPECTATOR = {"7.5": "ridge-jepa-7.5s-std-partial (latent)", "120": "ridge-jepa-120s-std-partial (latent)", "180": "ridge-jepa-180s-std-partial (latent)"}
KEYS = ("n", "x", "y", "xx", "xy", "yy")


def score(sums):
    return round(100 * (1 - sum(v[0] for v in sums.values()) / sum(v[1] for v in sums.values())), 2)


def sums_from_forecasts(length):
    """(per-unit, per-parcel sums over the saved forecasts; per-unit, per-parcel sums of the differences between two viewings)."""
    forecast = collections.defaultdict(lambda: {k: np.zeros(1000) for k in KEYS})
    state = collections.defaultdict(lambda: {k: np.zeros(1000) for k in ("pp", "py", "yy")})
    dev = {}
    for split in SPLITS:
        index = json.loads((FC / f"brainvista_{length}_s" / split / "index.json").read_text())["runs"]
        for run_id, meta in index.items():
            f = np.load(FC / f"brainvista_{length}_s" / split / meta["file"])
            o = np.load(FC / "_observed" / split / meta["observed"])
            pred, obs, ok = f["forecast"].astype(np.float32), o["observed"].astype(np.float32), o["ok"]
            w = ok[..., None].astype(np.float32)
            x, y = pred * w, obs * w
            s = forecast[meta["unit"]]
            s["n"] += w.sum((0, 1)) * np.ones(1000)
            s["x"] += x.sum((0, 1), dtype=np.float64)
            s["y"] += y.sum((0, 1), dtype=np.float64)
            s["xx"] += (x * x).sum((0, 1), dtype=np.float64)
            s["xy"] += (x * y).sum((0, 1), dtype=np.float64)
            s["yy"] += (y * y).sum((0, 1), dtype=np.float64)
            if split == "dev":
                dev[run_id] = (pred, obs, ok, f["origins"], meta["unit"])
    for run_id, (p1, y1, k1, o1, unit) in dev.items():  # the two viewings of a Movie10 segment: run-01 and run-02
        other = run_id[:-1] + "2"
        if not run_id.endswith("run-01") or other not in dev:
            continue
        p2, y2, k2, o2, _ = dev[other]
        common = np.intersect1d(o1, o2)
        i1, i2 = np.searchsorted(o1, common), np.searchsorted(o2, common)
        both = (k1[i1] & k2[i2])[..., None].astype(np.float32)
        dp, dy = (p1[i1] - p2[i2]) * both, (y1[i1] - y2[i2]) * both
        s = state[unit]
        s["pp"] += (dp * dp).sum((0, 1), dtype=np.float64)
        s["py"] += (dp * dy).sum((0, 1), dtype=np.float64)
        s["yy"] += (dy * dy).sum((0, 1), dtype=np.float64)
    return forecast, state


def cross_fitted_gains(forecast):
    """{unit: (gain [1000], offset [1000])}: least squares per parcel on the units of the other half."""
    units = sorted(forecast)
    half = {u: i % 2 for i, u in enumerate(units)}
    fitted = {}
    for h in (0, 1):
        t = {k: sum(forecast[u][k] for u in units if half[u] != h) for k in KEYS}
        var, cov = t["xx"] - t["x"] ** 2 / t["n"], t["xy"] - t["x"] * t["y"] / t["n"]
        gain = cov / var
        fitted[h] = (gain, (t["y"] - gain * t["x"]) / t["n"])
    return {u: fitted[half[u]] for u in units}


def row(length):
    name = f"brainvista {length} s"
    forecast, state = sums_from_forecasts(length)
    gains = cross_fitted_gains(forecast)
    stored_forecast, stored_updating = T.forecast(name, SPLITS), U.load(BV / "algonauts" / "updating", name)
    raw, new = {"Forecast": {}, "State": {}, "Updating": {}}, {"Forecast": {}, "State": {}, "Updating": {}}
    for unit, (a, b) in gains.items():
        s = forecast[unit]
        raw["Forecast"][unit] = ((s["yy"] - 2 * s["xy"] + s["xx"]).sum(), stored_forecast[unit][1])
        new["Forecast"][unit] = ((s["yy"] - 2 * a * s["xy"] - 2 * b * s["y"] + a * a * s["xx"] + 2 * a * b * s["x"] + s["n"] * b * b).sum(),
                                 stored_forecast[unit][1])
        if unit in state:
            d = state[unit]
            raw["State"][unit] = ((d["pp"] - 2 * d["py"] + d["yy"]).sum(), d["yy"].sum())
            new["State"][unit] = ((a * a * d["pp"] - 2 * a * d["py"] + d["yy"]).sum(), d["yy"].sum())
        e_raw = e_new = ref = 0.0
        for split in SPLITS:
            v = stored_updating[split]["4"].get(unit)
            if v is None:
                continue
            c = np.asarray(v["corr"], float)  # count, sum x, sum y, sum xy, sum xx, sum yy per parcel
            e_raw += (c[5] - 2 * c[3] + c[4]).sum()
            e_new += (c[5] - 2 * a * c[3] - 2 * b * c[2] + a * a * c[4] + 2 * a * b * c[1] + c[0] * b * b).sum()
            ref += float(v["ref"])
        raw["Updating"][unit], new["Updating"][unit] = (e_raw, ref), (e_new, ref)
    r = T.pearson(name, SPLITS)
    every = np.concatenate([a for a, _ in gains.values()])
    assert (every > 0).all(), "a negative gain would change r"
    spectator = SPECTATOR[length]
    minus = {"Forecast": T.diff(T.forecast(spectator, SPLITS), new["Forecast"]), "State": T.diff(T.state(spectator), new["State"]),
             "Updating": T.diff(T.updating(spectator, SPLITS), new["Updating"]), "r": T.diff_r(T.pearson(spectator, SPLITS), r)}
    return {"units": len(gains), "units in State": len(new["State"]),
            "gain: median, minimum, maximum": [round(float(v), 3) for v in (np.median(every), every.min(), every.max())],
            "as stored": {"Forecast": score(stored_forecast), "State": score(T.state(name)), "Updating": score(T.updating(name, SPLITS))},
            "recomputed without rescaling": {k: score(v) for k, v in raw.items()},
            "rescaled (the paper's row)": {**{k: score(v) for k, v in new.items()}, "r": round(sum(v[0] for v in r.values()) / sum(v[1] for v in r.values()), 4)},
            "Spectator minus BrainVista": {k: [round(v["value"], 4), round(v["low"], 4), round(v["high"], 4)] for k, v in minus.items()}}


if __name__ == "__main__":
    T.ROOTS = [*T.ROOTS, BV]
    out = A / "results-brainvista.json"
    results = json.loads(out.read_text()) if out.is_file() else {}
    for length in sys.argv[1:] or ("7.5", "120", "180"):
        results[length] = row(length)
        print(length, json.dumps(results[length]), flush=True)
        out.write_text(json.dumps(results, indent=1) + "\n")
