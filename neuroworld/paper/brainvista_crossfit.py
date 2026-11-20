"""neuroworld/paper/brainvista_crossfit.py: BrainVista's numbers for the appendix, with its forecasts rescaled as in Table 2 (the
version that favours BrainVista): one gain and one offset per parcel, shared by the viewers, cross-fitted over two
halves of the 66 evaluation units (paper-analysis/extra-nov14/bv/bv_crossfit.py; the gains here are fitted the same way).

  per length (7.5, 120, 180 s)   Pearson and global r over frames 1-20 and 1-40, Cls@Top10 (standard_metrics.py's definitions),
                                 for the appendix table of the metrics of prior work
  at 180 s                       Forecast and State per step, Updating per number of new frames, for the appendix figure

The pooled Forecast, State and Updating (4 new frames) must equal bv_crossfit_<L>.json, which Table 2 reports.
Output: paper-analysis/results-brainvista-crossfit.json and results-brainvista-steps-180s.npz.

    PYTHONPATH=<NeuralWorld> python -m neuroworld.paper.brainvista_crossfit [--device cuda]
"""
import collections
import json
import sys

import numpy as np

from neuroworld.benchmark import forecast as F, updating as U
from neuroworld.benchmark.metrics import MIN_ROWS, step_scores

from neuroworld.paper import standard_metrics as SM  # global_r, _normalize, cls_hits, SEG
from neuroworld.paths import EXTERNAL, OUTPUT

BV = EXTERNAL / "baselines/history-length-brainvista"
FC = EXTERNAL / "baselines/pred/forecasts"
CROSSFIT = OUTPUT / "paper-analysis/extra-nov14/bv"
A = OUTPUT / "paper-analysis"
SPLITS, TR, H = ("val", "dev", "test"), 1.49, 40
KEYS = ("n", "x", "y", "xx", "xy", "yy")


def runs(L):
    for split in SPLITS:
        index = json.loads((FC / f"brainvista_{L}_s" / split / "index.json").read_text())["runs"]
        for run_id, meta in index.items():
            f = np.load(FC / f"brainvista_{L}_s" / split / meta["file"])
            o = np.load(FC / "_observed" / split / meta["observed"])
            yield split, run_id, meta, f["forecast"].astype(np.float32), o["observed"].astype(np.float32), o["ok"], f["origins"]


def one_length(L, device, detail):
    name = f"brainvista {L} s"
    # pass 1: per-unit, per-step, per-parcel sums (the gains use their totals over the steps, as bv_crossfit.py)
    fsum = collections.defaultdict(lambda: {k: np.zeros((H, 1000)) for k in KEYS})
    ssum = collections.defaultdict(lambda: {k: np.zeros((H, 1000)) for k in ("pp", "py", "yy")})
    dev = {}
    for split, run_id, meta, pred, obs, ok, origins in runs(L):
        w = ok[..., None].astype(np.float32)
        x, y = pred * w, obs * w
        s = fsum[meta["unit"]]
        s["n"] += np.repeat(w.sum(0), 1000, axis=1)[:H]
        for k, v in (("x", x), ("y", y), ("xx", x * x), ("xy", x * y), ("yy", y * y)):
            s[k] += v.sum(0, dtype=np.float64)[:H]
        if split == "dev":
            dev[run_id] = (pred, obs, ok, origins, meta["unit"])
    for run_id, (p1, y1, k1, o1, unit) in dev.items():
        other = run_id[:-1] + "2"
        if not run_id.endswith("run-01") or other not in dev:
            continue
        p2, y2, k2, o2, _ = dev[other]
        common = np.intersect1d(o1, o2)
        i1, i2 = np.searchsorted(o1, common), np.searchsorted(o2, common)
        both = (k1[i1] & k2[i2])[..., None].astype(np.float32)
        dp, dy = (p1[i1] - p2[i2]) * both, (y1[i1] - y2[i2]) * both
        s = ssum[unit]
        s["pp"] += (dp * dp).sum(0, dtype=np.float64)[:H]; s["py"] += (dp * dy).sum(0, dtype=np.float64)[:H]
        s["yy"] += (dy * dy).sum(0, dtype=np.float64)[:H]
    units = sorted(fsum)
    half = {u: i % 2 for i, u in enumerate(units)}
    gains = {}
    for h in (0, 1):  # the gains of half h come from the other half
        t = {k: sum(fsum[u][k].sum(0) for u in units if half[u] != h) for k in KEYS}
        var, cov = t["xx"] - t["x"] ** 2 / t["n"], t["xy"] - t["x"] * t["y"] / t["n"]
        a = cov / var
        gains[h] = (a, (t["y"] - a * t["x"]) / t["n"])
    # pooled task scores, per step / per budget
    stored = {}
    for split in SPLITS:
        for row in F.load(BV / "algonauts" / split / name)["forecast"]:
            stored[row.get("run_id") or row.get("run")] = (row["unit"], np.nan_to_num(np.asarray(row["scores"]["P30"]["ref"], float)[:H]))
    f_ref = np.zeros(H)
    for unit, ref in stored.values():
        f_ref += ref
    f_err, s_err, s_energy = np.zeros(H), np.zeros(H), np.zeros(H)
    for u in units:
        a, b = gains[half[u]]
        s = fsum[u]
        f_err += (s["yy"] - 2 * a * s["xy"] - 2 * b * s["y"] + a * a * s["xx"] + 2 * a * b * s["x"] + s["n"] * b * b).sum(1)
        if u in ssum:
            d = ssum[u]
            s_err += (a * a * d["pp"] - 2 * a * d["py"] + d["yy"]).sum(1); s_energy += d["yy"].sum(1)
    sums = U.load(BV / "algonauts" / "updating", name)
    budgets = sorted({bud for split in SPLITS for bud in sums[split]}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    up = {bud: [0.0, 0.0] for bud in budgets}
    for split in SPLITS:
        for bud in budgets:
            for unit, v in sums[split].get(bud, {}).items():
                a, b = gains[half[unit]]
                c = np.asarray(v["corr"], float)
                up[bud][0] += (c[5] - 2 * a * c[3] - 2 * b * c[2] + a * a * c[4] + 2 * a * b * c[1] + c[0] * b * b).sum()
                up[bud][1] += float(v["ref"])
    skill = lambda e, r: 100 * (1 - e / r)
    pooled = {"Forecast": skill(f_err.sum(), f_ref.sum()), "State": skill(s_err.sum(), s_energy.sum()),
              "Updating": skill(*up["4"])}
    table2 = json.loads((CROSSFIT / f"bv_crossfit_{L}.json").read_text())["rescaled"]
    print(L, "pooled", {k: round(v, 2) for k, v in pooled.items()}, "Table 2 (bv_crossfit):", table2, flush=True)
    for k in pooled:
        assert abs(pooled[k] - table2[k]) < 0.01, (k, pooled[k], table2[k])
    # pass 2: the metrics of prior work on the rescaled forecasts
    pearson, glob, segments = [], [], {}
    for split, run_id, meta, pred, obs, ok, origins in runs(L):
        a, b = gains[half[meta["unit"]]]
        p = (pred * a + b).astype(np.float32)
        pearson.append(step_scores(p, obs, ok, origins, TR)["pearson"])
        glob.append(SM.global_r(p, obs, ok, MIN_ROWS))
        full = ok[:, :SM.SEG].all(1)
        if full.any():
            n = int(full.sum())
            seg = segments.setdefault(meta["subject"], {"p": [], "t": [], "content": []})
            seg["p"].append(SM._normalize(p[full, :SM.SEG].reshape(n, -1)).astype(np.float32))
            seg["t"].append(SM._normalize(obs[full, :SM.SEG].reshape(n, -1)).astype(np.float32))
            seg["content"] += [meta["content"]] * n
    hits = []
    for subject, seg in sorted(segments.items()):
        h, _ = SM.cls_hits(np.concatenate(seg["p"]), np.concatenate(seg["t"]), seg["content"], device)
        hits.append(h)
    pearson, glob = np.asarray(pearson), np.asarray(glob)
    m = lambda x, s=slice(None): float(np.nanmean(np.nanmean(x[:, s], 1)))
    metrics = {"Pearson 1-20": m(pearson, slice(0, SM.SEG)), "Pearson 1-40": m(pearson), "global r 1-20": m(glob, slice(0, SM.SEG)),
               "global r 1-40": m(glob), "Cls@Top10": float(np.concatenate(hits).mean()), "queries": int(sum(len(h) for h in hits))}
    print(L, {k: round(v, 4) for k, v in metrics.items()}, flush=True)
    if detail:
        np.savez(A / "results-brainvista-steps-180s.npz", forecast_err=f_err, forecast_ref=f_ref, state_err=s_err, state_energy=s_energy,
                 budgets=np.array(budgets), updating=np.array([skill(*up[bud]) for bud in budgets]))
    return {"pooled": pooled, "metrics": metrics, "gain: median, minimum, maximum":
            [float(np.median(np.r_[gains[0][0], gains[1][0]])), float(min(gains[0][0].min(), gains[1][0].min())),
             float(max(gains[0][0].max(), gains[1][0].max()))]}


def main():
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    out = {L: one_length(L, device, L == "180") for L in ("7.5", "120", "180")}
    (A / "results-brainvista-crossfit.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", A / "results-brainvista-crossfit.json", flush=True)


if __name__ == "__main__":
    main()
