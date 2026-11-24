"""state_of_mind/paper/analysis.py (paper, Sec. 5): numbers and figures from the stored benchmark results and from state_of_mind/paper/collect.py's forecasts.

    python -m state_of_mind.paper.analysis benchmark     skill by seconds ahead (Forecast, State), Updating by number of new fMRI frames, on all test data
    python -m state_of_mind.paper.analysis collected     per-parcel maps, segment identification, the state/movie split, fast/slow shares, viewer and
                                           viewing readouts, time constants, the movie's impulse response
    python -m state_of_mind.paper.analysis isc           inter-viewer correlation per parcel, measured fMRI, first viewings
Writes $STATE_OF_MIND_OUTPUT/paper-analysis/{results-<part>.json, figs/*.pdf|png}.
"""
import json
import sys
import zlib

import numpy as np
from state_of_mind.paths import DATA, OUTPUT

A = OUTPUT / "paper-analysis"
FIGS = A / "figs"
TR, H = 1.49, 40
# Benchmark result stores under $STATE_OF_MIND_OUTPUT (downloaded from https://huggingface.co/stanjsx/Spectator).
# UPDATING_ROOTS searches benchmark-results-own-fill first so the linear forecaster's Updating uses own-forecast gap
# filling (matching the paper) whether the own-fill files are separate or merged into benchmark-results/.
ROOTS = [OUTPUT / "benchmark-results"]
UPDATING_ROOTS = [OUTPUT / "benchmark-results-own-fill", *ROOTS]
ROWS = {"Spectator": "ridge-jepa-180s-std-partial (latent)",
        "Linear forecaster, 180 s": "linear [context] 180 s",
        "Linear forecaster, 7.5 s": "linear [context] 7.5 s",
        "Stimulus only (TRIBE v2)": "TRIBEv2 zero-shot",
        "IPSID": "IPSID [context] 180 s",
        "NeuroWorld": "NeuroWorld k1 180s (latent)"}
SPLITS = ("val", "dev", "test")


def find(rel):
    return next((r / rel for r in ROOTS if (r / rel).exists()), None)


# ------------------------------------------------------------------------------------------------ stored benchmark results
def forecast_steps(name):
    """[H] err and ref sums per step, pooled over every test split; None if a split is missing."""
    from state_of_mind.benchmark import forecast as F
    err, ref = np.zeros(H), np.zeros(H)
    for split in SPLITS:
        path = find(f"algonauts/{split}/{name}/results.json")
        if path is None:
            return None
        for row in F.load(path.parent)["forecast"]:
            s = row["scores"]["P30"]
            err += np.nan_to_num(np.asarray(s["err"], float)[:H])
            ref += np.nan_to_num(np.asarray(s["ref"], float)[:H])
    return err, ref


def state_steps(name):
    from state_of_mind.benchmark import state as S
    for r in ROOTS:
        if (r / "algonauts/state" / f"{name}.npz").is_file():
            sums = S.load(r / "algonauts/state", name)
            if "dev" not in sums:
                continue
            err = sum(np.asarray(v["difference step|err"], float) for v in sums["dev"].values())
            energy = sum(np.asarray(v["difference step|energy"], float) for v in sums["dev"].values())
            return err, energy
    return None


def updating_budgets(name):
    from state_of_mind.benchmark import updating as U
    out = {}
    for split in SPLITS:
        found = None
        for r in UPDATING_ROOTS:
            if (r / "algonauts/updating" / f"{name}.npz").is_file():
                loaded = U.load(r / "algonauts/updating", name)
                if split in loaded:
                    found = loaded[split]
                    break
        if found is None:
            return None
        for budget, units in found.items():
            se, ref = out.get(budget, (0.0, 0.0))
            out[budget] = (se + sum(float(v["se"]) for v in units.values()), ref + sum(float(v["ref"]) for v in units.values()))
    return {b: 100 * (1 - se / ref) for b, (se, ref) in out.items()}


def benchmark():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {"forecast_by_step": {}, "state_by_step": {}, "updating_by_budget": {}}
    seconds = (np.arange(H) + 1) * TR
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for label, name in ROWS.items():
        f = forecast_steps(name)
        if f is not None:
            skill = 100 * (1 - f[0] / f[1])
            res["forecast_by_step"][label] = skill.round(2).tolist()
            axes[0].plot(seconds, skill, label=label, lw=2 if label == "Spectator" else 1)
        s = state_steps(name)
        if s is not None:
            sk = 100 * (1 - s[0] / s[1])
            res["state_by_step"][label] = sk.round(2).tolist()
            axes[1].plot((np.arange(len(sk)) + 1) * TR, sk, label=label, lw=2 if label == "Spectator" else 1)
        u = updating_budgets(name)
        if u is not None:
            res["updating_by_budget"][label] = {b: round(v, 2) for b, v in u.items()}
    order = ["0", "2", "4", "8", "16", "all"]
    for label, u in res["updating_by_budget"].items():
        xs = [b for b in order if b in u]
        axes[2].plot(range(len(xs)), [u[b] for b in xs], marker="o", label=label, lw=2 if label == "Spectator" else 1)
    axes[2].set_xticks(range(len(order)), order)
    for ax, title, xl in zip(axes, ("Forecast", "State", "Updating"), ("seconds ahead", "seconds ahead", "new fMRI frames in the minute")):
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("skill (%)"); ax.axhline(0, color="gray", lw=0.5)
    axes[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIGS / "properties.pdf"); fig.savefig(FIGS / "properties.png", dpi=150)
    # bands of State: first 7.5 s, 7.5-30 s, after
    for label, sk in res["state_by_step"].items():
        e, en = state_steps(ROWS[label])
        b = [(0, 5), (5, 20), (20, len(e))]
        res.setdefault("state_bands", {})[label] = [round(100 * (1 - e[a:z].sum() / en[a:z].sum()), 2) for a, z in b]
    (A / "results-benchmark.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("state_bands", "updating_by_budget")}, indent=1))


# ------------------------------------------------------------------------------------------------ collected forecasts
def load_runs(folder):
    out = {}
    for split in SPLITS:
        for f in sorted((A / folder / split).glob("*.npz")):
            out[(split, f.stem)] = dict(np.load(f, allow_pickle=True))
    return out


def parcel_skill(pred, obs, ok, mean):
    """Per-parcel err and ref sums over the scored frames."""
    m = ok[..., None]
    err = (((pred.astype(np.float32) - obs.astype(np.float32)) ** 2) * m).sum((0, 1))
    ref = (((mean[None, None] - obs.astype(np.float32)) ** 2) * m).sum((0, 1))
    return err, ref


def segment_id(pred, obs, ok, mean, origins, rng, candidates=19, apart=20):
    """Top-1 accuracy of picking each forecast minute's true measured minute among `candidates` other minutes of the same recording,
    at least `apart` frames away, by Pearson correlation over the frames scored in both."""
    hits, chance, n = 0, 0.0, 0
    p = pred.astype(np.float32) - mean[None, None]
    o = obs.astype(np.float32) - mean[None, None]
    for i in range(len(origins)):
        far = np.flatnonzero(np.abs(origins - origins[i]) >= apart)
        if not len(far):
            continue
        pick = rng.choice(far, min(candidates, len(far)), replace=False)
        cand = np.r_[i, pick]
        scores = []
        for j in cand:
            both = ok[i] & ok[j]
            if both.sum() < 5:
                scores.append(-np.inf)
                continue
            a, b = p[i][both].ravel(), o[j][both].ravel()
            a, b = a - a.mean(), b - b.mean()
            scores.append(float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12)))
        hits += int(np.argmax(scores) == 0)
        chance += 1 / len(cand)
        n += 1
    return hits, chance, n


def collected():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    spec = load_runs("spectator")
    keys = sorted(spec)
    others = {label: load_runs(folder) for label, folder in (("Transformer counterpart", "counterpart"), ("Linear forecaster, 180 s", "linear180"),
                                                             ("No fMRI (movie only)", "nofmri")) if (A / folder).is_dir()}
    # per-parcel skill and segment identification, on the same starting points
    maps, seg = {}, {}
    for label, runs in [("Spectator", spec)] + list(others.items()):
        err, ref = np.zeros(1000), np.zeros(1000)
        hits = chance = n = 0
        for k in keys:
            if k not in runs:
                continue
            d = spec[k]
            pred = runs[k]["pred"]
            if pred.shape != d["obs"].shape:
                continue
            e, r = parcel_skill(pred, d["obs"], d["ok"], d["mean"])
            err += e; ref += r
            h, c, m = segment_id(pred, d["obs"], d["ok"], d["mean"], d["origins"], np.random.default_rng(zlib.crc32(str(k[1]).encode())))
            hits += h; chance += c; n += m
        maps[label] = (100 * (1 - err / np.maximum(ref, 1e-9))).tolist()
        seg[label] = {"top-1 %": round(100 * hits / max(n, 1), 2), "chance %": round(100 * chance / max(n, 1), 2), "queries": n}
    res["segment_identification"] = seg
    res["parcel_skill_mean"] = {k: round(float(np.mean(v)), 2) for k, v in maps.items()}
    np.savez(A / "parcel-maps.npz", **{k: np.asarray(v) for k, v in maps.items()})
    # the state/movie split by horizon, and per parcel
    sum_s = np.zeros((H, 1000)); sq_s = np.zeros((H, 1000)); sum_m = np.zeros((H, 1000)); sq_m = np.zeros((H, 1000)); cnt = 0
    f_sum = np.zeros(1000); f_sq = np.zeros(1000); s_sum = np.zeros(1000); s_sq = np.zeros(1000)
    states, subj, view, movie, split_of = [], [], [], [], []
    for k in keys:
        d = spec[k]
        ys, ym = d["ys"].astype(np.float64), d["ym"].astype(np.float64)
        sum_s += ys.sum(0); sq_s += (ys ** 2).sum(0); sum_m += ym.sum(0); sq_m += (ym ** 2).sum(0); cnt += len(ys)
        f0, s0 = d["fast0"].astype(np.float64), d["slow0"].astype(np.float64)
        f_sum += f0.sum(0); f_sq += (f0 ** 2).sum(0); s_sum += s0.sum(0); s_sq += (s0 ** 2).sum(0)
        states.append(d["state0"]); n = len(d["state0"])
        subj += [str(d["subject"])] * n
        tl = str(d["timeline"])
        view += [2 if tl.endswith("_run-2") else 1] * n
        movie += [str(d["content"])] * n
        split_of += [k[0]] * n
    var_s = sq_s / cnt - (sum_s / cnt) ** 2
    var_m = sq_m / cnt - (sum_m / cnt) ** 2
    share_state_h = var_s.sum(1) / (var_s.sum(1) + var_m.sum(1))
    res["state_share_by_seconds"] = {f"{(h + 1) * TR:.1f}": round(float(share_state_h[h]), 3) for h in range(H)}
    share_state_parcel = (var_s / (var_s + var_m)).mean(0)
    var_f, var_sl = f_sq / cnt - (f_sum / cnt) ** 2, s_sq / cnt - (s_sum / cnt) ** 2
    slow_share = var_sl / (var_f + var_sl)
    np.savez(A / "share-maps.npz", state_share=share_state_parcel, slow_share=slow_share)
    names = [l.split("\t")[1] for l in open(DATA / "atlases/neuroworld-schaefer1000-v2/Schaefer2018_1000Parcels_7Networks_order.tsv").read().splitlines()[1:] if l.strip()]
    nets = np.array([n.split("_")[2] for n in names])
    res["by_network"] = {net: {"slow share": round(float(slow_share[nets == net].mean()), 3), "state share": round(float(share_state_parcel[nets == net].mean()), 3),
                               "Spectator skill": round(float(np.asarray(maps["Spectator"])[nets == net].mean()), 2)} for net in dict.fromkeys(nets)}
    # viewer identity and viewing (first or second) from the inferred state; trained on Friends S6, tested on Movie10
    X = np.concatenate(states); subj, view, split_of = np.array(subj), np.array(view), np.array(split_of)
    tr, te = split_of == "val", split_of != "val"
    out = {}
    for part, cols in (("whole state", slice(None)), ("fast part", slice(0, X.shape[1] // 2)), ("slow part", slice(X.shape[1] // 2, None))):
        sc = StandardScaler().fit(X[tr][:, cols])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr][:, cols]), subj[tr])
        out[part] = round(100 * float((clf.predict(sc.transform(X[te][:, cols])) == subj[te]).mean()), 2)
    res["viewer_identity_accuracy"] = {**out, "chance": round(100 / len(set(subj)), 2)}
    dev = split_of == "dev"
    movies = np.array(movie)[dev]
    accs = []
    for held in sorted(set(movies)):  # leave one Movie10 part out
        a, b = movies != held, movies == held
        if len(set(view[dev][a])) < 2 or len(set(view[dev][b])) < 2:
            continue
        sc = StandardScaler().fit(X[dev][a])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[dev][a]), view[dev][a])
        accs.append(float((clf.predict(sc.transform(X[dev][b])) == view[dev][b]).mean()))
    res["viewing_accuracy"] = {"first vs second viewing (leave one part out)": round(100 * float(np.mean(accs)), 2) if accs else None, "chance": 50.0, "parts": len(accs)}
    # time constants and the movie's impulse response
    eig = np.load(A / "spectator/eigenvalues.npy")
    tau = -TR / np.log(np.clip(np.abs(eig), 1e-9, 0.999999))
    res["time_constants"] = {"largest |eigenvalue|": round(float(np.abs(eig).max()), 4), "longest (s)": round(float(tau.max()), 1),
                             "modes over 30 s": int((tau > 30).sum()), "modes over 60 s": int((tau > 60).sum()), "modes": int(len(tau))}
    mats = np.load(A / "spectator/matrices.npz")
    M, Wm = mats["M"].astype(np.float64), mats["W"].astype(np.float64)
    probe = np.load(sorted((A / "spectator").glob("probe-*.npz"))[0])
    Wp = probe["W"].astype(np.float64)                         # [D, 1000]
    C = Wm.shape[1] // 5
    slots = [Wm[:, i * C:(i + 1) * C] for i in range(5)]       # slot 4 is the newest frame
    immediate = Wp.T @ sum(slots)                              # [1000, C]
    _, _, vt = np.linalg.svd(immediate, full_matrices=False)
    e = vt[0]                                                  # the movie direction that drives the cortex most
    steps = 25
    x = np.zeros(M.shape[0]); resp = []
    for k in range(steps):                                     # impulse at the frame predicted by step 0
        drive = slots[4 - k] @ e if k <= 4 else np.zeros(M.shape[0])
        x = M @ x + drive
        resp.append(x @ Wp)
    resp = np.array(resp)                                      # [steps, 1000]
    top = np.argsort(-np.abs(resp).max(0))[:50]
    course = (resp[:, top] * np.sign(resp[np.abs(resp[:, top]).argmax(0), top])).mean(1)
    res["impulse_response"] = {"seconds": ((np.arange(steps)) * TR).round(2).tolist(), "top-50 parcels, signed mean": course.round(4).tolist(),
                               "peak (s)": round(float(np.argmax(course) * TR), 2)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot((np.arange(H) + 1) * TR, 100 * share_state_h); axes[0].set_xlabel("seconds ahead"); axes[0].set_ylabel("share carried by the state (%)")
    axes[1].hist(tau[tau > 0], bins=np.logspace(0, 3, 40)); axes[1].set_xscale("log"); axes[1].set_xlabel("time constant (s)"); axes[1].set_ylabel("modes")
    axes[2].plot(np.arange(steps) * TR, course); axes[2].set_xlabel("seconds after a movie change"); axes[2].set_ylabel("predicted response")
    fig.tight_layout(); fig.savefig(FIGS / "inside.pdf"); fig.savefig(FIGS / "inside.png", dpi=150)
    (A / "results-collected.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "state_share_by_seconds"}, indent=1)[:4000])
    print("state share by seconds:", {k: v for i, (k, v) in enumerate(res["state_share_by_seconds"].items()) if i in (0, 4, 9, 19, 39)})


# ------------------------------------------------------------------------------------------------ inter-viewer correlation
def isc():
    from state_of_mind.benchmark.data import BenchmarkData
    data = BenchmarkData("state_of_mind/configs/algonauts/base.yaml")
    by_tl = {}
    for split in SPLITS:
        for run in data.runs(split):
            if run.timeline_id.endswith("_run-2"):
                continue
            ok = run.brain_ok & run.support
            by_tl.setdefault(run.content, []).append((np.asarray(run.brain, np.float32), ok))  # the movie segment, across viewers
    num, den = np.zeros(1000), 0
    for tl, items in by_tl.items():
        if len(items) < 2:
            continue
        n = min(len(b) for b, _ in items)
        ok = np.all([o[:n] for _, o in items], 0)
        if ok.sum() < 50:
            continue
        ys = [b[:n][ok] for b, _ in items]
        for i, y in enumerate(ys):
            rest = np.mean([z for j, z in enumerate(ys) if j != i], 0)
            a, b = y - y.mean(0), rest - rest.mean(0)
            r = (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12)
            num += r * len(y); den += len(y)
    isc_map = num / den
    np.save(A / "isc.npy", isc_map)
    out = {"mean ISC": round(float(isc_map.mean()), 3), "timelines": len(by_tl)}
    shares = A / "share-maps.npz"
    if shares.is_file():
        s = np.load(shares)
        out["corr(ISC, state share)"] = round(float(np.corrcoef(isc_map, s["state_share"])[0, 1]), 3)
        out["corr(ISC, slow share)"] = round(float(np.corrcoef(isc_map, s["slow_share"])[0, 1]), 3)
    (A / "results-isc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ cortical maps
def surface_values(values):
    """Per-vertex values on fsaverage (164k) for each hemisphere from the 1,000 Schaefer parcels: the bundle's MNI atlas projected to the
    surface (nearest most frequent label between the white and pial surfaces); vertices outside every parcel are NaN."""
    import nibabel as nib
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    atlas = nib.load(DATA / "atlases/neuroworld-schaefer1000-v2/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
    out = {}
    for hemi, side in (("L", "left"), ("R", "right")):
        cache = A / f"labels-fsaverage-{hemi}.npy"
        if cache.is_file():
            labels = np.load(cache)
        else:
            labels = surface.vol_to_surf(atlas, fs[f"pial_{side}"], inner_mesh=fs[f"white_{side}"], interpolation="nearest_most_frequent").astype(int)
            from scipy.spatial import cKDTree  # fill small holes from the nearest labelled vertex within 3 mm; the medial wall stays empty
            coords = surface.load_surf_mesh(fs[f"white_{side}"]).coordinates
            have, miss = np.flatnonzero(labels > 0), np.flatnonzero(labels == 0)
            dist, idx = cKDTree(coords[have]).query(coords[miss])
            near = dist < 3.0
            labels[miss[near]] = labels[have[idx[near]]]
            np.save(cache, labels)
        v = np.full(len(labels), np.nan)
        inside = labels > 0
        v[inside] = np.asarray(values)[labels[inside] - 1]
        out[hemi] = v
    return out


def maps():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting
    spec = load_runs("spectator")
    short = load_runs("spectator7.5")
    err, ref, err7, ref7 = np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)
    for k, d in spec.items():
        if k not in short:
            continue
        e, r = parcel_skill(d["pred"], d["obs"], d["ok"], d["mean"]); err += e; ref += r
        e7, r7 = parcel_skill(short[k]["pred"], d["obs"], d["ok"], d["mean"]); err7 += e7; ref7 += r7
    skill, skill7 = 100 * (1 - err / ref), 100 * (1 - err7 / ref7)
    gain = skill - skill7
    shares = np.load(A / "share-maps.npz")
    isc_map = np.load(A / "isc.npy")
    names = [l.split("\t")[1] for l in open(DATA / "atlases/neuroworld-schaefer1000-v2/Schaefer2018_1000Parcels_7Networks_order.tsv").read().splitlines()[1:] if l.strip()]
    nets = np.array([n.split("_")[2] for n in names])
    res = {"mean history gain (180 s minus 7.5 s), parcels": round(float(gain.mean()), 2),
           "corr(history gain, ISC)": round(float(np.corrcoef(gain, isc_map)[0, 1]), 3),
           "corr(history gain, state share)": round(float(np.corrcoef(gain, shares["state_share"])[0, 1]), 3),
           "corr(state share, ISC)": round(float(np.corrcoef(shares["state_share"], isc_map)[0, 1]), 3),
           "by network": {n: {"history gain": round(float(gain[nets == n].mean()), 2), "ISC": round(float(isc_map[nets == n].mean()), 3),
                              "state share": round(float(shares["state_share"][nets == n].mean()), 3)} for n in dict.fromkeys(nets)}}
    np.savez(A / "cortex-maps.npz", skill=skill, skill7=skill7, gain=gain, isc=isc_map, state_share=shares["state_share"], slow_share=shares["slow_share"])
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    panels = [("Share carried by the inferred state", shares["state_share"], "viridis"), ("Agreement between viewers (ISC)", isc_map, "magma"),
              ("Forecast skill (%)", skill, "inferno"), ("Gain from 180 s over 7.5 s of history (points)", gain, "RdBu_r")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(6, 2.4 * len(panels)), subplot_kw={"projection": "3d"})
    for row, (title, values, cmap) in enumerate(panels):
        sv = surface_values(values)
        lo, hi = np.nanpercentile(values, 2), np.nanpercentile(values, 98)
        if cmap == "RdBu_r":
            hi = max(abs(lo), abs(hi)); lo = -hi
        for col, view in enumerate(("lateral", "medial")):
            plotting.plot_surf(fs["infl_left"], sv["L"], hemi="left", view=view, cmap=cmap, vmin=lo, vmax=hi, bg_map=fs["sulc_left"],
                               bg_on_data=True, axes=axes[row, col], colorbar=col == 1, darkness=0.5)
        axes[row, 0].set_title(title, fontsize=9, loc="left")
    fig.savefig(FIGS / "cortex.png", dpi=150, bbox_inches="tight"); fig.savefig(FIGS / "cortex.pdf", bbox_inches="tight")
    (A / "results-maps.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):  # this script parses no options: show its usage and run nothing
        sys.exit(print(__doc__))
    FIGS.mkdir(parents=True, exist_ok=True)
    {"benchmark": benchmark, "collected": collected, "isc": isc, "maps": maps}[sys.argv[1]]()
