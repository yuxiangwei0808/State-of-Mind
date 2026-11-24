"""state_of_mind/paper/decomposition.py (paper, Sec. 5.5 and Appendix D): post-hoc analyses of Spectator's exact linear
decomposition into initial-state and stimulus terms (produced by ``python -m state_of_mind.paper.collect spectator``).

    python -m state_of_mind.paper.decomposition swap-table      3x3 state x stimulus swap table (App. D.1)
    python -m state_of_mind.paper.decomposition state-swap      cross-viewer / cross-viewing / cross-time state swaps (Fig. 3b-c)
    python -m state_of_mind.paper.decomposition parcel-benefit  per-parcel state & stimulus Forecast gains vs ISC (Fig. 4)
    python -m state_of_mind.paper.decomposition gradshap        exact GradientSHAP input-parcel attribution (App. D.2)

All subcommands default to reading ``$STATE_OF_MIND_OUTPUT/paper-analysis/spectator/`` and writing JSON/NPZ outputs into
``$STATE_OF_MIND_OUTPUT/paper-analysis/``. For the exact starting points reported in Sec. 5.5 / App. D, collect with
``--every 1``.
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path

import numpy as np

from state_of_mind.paths import DATA, OUTPUT

A = OUTPUT / "paper-analysis"
DEFAULT_SRC = A / "spectator"
DEFAULT_MODEL = OUTPUT / "spectator-v1/algonauts/model"
ATLAS_TSV = DATA / "atlases/neuroworld-schaefer1000-v2/Schaefer2018_1000Parcels_7Networks_order.tsv"
BUNDLE = DATA / "algonauts/derivatives/neuroworld-schaefer1000-v2"
VIEWERS = ("Algonauts2025/01", "Algonauts2025/02", "Algonauts2025/03", "Algonauts2025/05")
NETS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
BANDS = {"all": slice(0, 40), "0-7.5s": slice(0, 5), "7.5-30s": slice(5, 20), "30-60s": slice(20, 40)}


def _parcel_networks(atlas_tsv=ATLAS_TSV):
    lines = [l for l in Path(atlas_tsv).read_text().splitlines()[1:] if l.strip()]
    names = [l.split("\t")[1] for l in lines]
    nets = np.array([n.split("_")[2] for n in names])
    return names, nets


# ---------------------------------------------------------------------------------------- Table 9: 3x3 state x stimulus

def swap_table(args):
    src = Path(args.src)
    out = Path(args.out or A / "results-swap-table.json")
    rng = np.random.default_rng(0)
    bias = {p.name[len("probe-"):-4].replace("__", "/"): np.load(p)["b"].astype(np.float32)
            for p in src.glob("probe-*.npz")}
    states, movies = ("own", "none", "other"), ("own", "none", "other")
    sums = {(s, m): collections.defaultdict(lambda: [0.0, 0.0]) for s in states for m in movies}
    points = 0
    for fp in sorted(glob.glob(str(src / "*" / "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        y, ok, mean = z["obs"].astype(np.float32), z["ok"], z["mean"].astype(np.float32)
        ys, ym = z["ys"].astype(np.float32), z["ym"].astype(np.float32)
        b, content, n = bias[str(z["subject"])], str(z["content"]), len(y)
        if n < 4:
            continue
        shift = np.roll(np.arange(n), n // 2)
        state = {"own": ys, "none": 0.0, "other": ys[shift]}
        movie = {"own": ym, "none": b, "other": ym[shift]}
        ref = float((((y - mean) ** 2).sum(-1) * ok).sum())
        for s in states:
            for m in movies:
                e = sums[(s, m)][content]
                e[0] += float((((y - (state[s] + movie[m])) ** 2).sum(-1) * ok).sum())
                e[1] += ref
        points += n
    contents = sorted(sums[("own", "own")])
    draws = rng.integers(0, len(contents), size=(2000, len(contents)))
    res = {"starting points": points, "content parts": len(contents), "table": {}}
    for (s, m), by in sums.items():
        e = np.array([by[c][0] for c in contents])
        r = np.array([by[c][1] for c in contents])
        boot = 100 * (1 - e[draws].sum(1) / r[draws].sum(1))
        res["table"][f"state {s}, movie {m}"] = [
            round(float(100 * (1 - e.sum() / r.sum())), 1),
            round(float(np.percentile(boot, 2.5)), 1),
            round(float(np.percentile(boot, 97.5)), 1),
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------- Fig. 3b-c: state swaps

def state_swap(args):
    src = Path(args.src)
    out = Path(args.out or A / "results-state-swap.json")
    rng = np.random.default_rng(0)
    M = np.load(src / "matrices.npz")["M"].astype(np.float32)
    MT = np.ascontiguousarray(M.T)
    probe = {}
    for v in VIEWERS:
        z = np.load(src / f"probe-{v.replace('/', '__')}.npz")
        probe[v] = (z["W"].astype(np.float32), z["b"].astype(np.float32))

    groups = collections.defaultdict(dict)
    for fp in sorted(glob.glob(str(src / "*" / "*.npz"))):
        base = os.path.basename(fp)[:-4]
        viewer, rec = base[:len("Algonauts2025__01")].replace("__", "/"), base[len("Algonauts2025__01__"):]
        groups[rec][viewer] = fp

    sums, counts = {v: 0.0 for v in VIEWERS}, {v: 0 for v in VIEWERS}
    for files in groups.values():
        for v, fp in files.items():
            s = np.load(fp, allow_pickle=True)["state0"]
            sums[v], counts[v] = sums[v] + s.sum(0, dtype=np.float64), counts[v] + len(s)
    mean_state = {v: (sums[v] / counts[v]).astype(np.float32) for v in VIEWERS}

    def readout(states, viewer):
        W = probe[viewer][0]
        x, parts = states, []
        for _ in range(40):
            x = x @ MT
            parts.append(x @ W)
        return np.stack(parts, 1)

    def add(acc, cond, content, pred, y, ok, mean):
        err = ((y - pred) ** 2).sum(-1) * ok
        ref = ((y - mean) ** 2).sum(-1) * ok
        for band, sl in BANDS.items():
            a = acc[cond][content][band]
            a[0] += float(err[:, sl].sum())
            a[1] += float(ref[:, sl].sum())

    new_acc = lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0])))
    main, twice = new_acc(), new_acc()
    largest, n_points, n_twice = 0.0, 0, 0
    loaded = {}

    def load(fp):
        if fp not in loaded:
            z = np.load(fp, allow_pickle=True)
            loaded[fp] = {k: z[k] for k in ("state0", "ym", "ys", "obs", "ok", "mean", "origins", "content", "pred")}
        return loaded[fp]

    for rec, files in sorted(groups.items()):
        if len(files) < 4:
            continue
        data = {v: load(fp) for v, fp in files.items()}
        common = sorted(set.intersection(*[set(d["origins"].tolist()) for d in data.values()]))
        if len(common) < 4:
            continue
        idx = {v: np.array([np.flatnonzero(d["origins"] == o)[0] for o in common]) for v, d in data.items()}
        n = len(common)
        shift = np.roll(np.arange(n), n // 2)
        for v in VIEWERS:
            d, i = data[v], idx[v]
            y, ok, mean = d["obs"][i].astype(np.float32), d["ok"][i], d["mean"].astype(np.float32)
            ym, content = d["ym"][i].astype(np.float32), str(d["content"])
            own = d["state0"][i]
            own_pred = readout(own, v) + ym
            largest = max(largest, float(np.abs(own_pred - d["pred"][i].astype(np.float32)).max()))
            add(main, "own", content, own_pred, y, ok, mean)
            add(main, "zero_state", content, ym, y, ok, mean)
            add(main, "mean_state", content, readout(np.repeat(mean_state[v][None], n, 0), v) + ym, y, ok, mean)
            add(main, "other_moment", content, readout(own[shift], v) + ym, y, ok, mean)
            others = [data[b]["state0"][idx[b]] for b in VIEWERS if b != v]
            add(main, "others_mean", content, readout(np.mean(others, 0), v) + ym, y, ok, mean)
            for s in others:
                add(main, "other_viewer", content, readout(s, v) + ym, y, ok, mean)
                add(main, "other_both", content, readout(s[shift], v) + ym, y, ok, mean)
            n_points += n
        loaded.clear()

    for rec, files in sorted(groups.items()):
        if "run-01" not in rec or not ("figures" in rec or "life" in rec):
            continue
        rec2 = rec.replace("run-01", "run-02")
        for v in VIEWERS:
            if v not in files or v not in groups.get(rec2, {}):
                continue
            a, b = load(files[v]), load(groups[rec2][v])
            common = sorted(set(a["origins"].tolist()) & set(b["origins"].tolist()))
            if not common:
                continue
            ia = np.array([np.flatnonzero(a["origins"] == o)[0] for o in common])
            ib = np.array([np.flatnonzero(b["origins"] == o)[0] for o in common])
            for d, i, e, j in ((a, ia, b, ib), (b, ib, a, ia)):
                y, ok, mean = d["obs"][i].astype(np.float32), d["ok"][i], d["mean"].astype(np.float32)
                ym, content = d["ym"][i].astype(np.float32), str(d["content"])
                add(twice, "own", content, readout(d["state0"][i], v) + ym, y, ok, mean)
                add(twice, "other_viewing", content, readout(e["state0"][j], v) + ym, y, ok, mean)
                add(twice, "zero_state", content, ym, y, ok, mean)
                n_twice += len(common)
        loaded.clear()

    def scores(acc, reference="own"):
        contents = sorted(acc[reference])
        table = {}
        draws = rng.integers(0, len(contents), size=(2000, len(contents)))
        for cond in acc:
            table[cond] = {}
            for band in BANDS:
                e = np.array([acc[cond][c][band][0] for c in contents])
                r = np.array([acc[cond][c][band][1] for c in contents])
                e0 = np.array([acc[reference][c][band][0] for c in contents])
                r0 = np.array([acc[reference][c][band][1] for c in contents])
                value = 100 * (1 - e.sum() / r.sum())
                diff = 100 * ((1 - e[draws].sum(1) / r[draws].sum(1)) - (1 - e0[draws].sum(1) / r0[draws].sum(1)))
                table[cond][band] = {
                    "score": round(value, 2),
                    "minus_own": round(value - 100 * (1 - e0.sum() / r0.sum()), 2),
                    "interval": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)],
                }
        return table, len(contents)

    result = {"starting_points": n_points, "largest_difference_own_vs_saved": largest}
    result["all_recordings"], result["content_parts"] = scores(main)
    result["watched_twice_points"] = n_twice
    result["watched_twice"], result["watched_twice_content_parts"] = scores(twice)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    for name in ("all_recordings", "watched_twice"):
        print(name)
        for cond, bands in result[name].items():
            print(f"  {cond:14s} " + "  ".join(
                f"{b}: {bands[b]['score']:6.2f} ({bands[b]['minus_own']:+.2f} [{bands[b]['interval'][0]:+.2f}, {bands[b]['interval'][1]:+.2f}])"
                for b in BANDS))


# ---------------------------------------------------------------------------------------- Fig. 4: per-parcel gains

def parcel_benefit(args):
    src = Path(args.src)
    out = Path(args.out or A / "results-parcel-benefit.json")
    bias = {p.name[len("probe-"):-4].replace("__", "/"): np.load(p)["b"].astype(np.float32)
            for p in src.glob("probe-*.npz")}
    names = ("full", "movie_term", "state_term")
    err = {k: np.zeros(1000) for k in names}
    ref = np.zeros(1000)
    sub_bands = {"0-7.5s": slice(0, 5), "7.5-30s": slice(5, 20), "30-60s": slice(20, 40)}
    err_band = {b: {k: np.zeros(1000) for k in names} for b in sub_bands}
    ref_band = {b: np.zeros(1000) for b in sub_bands}
    for fp in sorted(glob.glob(str(src / "*" / "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        y, ok, mean = z["obs"].astype(np.float32), z["ok"][..., None], z["mean"].astype(np.float32)
        preds = {"full": z["pred"].astype(np.float32), "movie_term": z["ym"].astype(np.float32),
                 "state_term": z["ys"].astype(np.float32) + bias[str(z["subject"])]}
        r = ((y - mean) ** 2) * ok
        ref += r.sum((0, 1))
        for b, sl in sub_bands.items():
            ref_band[b] += r[:, sl].sum((0, 1))
        for k, p in preds.items():
            e = ((y - p) ** 2) * ok
            err[k] += e.sum((0, 1))
            for b, sl in sub_bands.items():
                err_band[b][k] += e[:, sl].sum((0, 1))
    skill = {k: 100 * (1 - err[k] / ref) for k in names}
    benefit = {"state": skill["full"] - skill["movie_term"], "movie": skill["full"] - skill["state_term"]}
    isc = np.load(args.isc) if args.isc else (np.load(A / "isc.npy") if (A / "isc.npy").is_file() else None)
    shares_path = Path(args.shares) if args.shares else A / "share-maps.npz"
    share = np.load(shares_path)["state_share"] if shares_path.is_file() else None
    c = lambda a, b: round(float(np.corrcoef(a, b)[0, 1]), 3)
    res = {
        "pooled": {k: round(float(100 * (1 - err[k].sum() / ref.sum())), 2) for k in names},
        "mean over parcels": {k: round(float(v.mean()), 2) for k, v in {**skill, **{f"benefit of the {a}": b for a, b in benefit.items()}}.items()},
        "parcels where state benefit > movie benefit (%)": round(float(100 * (benefit["state"] > benefit["movie"]).mean()), 1),
        "state benefit by seconds ahead, mean over parcels": {
            b: round(float((100 * (err_band[b]["movie_term"] - err_band[b]["full"]) / ref_band[b]).mean()), 2) for b in sub_bands},
    }
    if isc is not None:
        corr = {"state benefit vs ISC": c(benefit["state"], isc), "movie benefit vs ISC": c(benefit["movie"], isc),
                "full skill vs ISC": c(skill["full"], isc), "state-term skill vs ISC": c(skill["state_term"], isc),
                "movie-term skill vs ISC": c(skill["movie_term"], isc)}
        if share is not None:
            corr["state benefit vs model's state share"] = c(benefit["state"], share)
            corr["movie benefit vs model's state share"] = c(benefit["movie"], share)
        res["correlations across parcels"] = corr
        res["state benefit vs ISC by seconds ahead"] = {
            b: c(100 * (err_band[b]["movie_term"] - err_band[b]["full"]) / ref_band[b], isc) for b in sub_bands}
    if Path(args.atlas_tsv).is_file():
        _, nets = _parcel_networks(args.atlas_tsv)
        res["by network"] = {
            n: {
                "full": round(float(skill["full"][nets == n].mean()), 1),
                "state benefit": round(float(benefit["state"][nets == n].mean()), 1),
                "movie benefit": round(float(benefit["movie"][nets == n].mean()), 1),
                **({"ISC": round(float(isc[nets == n].mean()), 3)} if isc is not None else {}),
                **({"model's state share": round(float(share[nets == n].mean()), 3)} if share is not None else {}),
            }
            for n in NETS
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), **{f"skill_{k}": v for k, v in skill.items()},
             benefit_state=benefit["state"], benefit_movie=benefit["movie"])
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------- Fig. 7: GradientSHAP

def gradshap(args):
    import torch
    from state_of_mind.train.readout import load_world_model
    from state_of_mind.utils.config import load_config

    src = Path(args.src)
    model_dir = Path(args.model_dir)
    bundle = Path(args.bundle)
    out = Path(args.out or A / "results-gradshap.npz")
    rng = np.random.default_rng(0)

    model, _ = load_world_model(load_config(model_dir / "config.yaml"), model_dir / "best.pt", "cpu")
    model.eval()
    J = torch.autograd.functional.jacobian(lambda w: model.encode(w[None])[0, -1], torch.zeros(120, 1000), vectorize=True)
    M = torch.from_numpy(np.load(src / "matrices.npz")["M"].astype(np.float32))
    Pm = torch.from_numpy(np.mean([np.load(fp)["W"] for fp in sorted(src.glob("probe-*.npz"))], 0).astype(np.float32))

    wins = []
    files = sorted(glob.glob(str(src / "*" / "*.npz")))
    rng.shuffle(files)
    for fp in files[:120]:
        z = np.load(fp, allow_pickle=True)
        base = os.path.basename(fp)[:-4].split("__")
        raw = bundle / "fmri/projected_native" / f"Algonauts2025%2F{base[1]}%2F{base[2]}%2F{base[3]}%2F{base[4]}.npy"
        if not raw.is_file():
            continue
        x = np.load(raw).astype(np.float32)
        valid = np.load(str(raw).replace("fmri/projected_native", "masks/valid_time"))
        wins += [x[o - 120:o] for o in z["origins"] if o >= 120 and valid[o - 120:o].all()]
    wins = torch.from_numpy(np.stack(wins))
    pairs = args.pairs
    a = rng.integers(0, len(wins), pairs)
    b = rng.integers(0, len(wins), pairs)
    parcel_names, nets = _parcel_networks(args.atlas_tsv) if Path(args.atlas_tsv).is_file() else (None, None)

    res, arrays = {"windows": int(len(wins)), "pairs": pairs}, {}
    Mk = torch.eye(2048)
    steps = {1: "1.5 s", 10: "15 s", 40: "60 s"}
    k_done = 0
    for k, name in steps.items():
        for _ in range(k - k_done):
            Mk = M @ Mk
        k_done = k
        G = torch.einsum("do,dli->oli", (Mk.T @ Pm), J)
        imp, self_att, total_att = torch.zeros(1000), torch.zeros(1000), torch.zeros(1000)
        with torch.no_grad():
            for s0 in range(0, pairs, 10):
                d = wins[a[s0:s0 + 10]] - wins[b[s0:s0 + 10]]
                att = torch.einsum("oli,mli->moi", G, d).abs()
                imp += att.sum(1).sum(0)
                self_att += torch.diagonal(att, dim1=1, dim2=2).sum(0)
                total_att += att.sum(2).sum(0)
        imp = (imp / pairs).numpy()
        arrays[f"importance_{k}"] = imp
        ipn = imp / imp.mean()
        entry = {
            "5th, 95th percentile across regions": [round(float(x), 2) for x in np.percentile(ipn, [5, 95])],
            "share of a region's forecast attributed to the same region (mean)": round(float((self_att / total_att).mean()), 3),
        }
        if nets is not None:
            entry["importance by network (1 = cortex average)"] = {n: round(float(ipn[nets == n].mean()), 2) for n in NETS}
            entry["top 10 regions"] = [parcel_names[i] for i in np.argsort(-imp)[:10]]
        res[name] = entry
        del G
    corr = lambda u, v: round(float(np.corrcoef(u, v)[0, 1]), 3)
    res["correlation between horizons"] = {
        "1.5 s vs 15 s": corr(arrays["importance_1"], arrays["importance_10"]),
        "1.5 s vs 60 s": corr(arrays["importance_1"], arrays["importance_40"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, desc in (
        ("swap-table", swap_table, "3x3 state x stimulus swap table (App. D.1)"),
        ("state-swap", state_swap, "Cross-viewer / cross-viewing / cross-time state swaps (Fig. 3b-c)"),
        ("parcel-benefit", parcel_benefit, "Per-parcel state and stimulus Forecast gains vs ISC (Fig. 4)"),
        ("gradshap", gradshap, "Exact GradientSHAP input-parcel attribution (App. D.2)"),
    ):
        p = sub.add_parser(name, help=desc)
        p.add_argument("--src", default=str(DEFAULT_SRC), help="Directory produced by `collect spectator`")
        p.add_argument("--out", help="Output JSON (or NPZ for gradshap) path")
        if name in ("parcel-benefit", "gradshap"):
            p.add_argument("--atlas-tsv", default=str(ATLAS_TSV))
        if name == "parcel-benefit":
            p.add_argument("--isc", help="Path to isc.npy (from `analysis isc`)")
            p.add_argument("--shares", help="Path to share-maps.npz (from `analysis collected`)")
        if name == "gradshap":
            p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
            p.add_argument("--bundle", default=str(BUNDLE))
            p.add_argument("--pairs", type=int, default=300)
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
