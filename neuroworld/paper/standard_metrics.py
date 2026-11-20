"""neuroworld/paper/standard_metrics.py: the metrics of prior work for the models of Table 2 (the paper's appendix), on all
Algonauts test data, with NeuroWorld's definitions in the benchmark's Forecast setting (NeuroWorld's evaluation
code): an origin every 3 s, forecasts of the next 40 frames.
  Pearson    per recording and frame ahead, each parcel's correlation between forecast and measured fMRI across origins,
             averaged over parcels; then over frames 1-20 (NeuroWorld's 20-frame horizon) or 1-40, then over recordings.
  global r   per recording and frame ahead, one correlation over all origins x parcels; averaged the same way.
  Cls@Top10  each origin's forecast of frames 1-20 (x 1000 parcels) is correlated with the measured frames 1-20 at every
             origin of the same viewer; a hit when its own measured segment ranks in the top 10 among segments of other
             content (other episodes and movies; the other viewing of a Movie10 clip and the other half of an episode
             are not candidates). Origins whose 20 frames are not all scoreable are left out. Chance is 10 / candidates.

    PYTHONPATH=<code-v27> python standard_metrics.py save <model> <history>   forecasts of one row -> FORECASTS
    PYTHONPATH=<code-v27> python standard_metrics.py compute [--device cuda]  every saved row -> results-standard-metrics.json

`save` rebuilds each row as it was scored for Table 2 (same code, history and probe; the probe of each split is read from the
store that holds that split's Table 2 result), reruns Forecast with every forecast saved, and checks that its per-frame
Pearson equals the stored row's. Its result files go to a scratch store, never to the paper's stores.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from neuroworld.paths import EXTERNAL, OUTPUT
from neuroworld.paper.analysis import ROOTS as STORES


N = OUTPUT
ROOTS = list(STORES)  # the benchmark result stores, as in analysis.py
A = N / "paper-analysis" / "standard-metrics"
FORECASTS = A / "forecasts"
SCRATCH = A / "store"
W = N / "checkpoints/algonauts/wave2_20261102"
K1 = N / "checkpoints/algonauts/neuroworld_history_20261112"
K1_ORIGINAL = Path(EXTERNAL / "NeuralWorld/checkpoints/algonauts/"
                   "latent-stimulus_k1c3d1024l6_seq19_lag3_roll16w1_sig0.09_lr5e-05_bs1024_ep100_seed0")
SPLITS = ("val", "dev", "test")
LENGTHS = ("7.5 s", "120 s", "180 s")
SEG, K = 20, 10


def row_name(model, h):
    """Table 2's benchmark row of <model> at history <h> (neuroworld/paper/history_table.py ROWS)."""
    s = h[:-2]
    return {"Linear forecaster": f"linear [context] {h}",
            "IPSID": {"7.5 s": "IPSID [context] 7.5 s", "120 s": "IPSID [context]", "180 s": "IPSID [context] 180 s"}[h],
            "NeuroWorld": "NeuroWorld-style k1 (latent)" if h == "7.5 s" else f"NeuroWorld k1 {s}s (latent)",
            "TRIBE v2": "TRIBEv2 zero-shot",
            "Spectator": f"ridge-jepa-{s}s-std-partial (latent)"}[model]


def safe(text):
    return "".join(c if c.isalnum() or c in "-_.[]" else "_" for c in text)


def log(text):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {text}", flush=True)


# ------------------------------------------------------------------------------------------------ save

def _store_for(root):
    """A scratch store that reads <root>'s probes and stimulus-row caches (links) and takes the result files."""
    store = SCRATCH / root.name
    (store / "algonauts" / "probes").mkdir(parents=True, exist_ok=True)
    for f in (root / "algonauts" / "probes").glob("*.npz"):
        _link(store / "algonauts" / "probes" / f.name, f)
    for f in (root / "algonauts").glob("stimulus-rows-*.npz"):
        _link(store / "algonauts" / f.name, f)
    return store


def _link(link, target):
    try:  # two processes may build the same store at once
        link.symlink_to(target)
    except FileExistsError:
        pass


def _patch(F, Q, current):
    """F.forecast_rows that also saves every forecast (float16) and, once per recording, the measured targets."""
    def _atomic(path, **arrays):
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)

    def forecast_rows(data, models, run):
        grid = data.grid
        origins = Q.forecast_origins(run, grid)
        if not len(origins):
            return [None] * len(models)
        observed, ok = Q.targets(run, origins, grid.horizon_steps)
        mean = Q.run_mean(run)
        out = []
        for model in models:
            forecast = F.predict(model, data.dataset, grid.tr, run, origins, grid.history_frames(), grid.horizon_steps)
            split = current["split"]
            folder = FORECASTS / safe(current["label"]) / split
            folder.mkdir(parents=True, exist_ok=True)
            _atomic(folder / f"{safe(run.run_id)}.npz", forecast=forecast.astype(np.float16), origins=origins)
            key = hashlib.sha1(np.ascontiguousarray(origins).tobytes()).hexdigest()[:10]
            obs = FORECASTS / "_observed" / split / f"{safe(run.run_id)}__{key}.npz"
            obs.parent.mkdir(parents=True, exist_ok=True)
            if not obs.exists():
                _atomic(obs, observed=observed.astype(np.float32), ok=ok, origins=origins)
            meta = F._meta(data, run)
            current["index"][run.run_id] = {**meta, "file": f"{safe(run.run_id)}.npz", "observed": obs.name}
            scores = {name: F.step_scores(forecast, observed, ok, origins, grid.tr, mean=mean) for name in grid.prefixes()}
            out.append({**meta, "origins": int(len(origins)), "scores": scores})
        return out

    F.forecast_rows = forecast_rows


def _build(model, h, data, store):
    """The row's model (named as in Table 2), built as it was scored, with its probe read from <store>."""
    from neuroworld.benchmark import grid as G
    from neuroworld.benchmark.__main__ import _models
    seconds = float(h[:-2])
    if model == "Spectator":
        from neuroworld.spectator.score import latent_row
        _, row = latent_row(W / f"ridge-jepa-{h[:-2]}s-std-partial", f"ridge-jepa-{h[:-2]}s-std-partial", seconds,
                            "algonauts", store, "cuda", log)
        return row
    if model == "Linear forecaster":  # refitted for each length; queries carry 240 s, the model reads its last L (nov12h_linear.py)
        from neuroworld.experiments.history_length import LONGEST, models
        G.HISTORY_SECONDS = LONGEST
        return next(models("linear", data, [seconds], "cuda", log, "context", N / "benchmark-test"))[1]
    if model == "IPSID":  # IPSID on the contextual rows, queries of <h> (with_history.py; 120 s is the CLI default)
        G.HISTORY_SECONDS = seconds
        m = _models("ipsid", data, "cuda", str(store), "context")[0]
        m.name = row_name(model, h)
        return m
    if model == "TRIBE v2":
        return _models("tribev2", data, "cuda", str(store))[0]
    if model == "NeuroWorld":  # the original reads its own 4.5 s from the CLI's default 120-s queries
        G.HISTORY_SECONDS = 120.0 if h == "7.5 s" else seconds
        run_dir = K1_ORIGINAL if h == "7.5 s" else K1 / f"k1-{h[:-2]}s"
        m = _models(f"latent:run:{run_dir}", data, "cuda", str(store))[-1]  # the latent row (probe), not the own decoder
        m.name = row_name(model, h)
        return m
    raise ValueError(model)


def save(model, h):
    from neuroworld.benchmark import forecast as F, queries as Q
    from neuroworld.benchmark.adapt import calibrate
    from neuroworld.benchmark.data import BenchmarkData
    from neuroworld.experiments.history_length import CONFIGS
    name = row_name(model, h)
    label = f"{model} {h}"
    current = {"label": label, "split": None, "index": {}}
    _patch(F, Q, current)
    groups = {}
    for split in SPLITS:  # each split with the store that holds its Table 2 result (its probe)
        root = next(r for r in ROOTS if (r / "algonauts" / split / name / "results.json").is_file())
        groups.setdefault(root, []).append(split)
    for root, splits in groups.items():
        store = _store_for(root)
        data = BenchmarkData(CONFIGS["algonauts"])
        m = _build(model, h, data, store)
        for split in splits:
            if (FORECASTS / safe(label) / split / "index.json").is_file():
                log(f"{label} {split}: already saved")
                continue
            current["split"], current["index"] = split, {}
            calibrate(m, data, split, log=log)
            started = time.time()
            F.run(data, m, split, output=store, log=log)
            index = {"model": name, "label": label, "split": split, "probe store": str(root), "runs": current["index"],
                     "code": os.environ.get("PYTHONPATH", ""), "stamp": time.strftime("%F %T")}
            check = _check(store / "algonauts" / split / m.name / "results.json", root / "algonauts" / split / name / "results.json")
            index["matches table 2"] = check
            (FORECASTS / safe(label) / split / "index.json").write_text(json.dumps(index, indent=1) + "\n")
            log(f"{label} {split}: {len(current['index'])} recordings saved in {time.time() - started:.0f} s; "
                f"largest per-frame Pearson difference from Table 2's row {check:.2e}")


def _check(new, old):
    """Largest absolute difference in per-recording, per-frame Pearson between the rerun and Table 2's stored row."""
    a = {r["run_id"]: np.asarray(r["scores"]["P30"]["pearson"], float) for r in json.loads(Path(new).read_text())["forecast"]}
    b = {r["run_id"]: np.asarray(r["scores"]["P30"]["pearson"], float) for r in json.loads(Path(old).read_text())["forecast"]}
    assert a.keys() == b.keys(), (len(a), len(b))
    return float(max(np.nanmax(np.abs(a[k] - b[k])) for k in a))


# --------------------------------------------------------------------------------------------- compute

def global_r(pred, obs, ok, min_rows):
    out = np.full(pred.shape[1], np.nan)
    for k in range(pred.shape[1]):
        rows = ok[:, k]
        if rows.sum() < min_rows:
            continue
        x = pred[rows, k].astype(np.float64).ravel()
        y = obs[rows, k].astype(np.float64).ravel()
        x -= x.mean()
        y -= y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        out[k] = (x * y).sum() / d if d > 0 else np.nan
    return out


def _normalize(rows):
    rows = rows - rows.mean(1, keepdims=True)
    return rows / np.maximum(np.sqrt((rows * rows).mean(1, keepdims=True)), 1e-12)


def cls_hits(pred, obs, content, device):
    """Per query of one viewer: whether its own measured segment ranks in the top K among segments of other content."""
    import torch
    dev = torch.device(device)
    p, t = torch.from_numpy(pred).to(dev), torch.from_numpy(obs).to(dev)
    codes = {c: i for i, c in enumerate(sorted(set(content)))}
    c = torch.tensor([codes[x] for x in content], device=dev)
    own = (p * t).sum(1) / p.shape[1]
    hits = torch.zeros(len(p), dtype=torch.bool, device=dev)
    for a in range(0, len(p), 2048):
        s = p[a:a + 2048] @ t.T / p.shape[1]
        better = ((s > own[a:a + 2048, None]) & (c[a:a + 2048, None] != c[None, :])).sum(1)
        hits[a:a + 2048] = better < K
    candidates = float((len(c) - torch.bincount(c)[c]).float().mean().item())
    return hits.cpu().numpy(), candidates


def compute(device):
    from neuroworld.benchmark.metrics import MIN_ROWS, step_scores
    results = {}
    for folder in sorted(p for p in FORECASTS.iterdir() if p.is_dir() and p.name != "_observed"):
        if not all((folder / s / "index.json").is_file() for s in SPLITS):
            log(f"{folder.name}: not every split saved, skipped")
            continue
        pearson, glob, segments = [], [], {}
        for split in SPLITS:
            index = json.loads((folder / split / "index.json").read_text())
            for run_id, meta in index["runs"].items():
                f = np.load(folder / split / meta["file"])
                o = np.load(FORECASTS / "_observed" / split / meta["observed"])
                pred, obs, ok = f["forecast"].astype(np.float32), o["observed"], o["ok"]
                pearson.append(step_scores(pred, obs, ok, f["origins"], 1.49)["pearson"])
                glob.append(global_r(pred, obs, ok, MIN_ROWS))
                full = ok[:, :SEG].all(1)
                if full.any():
                    n = int(full.sum())
                    seg = segments.setdefault(meta["subject"], {"p": [], "t": [], "content": []})
                    seg["p"].append(_normalize(pred[full, :SEG].reshape(n, -1)).astype(np.float32))
                    seg["t"].append(_normalize(obs[full, :SEG].reshape(n, -1)).astype(np.float32))
                    seg["content"] += [meta["content"]] * n
        hits, candidates = [], []
        for subject, seg in sorted(segments.items()):
            h, c = cls_hits(np.concatenate(seg["p"]), np.concatenate(seg["t"]), seg["content"], device)
            hits.append(h)
            candidates.append(c)
        pearson, glob = np.asarray(pearson), np.asarray(glob)
        results[index["label"]] = {
            "model": index["model"], "recordings": len(pearson),
            "Pearson 1-20": float(np.nanmean(np.nanmean(pearson[:, :SEG], 1))), "Pearson 1-40": float(np.nanmean(np.nanmean(pearson, 1))),
            "global r 1-20": float(np.nanmean(np.nanmean(glob[:, :SEG], 1))), "global r 1-40": float(np.nanmean(np.nanmean(glob, 1))),
            "Cls@Top10": float(np.concatenate(hits).mean()), "queries": int(sum(len(h) for h in hits)),
            "candidates per query": float(np.mean(candidates))}
        log(f"{index['label']}: " + ", ".join(f"{k} {v:.3f}" for k, v in results[index["label"]].items() if isinstance(v, float)))
    (N / "paper-analysis" / "results-standard-metrics.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    s = steps.add_parser("save")
    s.add_argument("model", choices=("Linear forecaster", "IPSID", "NeuroWorld", "TRIBE v2", "Spectator"))
    s.add_argument("history", choices=LENGTHS)
    c = steps.add_parser("compute")
    c.add_argument("--device", default="cuda")
    args = parser.parse_args()
    save(args.model, args.history) if args.step == "save" else compute(args.device)
