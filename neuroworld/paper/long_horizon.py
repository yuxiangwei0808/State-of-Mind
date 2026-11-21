"""neuroworld/paper/long_horizon.py (Appendix C.2, Fig. 7): Forecast up to 5 minutes ahead on all Algonauts test data
(Friends S6 and the four Movie10 films).

    python -m neuroworld.paper.long_horizon run --part latent --out <dir> [--transformer-dir <run>]
    python -m neuroworld.paper.long_horizon run --part ridge --out <dir>
    python -m neuroworld.paper.long_horizon report --out <dir>

Queries: the benchmark's Forecast origins (every 2 frames, the last 30 s complete, 180 s of history supplied) whose next 5 minutes
(HORIZON = 201 frames, 299.5 s) lie in one supported segment of the recording. Models, none refitted:
  Spectator          outputs/spectator-v1/algonauts/model (with its shipped probe.npz) and its exact state/stimulus split
  movie only         the same Spectator with model.zero_brain = True (empty fMRI history)
  Transformer        optional Transformer counterpart passed via --transformer-dir (with probe.npz or --transformer-probe)
  TRIBE v2           the stimulus-only zero-shot predictions (references.SavedEncoder)
  stimulus ridge     the linear forecaster's own stimulus part on the contextual rows (references.StimulusRidge)
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from neuroworld.paths import OUTPUT

TR, HORIZON, SPLITS = 1.49, 201, ("val", "dev", "test")
BANDS = ((0, 60), (60, 120), (120, 180), (180, 300))
RELEASE = OUTPUT / "spectator-v1/algonauts/model"
MOVIE_ONLY = "Movie only (Spectator from empty frames)"
TRANSFORMER = "Transformer counterpart"
TRIBE = "TRIBE v2 (stimulus only)"
RIDGE = "Stimulus ridge (the linear forecaster's movie part)"
log = lambda text: print(text, flush=True)


def base_config(out):
    """The Algonauts base config with its paths resolved (neuroworld/paths.py), written beside the results."""
    import yaml
    from neuroworld.paths import expand
    cfg = expand(yaml.safe_load(Path("neuroworld/configs/algonauts/base.yaml").read_text()))
    path = Path(out) / "base.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def origins_for(run, grid, horizon):
    """Forecast origins (queries.forecast_origins) whose next `horizon` targets lie in one supported segment."""
    from neuroworld.benchmark.grid import LONGEST_PREFIX
    longest = grid.prefixes()[LONGEST_PREFIX]
    return np.array([t0 for t0 in range(longest, run.n - horizon + 1, grid.stride_trs)
                     if run.prefix_complete(t0, longest) and run.target_span_ok(t0, horizon)], dtype=np.int64)


def sums(pred, obs, ok, mean):
    """Per-step squared error of the forecast and of the run mean, summed over parcels and scoreable targets (metrics.step_scores)."""
    err = np.where(ok, ((pred.astype(np.float64) - obs) ** 2).sum(-1), 0.0).sum(0)
    ref = np.where(ok, ((obs.astype(np.float64) - mean) ** 2).sum(-1), 0.0).sum(0)
    return err, ref


class Decomposed:
    """Spectator's forecast as its state part plus its movie part (collect.spectator), from the states its RolloutAdapter would use."""

    def __init__(self, teacher, device):
        import torch
        self.adapter, self.probe, self.device = teacher.row.model, teacher.row.probe, device
        model = self.adapter.model
        p = model.predictor
        L = p.linear.weight.detach().double()
        self.Wm, self.c = p.condition.weight.detach().double(), p.condition.bias.detach().double()
        self.M = torch.eye(L.shape[0], device=L.device, dtype=torch.float64) + L
        self.model, self.p = model, p

    def __call__(self, batch, horizon):
        """(state part, movie part) [n, horizon, 1000] for a QueryBatch."""
        import torch
        a = self.adapter
        filled = a._fill_gaps(batch)
        window = torch.from_numpy(a._window(filled)).to(self.device)
        first = filled.origins - a.frames + a.state_frames - a.stimulus.lag - (a.stimulus.context - 1)
        length = a.prefix_states + horizon - 1 + a.stimulus.context - 1
        rows = [a.stimulus.rows(filled.timeline_ids[j], int(first[j]), length) for j in range(len(filled))]
        stimuli = {k: torch.from_numpy(np.stack([r[k] for r in rows])).to(self.device) for k in rows[0]}
        with torch.no_grad():
            x0 = self.model.encode(window)[:, -1].double()
            u = self.p.condition_norm(self.model.encode_stimulus(stimuli).float()).double()
            drive = u @ self.Wm.T + self.c
            xs, xm = x0, torch.zeros_like(x0)
            state_part, movie_part = [], []
            for k in range(horizon):
                xs = xs @ self.M.T
                xm = xm @ self.M.T + drive[:, k]
                state_part.append(xs)
                movie_part.append(xm)
            state_part, movie_part = torch.stack(state_part, 1), torch.stack(movie_part, 1)
        out_s, out_m = np.empty((len(filled), horizon, 1000), np.float32), np.empty((len(filled), horizon, 1000), np.float32)
        subjects = np.array(filled.subject_ids)
        for subject in np.unique(subjects):
            rows_ = np.flatnonzero(subjects == subject)
            Wp, bp = self.probe.weights[subject]
            Wp = torch.from_numpy(np.asarray(Wp, np.float64)).to(self.device)
            bp = torch.from_numpy(np.asarray(bp, np.float64)).to(self.device)
            idx = torch.from_numpy(rows_).to(self.device)
            out_s[rows_] = (state_part[idx] @ Wp).float().cpu().numpy()
            out_m[rows_] = (movie_part[idx] @ Wp + bp).float().cpu().numpy()
        return out_s, out_m


class GPUHead:
    """A latent row's stored probe applied on the GPU, in a RolloutAdapter's decoder place (adapters.ProbeHead does it in NumPy)."""

    def __init__(self, probe, device):
        import torch
        self.weights = {g: (torch.from_numpy(np.asarray(W, np.float32)).to(device), torch.from_numpy(np.asarray(b, np.float32)).to(device))
                        for g, (W, b) in probe.weights.items()}

    def subject_indices(self, subject_ids, device=None):
        return list(subject_ids)

    def __call__(self, states, subjects):
        import torch
        out = torch.empty(*states.shape[:-1], 1000, device=states.device)
        subjects = np.array(subjects)
        for g in np.unique(subjects):
            rows = torch.from_numpy(np.flatnonzero(subjects == g)).to(states.device)
            W, b = self.weights[g]
            out[rows] = states[rows].float() @ W + b
        return out


def latent_forecast(teacher, batch):
    """[n, horizon, 1000] the row's forecast: its adapter's rollout read by its probe (LatentProbe.predict), with the probe on the GPU and
    without the adapter's memory of earlier rollouts."""
    a = teacher.row.model
    return a._rollout(a._fill_gaps(batch))[0]


def stimulus_ridge(data, out):
    """The stimulus ridge on the contextual rows (references.StimulusRidge), fitted once and stored in out/stimulus-ridge.npz."""
    from neuroworld.benchmark.references import DELAYS, StimulusRidge, StimulusRows, feature_tag
    out = Path(out)
    cache = out / "stimulus-rows-context.npz"
    path = out / "stimulus-ridge.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"]))
            ridge = StimulusRidge.__new__(StimulusRidge)
            ridge.data, ridge.rows = data, StimulusRows(data, DELAYS[data.dataset], cache, "context")
            ridge.name = "stimulus-only ridge" + feature_tag(data, "context")
            ridge.selection, ridge.alpha = meta["selection"], meta["selection"]["alpha"]
            ridge.weights = {g: tuple(saved[f"{i} {k}"] for k in ("mx", "my", "W")) for i, g in enumerate(meta["groups"])}
        log(f"stimulus ridge: loaded {path} (alpha {ridge.alpha:g})")
        return ridge
    ridge = StimulusRidge(data, cache=cache, log=log, features="context")
    groups = sorted(ridge.weights)
    np.savez(path, meta=json.dumps({"groups": groups, "selection": ridge.selection}, default=float),
             **{f"{i} {k}": v for i, g in enumerate(groups) for k, v in zip(("mx", "my", "W"), ridge.weights[g])})
    return ridge


def per_frame(predict_frames, run, origins, horizon):
    """[n, horizon, 1000] a stimulus model's predictions at every query's targets, computed once per recording frame (the same values
    its predict(batch) gives a query that is not remapped)."""
    frames = predict_frames(run.subject_id, run.stimulus_id, np.arange(run.n))  # queries name the stimulus as their timeline
    index = np.minimum(origins[:, None] + np.arange(horizon)[None], run.n - 1)
    return np.asarray(frames, np.float32)[index]


def run(args):
    """part latent: Spectator (with its state/movie split and the 60-s check), movie only, optional Transformer counterpart, and TRIBE v2.
    part ridge: the stimulus ridge. Each writes <part>-<split>.npz."""
    import torch
    from neuroworld.benchmark import queries as Q
    from neuroworld.benchmark.data import BenchmarkData
    from neuroworld.benchmark.grid import frames_within
    from neuroworld.benchmark.references import SavedEncoder
    from neuroworld.paper.generated import Teacher
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = BenchmarkData(base_config(out))
    grid, prefix = data.grid, frames_within(180.0, data.grid.tr)
    if args.part == "latent":
        model_dir = Path(args.model_dir)
        teachers = {
            "Spectator": Teacher("spectator", 180.0, data, args.device, run_dir=model_dir, probe_path=args.probe),
            MOVIE_ONLY: Teacher("movie-only", 180.0, data, args.device, run_dir=model_dir, probe_path=args.probe, zero_brain=True),
        }
        if args.transformer_dir:
            teachers[TRANSFORMER] = Teacher("transformer", 180.0, data, args.device,
                                            run_dir=args.transformer_dir, probe_path=args.transformer_probe)
        for t in teachers.values():
            t.row.model.readout = GPUHead(t.row.probe, args.device)
        tribe = SavedEncoder(data)
        decomposed = Decomposed(teachers["Spectator"], args.device)
        extra_teachers = [k for k in teachers if k != "Spectator"]
        labels = list(teachers) + [TRIBE]
    else:
        ridge = stimulus_ridge(data, out)
        labels = [RIDGE]
    checked = False
    for split in SPLITS:
        target = out / f"{args.part}-{split}.npz"
        if target.is_file():
            log(f"{args.part} {split}: done before")
            continue
        started = time.time()
        acc = {label: {} for label in labels}
        check = {}
        share = {"sum_s": np.zeros((HORIZON, 1000)), "sq_s": np.zeros((HORIZON, 1000)), "sum_m": np.zeros((HORIZON, 1000)),
                 "sq_m": np.zeros((HORIZON, 1000)), "count": 0}
        runs = data.runs(split)
        for i, r in enumerate(runs):
            unit = data.bootstrap_unit(r)
            mean = Q.run_mean(r)
            if args.part == "latent":  # the benchmark's own 60-s queries, for the check
                standard = Q.forecast_origins(r, grid)
                for start in range(0, len(standard), args.batch):
                    o = standard[start:start + args.batch]
                    obs, ok = Q.targets(r, o, grid.horizon_steps)
                    pred = latent_forecast(teachers["Spectator"], Q.batch(data.dataset, grid.tr, r, o, prefix, grid.horizon_steps))
                    e, f = sums(pred, obs, ok, mean)
                    a, b = check.get(unit, (0.0, 0.0))
                    check[unit] = (a + e.sum(), b + f.sum())
            origins = origins_for(r, grid, HORIZON)
            if len(origins):
                obs, ok = Q.targets(r, origins, HORIZON)
                if args.part == "ridge":
                    stimulus_all = per_frame(ridge.predict_frames, r, origins, HORIZON)
                else:
                    tribe_all = per_frame(tribe.predict_frames, r, origins, HORIZON)
                for start in range(0, len(origins), args.batch):
                    sl = slice(start, start + args.batch)
                    o, ob, okb = origins[sl], obs[sl], ok[sl]
                    if args.part == "ridge":
                        preds = {RIDGE: stimulus_all[sl]}
                        if not checked:
                            batch = Q.batch(data.dataset, grid.tr, r, o, prefix, HORIZON)
                            rdiff = float(np.abs(np.asarray(ridge.predict(batch), np.float32) - stimulus_all[sl]).max())
                            log(f"check: stimulus ridge per frame vs predict {rdiff:.2e}")
                            (out / "ridge-check.json").write_text(json.dumps({"largest difference": rdiff, "run": r.run_id}))
                            checked = True
                    else:
                        batch = Q.batch(data.dataset, grid.tr, r, o, prefix, HORIZON)
                        ys, ym = decomposed(batch, HORIZON)
                        preds = {"Spectator": ys + ym}
                        if not checked:  # the split is exact: it reproduces the adapter's own rollout
                            diff = float(np.abs(latent_forecast(teachers["Spectator"], batch) - preds["Spectator"]).max())
                            tdiff = float(np.abs(np.asarray(tribe.predict(batch), np.float32) - tribe_all[sl]).max())
                            log(f"checks: decomposition vs the adapter's {HORIZON}-step forecast {diff:.2e}; TRIBE per frame vs predict {tdiff:.2e}")
                            (out / "decomposition-check.json").write_text(json.dumps({"largest difference": diff, "TRIBE per-frame difference": tdiff,
                                                                                      "run": r.run_id}))
                            checked = True
                        for label in extra_teachers:
                            preds[label] = latent_forecast(teachers[label], batch)
                        preds[TRIBE] = tribe_all[sl]
                        ys64, ym64 = ys.astype(np.float64), ym.astype(np.float64)
                        share["sum_s"] += ys64.sum(0); share["sq_s"] += (ys64 ** 2).sum(0)
                        share["sum_m"] += ym64.sum(0); share["sq_m"] += (ym64 ** 2).sum(0)
                        share["count"] += len(ys)
                    for label, pred in preds.items():
                        e, f = sums(pred, ob, okb, mean)
                        a = acc[label].setdefault(unit, [np.zeros(HORIZON), np.zeros(HORIZON)])
                        a[0] += e
                        a[1] += f
            r.__dict__.pop("brain", None)
            if (i + 1) % 25 == 0:
                log(f"{args.part} {split} {i + 1}/{len(runs)} ({time.time() - started:.0f}s)")
        arrays = {}
        for j, label in enumerate(labels):
            units = sorted(acc[label])
            arrays[f"units {j}"] = np.array(units)
            arrays[f"err {j}"] = np.stack([acc[label][u][0] for u in units]) if units else np.zeros((0, HORIZON))
            arrays[f"ref {j}"] = np.stack([acc[label][u][1] for u in units]) if units else np.zeros((0, HORIZON))
        units = sorted(check)
        np.savez(target, labels=np.array(labels), check_units=np.array(units), check_err=np.array([check[u][0] for u in units]),
                 check_ref=np.array([check[u][1] for u in units]), **arrays, **{f"share {k}": np.asarray(v) for k, v in share.items()})
        log(f"{args.part} {split}: {len(runs)} recordings in {time.time() - started:.0f}s")
        torch.cuda.empty_cache()
    log(f"long-horizon {args.part} finished")


def _stored_check():
    """Spectator's stored Forecast unit sums (paper Table 2), every split from the root that holds it."""
    from neuroworld.benchmark import forecast as F
    from neuroworld.paper.analysis import ROOTS
    out = {}
    for split in SPLITS:
        path = next((r / "algonauts" / split / "ridge-jepa-180s-std-partial (latent)" for r in ROOTS
                     if (r / "algonauts" / split / "ridge-jepa-180s-std-partial (latent)" / "results.json").is_file()), None)
        if path is None:
            return None
        for row in F.load(path)["forecast"]:
            s = row["scores"]["P30"]
            a, b = out.get(row["unit"], (0.0, 0.0))
            out[row["unit"]] = (a + float(np.nansum(s["err"])), b + float(np.nansum(s["ref"])))
    return out


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neuroworld.benchmark import profile as P
    out = Path(args.out)
    loaded = [np.load(out / f"latent-{split}.npz") for split in SPLITS]
    ridge_files = [out / f"ridge-{split}.npz" for split in SPLITS]
    parts = loaded + ([np.load(p) for p in ridge_files] if all(p.is_file() for p in ridge_files) else [])
    labels = list(dict.fromkeys(str(x) for d in parts for x in d["labels"]))
    units = {label: {} for label in labels}
    for d in parts:
        for j, label in enumerate(str(x) for x in d["labels"]):
            for u, e, f in zip(d[f"units {j}"], d[f"err {j}"], d[f"ref {j}"]):
                units[label][str(u)] = (e, f)
    seconds = (np.arange(HORIZON) + 1) * TR
    skill = lambda by_unit, steps: 100 * (1 - sum(v[0][steps].sum() for v in by_unit.values()) / sum(v[1][steps].sum() for v in by_unit.values()))
    band_units = lambda by_unit, steps: {u: (float(v[0][steps].sum()), float(v[1][steps].sum())) for u, v in by_unit.items()}
    fmt = lambda d: None if d is None or d.get("low") is None else [round(d["value"], 2), round(d["low"], 2), round(d["high"], 2)]
    res = {"queries": {}, "bands": {}, "spectator minus": {}, "by_step": {}, "state_share": {}, "check": {}}
    bands = {f"{a}-{b} s": np.flatnonzero((seconds > a) & (seconds <= b)) for a, b in BANDS}
    for label in labels:
        res["bands"][label] = {name: round(skill(units[label], steps), 2) for name, steps in bands.items()}
        res["by_step"][label] = [round(100 * (1 - sum(v[0][k] for v in units[label].values()) / sum(v[1][k] for v in units[label].values())), 3)
                                 for k in range(HORIZON)]
        if label != "Spectator":
            res["spectator minus"][label] = {name: fmt(P._paired(band_units(units["Spectator"], steps), band_units(units[label], steps)))
                                             for name, steps in bands.items()}
    # State share by seconds ahead: variance across queries of each part (analysis.collected's definition).
    s = {k: sum(d[f"share {k}"] for d in loaded) for k in ("sum_s", "sq_s", "sum_m", "sq_m", "count")}
    var_s = s["sq_s"] / s["count"] - (s["sum_s"] / s["count"]) ** 2
    var_m = s["sq_m"] / s["count"] - (s["sum_m"] / s["count"]) ** 2
    share = var_s.sum(1) / (var_s.sum(1) + var_m.sum(1))
    res["state_share"] = {f"{seconds[k]:.1f}": round(float(share[k]), 4) for k in range(HORIZON)}
    res["state_share_bands"] = {name: round(float(share[steps].mean()), 4) for name, steps in bands.items()}
    res["queries"] = {"5-min queries": int(s["count"]), "units": len(units["Spectator"])}
    # The check: Spectator on the benchmark's own 60-s queries against its stored row.
    mine = {str(u): (float(e), float(f)) for d in loaded for u, e, f in zip(d["check_units"], d["check_err"], d["check_ref"])}
    stored = _stored_check()
    sk = lambda u: 100 * (1 - sum(v[0] for v in u.values()) / sum(v[1] for v in u.values()))
    res["check"] = {"Spectator Forecast, benchmark queries, recomputed": round(sk(mine), 3)}
    if stored:
        res["check"]["stored (Table 2)"] = round(sk(stored), 3)
        res["check"]["largest relative unit difference"] = float(max(abs(mine[u][0] - stored[u][0]) / stored[u][0] for u in stored))
    if (out / "decomposition-check.json").is_file():
        res["check"]["decomposition"] = json.loads((out / "decomposition-check.json").read_text())
    (out / "results.json").write_text(json.dumps(res, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = {"Spectator": "#C0392B", MOVIE_ONLY: "#E67E22", TRANSFORMER: "#2E86C1", TRIBE: "#AAB7B8", RIDGE: "#555555"}
    for label in labels:
        axes[0].plot(seconds, res["by_step"][label], label=label, color=colors[label], lw=2 if label == "Spectator" else 1.1)
    axes[0].set_xlabel("seconds ahead"); axes[0].set_ylabel("Forecast skill (%)"); axes[0].axhline(0, color="#cccccc", lw=0.5)
    axes[0].set_ylim(-5, 30)
    if TRANSFORMER in res["by_step"]:
        leaves = next(((k + 1) * TR for k, v in enumerate(res["by_step"][TRANSFORMER]) if v < -5), None)
        if leaves:
            axes[0].annotate(f"Transformer counterpart\nbelow -5 after {leaves:.0f} s", (leaves, -5), xytext=(leaves - 95, 3), fontsize=6,
                             color=colors[TRANSFORMER], arrowprops=dict(arrowstyle="-", color=colors[TRANSFORMER], lw=0.6))
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    axes[1].plot(seconds, 100 * share, color=colors["Spectator"]); axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("seconds ahead"); axes[1].set_ylabel("forecast variance from\nthe inferred state (%)")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "long_horizon.pdf"); fig.savefig(out / "long_horizon.png", dpi=200)
    print(json.dumps({k: res[k] for k in ("queries", "bands", "spectator minus", "state_share_bands", "check")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--part", choices=("latent", "ridge"), required=True)
    r.add_argument("--model-dir", default=str(RELEASE), help="Spectator model folder (config.yaml, best.pt, probe.npz)")
    r.add_argument("--probe", help="Optional probe .npz overriding <model-dir>/probe.npz")
    r.add_argument("--transformer-dir", help="Optional Transformer counterpart model folder")
    r.add_argument("--transformer-probe", help="Optional probe .npz for --transformer-dir")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"run": run, "report": report}[args.command](args)
