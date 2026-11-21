"""neuroworld/paper/collect.py (paper, Sec. 5.5 and Appendix D): forecasts on a fixed set of starting points for post-hoc
analysis, and for Spectator its exact linear decomposition into state and stimulus terms.

    python -m neuroworld.paper.collect spectator [--model-dir outputs/spectator-v1/algonauts/model] \
        [--out outputs/paper-analysis/spectator] [--every 10] [--device cuda]
    python -m neuroworld.paper.collect latent <model-dir> <row-name> <seconds> <store> <out-dir> [--device cuda]
    python -m neuroworld.paper.collect linear <seconds> <out-dir> [--device cuda]

Starting points: every ``--every``-th forecast starting point (default 10, about 15 s apart; use ``--every 1`` for every
3 s benchmark origin as in Table 9 / Fig. 3-4) of every Algonauts test recording (Friends S6 and all four Movie10 films).
Per recording, ``<out dir>/<split>/<run>.npz`` holds ``pred`` [n, 40, 1000] float16 and (for spectator) ``obs``, ``ok``,
``mean``, ``origins``, ``subject``, ``timeline``, ``content``, ``state0`` [n, 2048], ``ys``, ``ym``, ``fast0``, ``slow0``.

Spectator (x' = x + L x + W u + c, read by the probe y = x P + b): with M = I + L, a forecast k steps ahead splits exactly:
  x_k = M^k x_0  (carried by the inferred state)  +  sum_j M^(k-1-j) (W u_j + c)  (driven by the stimulus).
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from neuroworld.benchmark import grid as G, queries as Q
from neuroworld.benchmark.adapters import RolloutAdapter
from neuroworld.benchmark.data import BenchmarkData
from neuroworld.benchmark.forecast import predict
from neuroworld.benchmark.latent import LatentProbe, Probe
from neuroworld.paths import OUTPUT
from neuroworld.train.readout import load_world_model
from neuroworld.utils.config import load_config

RELEASE_ALGONAUTS = OUTPUT / "spectator-v1/algonauts/model"
SPLITS = ("val", "dev", "test")
log = lambda text: print(text, flush=True)


class Head:
    subjects = []

    def __init__(self, probe=None):
        self.probe = probe

    def subject_indices(self, subject_ids, device=None):
        return list(subject_ids)

    def __call__(self, states, subjects):
        if self.probe is None:
            return torch.zeros(*states.shape[:-1], 1000, device=states.device)
        return torch.from_numpy(self.probe(states.float().cpu().numpy(), subjects)).to(states.device)


def load_saved_probe(probe_path, dataset):
    """Load a shipped probe .npz (e.g. spectator-v1/algonauts/model/probe.npz) without refitting."""
    path = Path(probe_path)
    probe = object.__new__(Probe)
    with np.load(path, allow_pickle=False) as saved:
        probe.meta = json.loads(str(saved["meta"]))
        probe.weights = {g: (saved[f"W {i}"], saved[f"b {i}"]) for i, g in enumerate(probe.meta["groups"])}
    probe.dataset, probe.open_loop_only, probe.path = dataset, False, path
    probe.alpha = probe.meta["penalty"]
    probe.provenance = {"probe": str(path), "penalty": probe.alpha}
    return probe


def latent_row(model_dir, name, seconds, store, data, device="cuda", probe_path=None):
    """A latent row: the model at ``model_dir`` read through its shipped ``probe.npz`` (if present) or a fitted Probe."""
    G.HISTORY_SECONDS = float(seconds)
    model_dir = Path(model_dir)
    cfg = load_config(model_dir / "config.yaml")
    model, _ = load_world_model(cfg, model_dir / "best.pt", device)
    states = getattr(model, "read_states", cfg["model"]["context_states"]) * getattr(model, "state_step", 1)
    adapter = RolloutAdapter(f"{name} open-loop", model, Head(), cfg, device, states)
    shipped = Path(probe_path) if probe_path else model_dir / "probe.npz"
    if shipped.is_file():
        probe = load_saved_probe(shipped, data.dataset)
        adapter.name = name
        adapter.readout = Head(probe)
        adapter._memory, adapter._memory_bytes = OrderedDict(), 0
        return adapter, probe, LatentProbe(adapter, probe, f"{name} (latent)"), cfg, model
    from neuroworld.benchmark import updating as U
    origins = U.origins
    U.origins = lambda data, run: np.zeros(0, dtype=np.int64)
    first = Probe(data, adapter, store, log=log)
    U.origins = origins
    adapter.readout = Head(first)
    adapter._memory, adapter._memory_bytes = OrderedDict(), 0
    adapter.name = name
    probe = Probe(data, adapter, store, log=log)
    adapter.readout = Head(probe)
    adapter._memory, adapter._memory_bytes = OrderedDict(), 0
    return adapter, probe, LatentProbe(adapter, probe, f"{name} (latent)"), cfg, model


def queries(data, run, every=10):
    grid = data.grid
    return Q.forecast_origins(run, grid)[::every]


def save(out, split, run, **arrays):
    folder = Path(out) / split
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{run.run_id.replace('/', '__')}.npz", **arrays)


def spectator(model_dir, out, *, seconds=180.0, store=None, every=10, device="cuda", probe_path=None):
    data = BenchmarkData("neuroworld/configs/algonauts/base.yaml")
    model_dir = Path(model_dir)
    store = Path(store) if store else OUTPUT / "benchmark-results"
    adapter, probe, row, cfg, model = latent_row(model_dir, model_dir.name, seconds, store, data, device=device, probe_path=probe_path)
    p = model.predictor
    D = model.encoder.state_dim if hasattr(model.encoder, "state_dim") else p.linear.weight.shape[0]
    L = p.linear.weight.detach().double()              # x' = x + x L^T + u W^T + c
    Wm, c = p.condition.weight.detach().double(), p.condition.bias.detach().double()
    M = torch.eye(L.shape[0], device=L.device, dtype=torch.float64) + L
    eig = torch.linalg.eigvals(M.cpu()).numpy()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "eigenvalues.npy", eig)
    np.savez(out / "matrices.npz", M=M.float().cpu().numpy(), W=Wm.float().cpu().numpy(), c=c.float().cpu().numpy(),
             probes=json.dumps({k: None for k in probe.weights}))
    for key, (Wp, bp) in probe.weights.items():
        np.savez(out / f"probe-{key.replace('/', '__')}.npz", W=np.asarray(Wp, np.float32), b=np.asarray(bp, np.float32))
    grid, horizon = data.grid, data.grid.horizon_steps
    need = adapter.frames
    checked = False
    for split in SPLITS:
        runs = data.runs(split)
        for i, run in enumerate(runs):
            origins = queries(data, run, every=every)
            if not len(origins):
                continue
            obs, ok = Q.targets(run, origins, horizon)
            mean = Q.run_mean(run)
            batch = Q.batch(data.dataset, grid.tr, run, origins, grid.history_frames(), horizon)
            window = torch.from_numpy(adapter._window(batch)).to(device)
            first = batch.origins - need + adapter.state_frames - adapter.stimulus.lag - (adapter.stimulus.context - 1)
            length = adapter.prefix_states + horizon - 1 + adapter.stimulus.context - 1
            rows = [adapter.stimulus.rows(batch.timeline_ids[j], int(first[j]), length) for j in range(len(batch))]
            stimuli = {k: torch.from_numpy(np.stack([r[k] for r in rows])).to(device) for k in rows[0]}
            with torch.no_grad():
                x0 = model.encode(window)[:, -1].double()                        # [n, D] the inferred state
                u = model.encode_stimulus(stimuli).double()                       # [n, horizon, C]
                u = p.condition_norm(u.float()).double()
                drive = u @ Wm.T + c                                              # [n, horizon, D]
                xs, xm = x0, torch.zeros_like(x0)
                state_part, movie_part = [], []
                for k in range(horizon):
                    xs = xs @ M.T
                    xm = xm @ M.T + drive[:, k]
                    state_part.append(xs)
                    movie_part.append(xm)
                state_part, movie_part = torch.stack(state_part, 1), torch.stack(movie_part, 1)
                total = (state_part + movie_part).float()
                if not checked:  # the decomposition reproduces the model's own rollout
                    roll = model.rollout(window, stimuli, horizon_trs=horizon).float()
                    log(f"decomposition check: largest difference {float((roll - total).abs().max()):.2e} (state scale {float(total.abs().mean()):.3f})")
                    checked = True
            Wp, bp = probe.weights[run.subject_id] if run.subject_id in probe.weights else next(iter(probe.weights.values()))
            Wp, bp = torch.from_numpy(np.asarray(Wp, np.float32)).to(device).double(), torch.from_numpy(np.asarray(bp, np.float32)).to(device).double()
            ys, ym = state_part @ Wp, movie_part @ Wp + bp                        # [n, horizon, 1000]
            half = D // 2
            fast0, slow0 = x0[:, :half] @ Wp[:half], x0[:, half:] @ Wp[half:]      # the inferred state's reading, by component
            pred = (ys + ym).float().cpu().numpy()
            save(out, split, run, pred=pred.astype(np.float16), obs=obs.astype(np.float16), ok=ok, mean=mean.astype(np.float32),
                 origins=origins, subject=run.subject_id, timeline=run.timeline_id, content=run.group, state0=x0.float().cpu().numpy(),
                 ys=ys.float().cpu().numpy().astype(np.float16), ym=ym.float().cpu().numpy().astype(np.float16),
                 fast0=fast0.float().cpu().numpy().astype(np.float16), slow0=slow0.float().cpu().numpy().astype(np.float16))
            if (i + 1) % 25 == 0:
                log(f"{split} {i + 1}/{len(runs)}")
    log("spectator finished")


def latent(model_dir, name, seconds, store, out, *, every=10, device="cuda"):
    data = BenchmarkData("neuroworld/configs/algonauts/base.yaml")
    _, _, row, _, _ = latent_row(model_dir, name, seconds, store, data, device=device)
    collect(data, row, out, every=every)


def linear(seconds, out, *, store=None, every=10, device="cuda"):
    from neuroworld.benchmark import updating as U
    from neuroworld.experiments.history_length import BUDGETS, CONFIGS, LONGEST, models
    G.HISTORY_SECONDS = LONGEST
    U.BUDGETS = BUDGETS
    data = BenchmarkData(CONFIGS["algonauts"])
    _, model = next(models("linear", data, [float(seconds)], device, log, "context", store or OUTPUT / "benchmark-results"))
    collect(data, model, out, every=every)


def collect(data, model, out, every=10):
    grid, horizon = data.grid, data.grid.horizon_steps
    for split in SPLITS:
        runs = data.runs(split)
        for i, run in enumerate(runs):
            origins = queries(data, run, every=every)
            if not len(origins):
                continue
            pred = predict(model, data.dataset, grid.tr, run, origins, grid.history_frames(), horizon)
            save(out, split, run, pred=pred.astype(np.float16), origins=origins)
            if (i + 1) % 25 == 0:
                log(f"{split} {i + 1}/{len(runs)}")
    log(f"{model.name} finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)

    sp = sub.add_parser("spectator", help="Collect Spectator's forecasts and exact state/stimulus decomposition")
    sp.add_argument("--model-dir", default=str(RELEASE_ALGONAUTS), help="Spectator model folder (config.yaml, best.pt, optional probe.npz)")
    sp.add_argument("--probe", help="Optional probe .npz overriding <model-dir>/probe.npz")
    sp.add_argument("--store", default=str(OUTPUT / "benchmark-results"), help="Store for fitting/caching a probe if absent")
    sp.add_argument("--out", default=str(OUTPUT / "paper-analysis/spectator"), help="Output directory")
    sp.add_argument("--seconds", type=float, default=180.0)
    sp.add_argument("--every", type=int, default=10, help="Subsampling stride over benchmark origins (1 = all 78,800 origins)")
    sp.add_argument("--device", default="cuda")

    lt = sub.add_parser("latent", help="Collect forecasts of any latent model directory")
    lt.add_argument("model_dir")
    lt.add_argument("name")
    lt.add_argument("seconds", type=float)
    lt.add_argument("store")
    lt.add_argument("out")
    lt.add_argument("--every", type=int, default=10)
    lt.add_argument("--device", default="cuda")

    ln = sub.add_parser("linear", help="Collect forecasts of the linear forecaster")
    ln.add_argument("seconds", type=float)
    ln.add_argument("out")
    ln.add_argument("--store", default=str(OUTPUT / "benchmark-results"))
    ln.add_argument("--every", type=int, default=10)
    ln.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.kind == "spectator":
        spectator(args.model_dir, args.out, seconds=args.seconds, store=args.store, every=args.every, device=args.device, probe_path=args.probe)
    elif args.kind == "latent":
        latent(args.model_dir, args.name, args.seconds, args.store, args.out, every=args.every, device=args.device)
    else:
        linear(args.seconds, args.out, store=args.store, every=args.every, device=args.device)


if __name__ == "__main__":
    main()
