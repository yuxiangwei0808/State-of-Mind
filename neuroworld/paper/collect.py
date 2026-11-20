"""neuroworld/paper/collect.py (paper, Sec. 5): forecasts on a fixed set of starting points for post-hoc analysis, and for Spectator its exact
linear decomposition.

    python -m neuroworld.paper.collect spectator <arm> <store> <out dir>        ridge-jepa's internals and forecasts
    python -m neuroworld.paper.collect latent <arm> <row name> <seconds> <store> <out dir>   a latent row's forecasts (its stored probe)
    python -m neuroworld.paper.collect linear <seconds> <out dir>                the linear forecaster on the contextual rows (history_length.py)

Starting points: every 10th forecast starting point (about 15 s apart) of every Algonauts test recording (Friends S6, all four
Movie10 movies), with the benchmark's queries, targets and run means. Per recording, <out dir>/<split>/<run>.npz holds:
  pred [n, 40, 1000] float16, and (spectator) obs, ok, mean, origins, subject, timeline, content, and the parts below.
Spectator (latent model x' = x + L x + W u + c, read by the probe y = x P + b): with M = I + L, a forecast k steps ahead is
  x_k = M^k x_0  (the part carried by the inferred state)  +  sum_j M^(k-1-j) (W u_j + c)  (the part driven by the movie),
so each forecast splits exactly. Saved per query: state0 [2048] (the inferred state), and per-parcel sums for the parts.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from neuroworld.benchmark import grid as G, queries as Q
from neuroworld.benchmark.adapters import RolloutAdapter
from neuroworld.benchmark.data import BenchmarkData
from neuroworld.benchmark.forecast import predict
from neuroworld.benchmark.latent import LatentProbe, Probe
from neuroworld.train.readout import load_world_model
from neuroworld.utils.config import load_config
from neuroworld.paths import OUTPUT

R = OUTPUT / "checkpoints/algonauts/wave2_20261102"
SPLITS, EVERY = ("val", "dev", "test"), 10
device = "cuda"
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


def latent_row(arm, name, seconds, store, data):
    """The row panel_grid.py / panel_test.py score: the arm's world model through its stored probes."""
    G.HISTORY_SECONDS = float(seconds)
    cfg = load_config(R / arm / "config.yaml")
    model, _ = load_world_model(cfg, R / arm / "best.pt", device)
    adapter = RolloutAdapter(f"{name} open-loop", model, Head(), cfg, device,
                             getattr(model, "read_states", cfg["model"]["context_states"]) * getattr(model, "state_step", 1))
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


def queries(data, run):
    grid = data.grid
    origins = Q.forecast_origins(run, grid)[::EVERY]
    return origins


def save(out, split, run, **arrays):
    folder = Path(out) / split
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{run.run_id.replace('/', '__')}.npz", **arrays)


def spectator(arm, store, out):
    data = BenchmarkData("neuroworld/configs/algonauts/base.yaml")
    adapter, probe, row, cfg, model = latent_row(arm, arm, 180, store, data)
    p = model.predictor
    D = model.encoder.state_dim if hasattr(model.encoder, "state_dim") else p.linear.weight.shape[0]
    L = p.linear.weight.detach().double()              # x' = x + x L^T + u W^T + c
    Wm, c = p.condition.weight.detach().double(), p.condition.bias.detach().double()
    M = torch.eye(L.shape[0], device=L.device, dtype=torch.float64) + L
    eig = torch.linalg.eigvals(M.cpu()).numpy()
    Path(out).mkdir(parents=True, exist_ok=True)
    np.save(Path(out) / "eigenvalues.npy", eig)
    np.savez(Path(out) / "matrices.npz", M=M.float().cpu().numpy(), W=Wm.float().cpu().numpy(), c=c.float().cpu().numpy(),
             probes=json.dumps({k: None for k in probe.weights}))
    for key, (Wp, bp) in probe.weights.items():
        np.savez(Path(out) / f"probe-{key.replace('/', '__')}.npz", W=np.asarray(Wp, np.float32), b=np.asarray(bp, np.float32))
    grid, horizon = data.grid, data.grid.horizon_steps
    need = adapter.frames
    checked = False
    for split in SPLITS:
        runs = data.runs(split)
        for i, run in enumerate(runs):
            origins = queries(data, run)
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


def latent(arm, name, seconds, store, out):
    data = BenchmarkData("neuroworld/configs/algonauts/base.yaml")
    _, _, row, _, _ = latent_row(arm, name, seconds, store, data)
    collect(data, row, out)


def linear(seconds, out):
    from neuroworld.benchmark import updating as U
    from neuroworld.experiments.history_length import BUDGETS, CONFIGS, LONGEST, models
    G.HISTORY_SECONDS = LONGEST
    U.BUDGETS = BUDGETS
    data = BenchmarkData(CONFIGS["algonauts"])
    _, model = next(models("linear", data, [float(seconds)], "cuda", log, "context", OUTPUT / "benchmark-test"))
    collect(data, model, out)


def collect(data, model, out):
    grid, horizon = data.grid, data.grid.horizon_steps
    for split in SPLITS:
        runs = data.runs(split)
        for i, run in enumerate(runs):
            origins = queries(data, run)
            if not len(origins):
                continue
            pred = predict(model, data.dataset, grid.tr, run, origins, grid.history_frames(), horizon)
            save(out, split, run, pred=pred.astype(np.float16), origins=origins)
            if (i + 1) % 25 == 0:
                log(f"{split} {i + 1}/{len(runs)}")
    log(f"{model.name} finished")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "spectator":
        spectator(sys.argv[2], sys.argv[3], sys.argv[4])
    elif kind == "latent":
        latent(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif kind == "linear":
        linear(sys.argv[2], sys.argv[3])
