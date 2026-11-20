"""Step 2 of Spectator's training: the slow latent's lag weights, in closed form.

For each of the D fast-latent dimensions, a ridge regression from that dimension's fast latents at every lag of the window
to the mean of its next `ahead` fast latents. Fast latents are taken about what an empty window encodes to (W x, without the
bias), and frames the benchmark marks unusable enter as zero.

- Fit: the training split's frames. Where data.split_frames limits a run (HCP), each allowed stretch is cut out of its run,
  so no window reads a held-out clip.
- Penalty: picked from PENALTIES by R^2 on the val split, on windows whose target frames all lie in that split's frames.
- Scale: one number, set so the slow latent spreads as much as the fast latent.

Algonauts: 120 frames (180 s), 16 frames ahead (24 s), from stage 0's fast map. HCP: 180 frames (180 s), 24 ahead (24 s),
from the Algonauts encoder's fast map.
"""
import json

import numpy as np
import torch

from neuroworld.benchmark.data import BenchmarkData
from neuroworld.data.splits import frame_ranges

PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@torch.no_grad()
def fit_lags(source_path, dataset_config, frames, ahead, out, device="cuda"):
    device = torch.device(device)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    # a JEPA-phase encoder file, or a stage-0 checkpoint
    key = "encoder.fast.projection" if "encoder.fast.projection.weight" in source["model"] else "encoder.projection"
    W, b = source["model"][f"{key}.weight"].to(device), source["model"][f"{key}.bias"].to(device)
    D = W.shape[0]
    fast = W.shape[1] // 1000
    lags = frames - fast + 1
    data = BenchmarkData(dataset_config)

    def empty():
        return [torch.zeros(D, lags, lags, device=device, dtype=torch.float64), torch.zeros(D, lags, device=device, dtype=torch.float64),
                torch.zeros(D, device=device, dtype=torch.float64), torch.zeros(D, device=device, dtype=torch.float64), 0]

    def add(acc, windows, target):
        for part, tpart in zip(windows.split(256), target.split(256)):
            acc[0] += torch.einsum("mdl,mdk->dlk", part, part)
            acc[1] += torch.einsum("mdl,md->dl", part, tpart)
            acc[2] += (tpart ** 2).sum(0)
            acc[3] += tpart.sum(0)
            acc[4] += len(part)

    def pairs(y):
        """[M, D, L] lag windows (oldest first) and [M, D] targets; window m ends at frame m + frames - 1."""
        z = y.unfold(0, fast, 1).transpose(1, 2).flatten(1) @ W.T
        windows = z.unfold(0, lags, 1)[:-ahead].double()
        cum = torch.cat((torch.zeros(1, D, device=device, dtype=z.dtype), z.cumsum(0)))
        target = ((cum[ahead:] - cum[:-ahead]) / ahead)[lags:].double()
        return windows, target[:len(windows)]

    def brain(run):
        y = torch.from_numpy(np.nan_to_num(np.asarray(run.brain, dtype=np.float32))).to(device)
        run.__dict__.pop("brain", None)
        return y * torch.from_numpy(np.asarray(run.brain_ok)).to(device)[:, None]

    train, val = empty(), empty()
    for run in data.runs("train"):
        y = brain(run)
        for start, stop in frame_ranges(data.cfg, "train", run.run_id) or [(0, len(y))]:
            if stop - start >= frames + ahead:
                add(train, *pairs(y[start:stop]))
    for run in data.runs("val"):
        y = brain(run)
        if len(y) < frames + ahead:
            continue
        windows, target = pairs(y)
        inside = np.zeros(len(y) + 1, dtype=bool)
        for start, stop in frame_ranges(data.cfg, "val", run.run_id) or [(0, len(y))]:
            inside[start:stop] = True
        first = np.arange(len(windows)) + frames  # the first target frame of each window
        ok = np.array([inside[f:f + ahead].all() for f in first])
        if ok.any():
            index = torch.from_numpy(np.flatnonzero(ok)).to(device)
            add(val, windows[index], target[index])
    Gt, Ct, _, _, nt = train
    Gv, Cv, yyv, y1v, nv = val
    base = (yyv - y1v ** 2 / nv).clamp(min=1e-9)
    var = Gt.diagonal(dim1=1, dim2=2)[:, -1] / nt

    def fit(lam):
        diag = Gt.diagonal(dim1=1, dim2=2).mean(-1)
        eye = torch.eye(lags, device=device, dtype=torch.float64)
        return torch.linalg.solve(Gt + (lam * diag)[:, None, None] * eye, Ct[..., None])[..., 0]

    def r2(w):
        return 1 - (yyv - 2 * (w * Cv).sum(-1) + torch.einsum("dl,dlk,dk->d", w, Gv, w)) / base

    total = {lam: float(1 - ((1 - r2(fit(lam))) * base).sum() / base.sum()) for lam in PENALTIES}
    lam = max(total, key=total.get)
    w = fit(lam).float()
    slow_var = torch.einsum("dl,dlk,dk->d", w.double(), Gt, w.double()) / nt
    scale = float((var.mean() / slow_var.mean()).sqrt())
    report = {"frames": frames, "lags": lags, "ahead": ahead, "training windows": nt, "val windows": nv,
              f"val R^2 of the next {ahead} frames' mean, by penalty": {f"{k:g}": round(v, 4) for k, v in total.items()},
              "penalty": lam, "scale": round(scale, 4), "sum of lag weights, mean over dimensions": round(float(w.sum(-1).mean()), 3)}
    config = json.loads(json.dumps(source["config"]))
    config["model"].update(state_window_trs=frames, state_dim=2 * D, fast_window_trs=fast, state_tokens=2, slow_per_dimension=True)
    config["data"]["tr_seconds"] = data.cfg["data"]["tr_seconds"]  # the lag weights are at this dataset's frames
    model = {"encoder.fast.projection.weight": W.cpu(), "encoder.fast.projection.bias": b.cpu(), "encoder.lag_weights": w.cpu(),
             "encoder.scale": torch.tensor(scale, dtype=torch.float32)}
    torch.save({"format": source["format"], "config": config, "model": model, "lag_fit": report}, out)
    return report
