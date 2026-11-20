"""Latent scores (docs/benchmark.md#how-a-model-is-queried): a world model is judged by its predicted latent, not by its
decoder. A model that exposes predict_states(batch) -> (its own forecast fMRI or None, [B, H, D] predicted states) is
read out by a frozen linear probe and scored as an ordinary model on every task, beside its own decoder's row.

  probe    a ridge map from the model's predicted states to the measured fMRI at the predicted frames, fitted on the
           states the model produces under the benchmark's protocols on training recordings: open-loop rollouts from
           origins every 30 s and rollouts with 16 measurements in the minute, all steps pooled. As for the linear
           references (references.py), one probe per participant on Algonauts and one pooled probe on HCP (new
           participants use it unchanged). Each latent coordinate is standardized by its training statistics (inside
           each cross-validation fold too), so the probe does not depend on the latent's coordinate scales. The penalty
           is chosen by cross-validation across movies (a held-out fold's movie is unseen), then the probe is frozen
           before any scored recording.
  stored   <output>/<dataset>/probes/<model>.npz with its provenance (the model's weight digest, the benchmark code
           digest, the training recordings, penalty, cross-validation errors, sample counts and effective rank); it is
           reused while the model and the code match, and every task result of the latent row names it.
  scoring  LatentProbe(model, probe) predicts probe(predicted states); the tasks score it unchanged. In Updating the
           model keeps its own handling of hidden frames (designs B and C fill them with decoded forecasts and re-encode).

    python -m neuroworld.benchmark suite --model latent:run:<dir> --name "the Transformer world model" --device cuda
                                     (both readouts, scored together: "the Transformer world model" and "the Transformer world model (latent)")
"""
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np

from neuroworld.benchmark import queries as Q
from neuroworld.benchmark.grid import code_digest, frames_within
from neuroworld.eval.intersubject import stimulus_key

ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)  # times the sample count (the standardized
# Gram matrix's mean eigenvalue over its non-constant coordinates)
FOLDS = 5
STRIDE_SECONDS = 30.0  # open-loop probe-fitting origins on training recordings
UPDATE_BUDGET = 16     # measurements per minute in the probe's rollouts with measurements


# ----------------------------------------------------------------------------------------------- ridge on moments

def _moments():
    return {"G": 0.0, "C": 0.0, "z": 0.0, "y": 0.0, "yy": 0.0, "n": 0}


def _add(m, x, y):
    m["G"] = m["G"] + x.T @ x; m["C"] = m["C"] + x.T @ y; m["z"] = m["z"] + x.sum(0); m["y"] = m["y"] + y.sum(0)
    m["yy"] = m["yy"] + float((y ** 2).sum()); m["n"] += len(x)


def _minus(a, b):
    return {k: a[k] - b[k] for k in a}


def _standardized(m):
    """(mean, scale, eigenvalues, eigenvectors) of the inputs behind moments m, each coordinate scaled by its standard
    deviation there. A coordinate whose spread is below float32 resolution of its mean is constant: it is left out."""
    mx = m["z"] / m["n"]
    G = m["G"] - m["n"] * np.outer(mx, mx)
    s = np.sqrt(np.clip(np.diag(G) / m["n"], 0, None))
    s = np.where(s > 1e-6 * np.abs(mx), s, np.inf)
    lam, V = np.linalg.eigh(G / np.outer(s, s))
    return mx, s, np.clip(lam, 0, None), V


def _solutions(m, alphas):
    """[(W, b)] of the ridge on standardized inputs, one per penalty (times the sample count), in the original
    coordinates: rescaling any coordinate rescales its weights and leaves the predictions unchanged."""
    mx, s, lam, V = _standardized(m)
    my = m["y"] / m["n"]
    VC = V.T @ ((m["C"] - m["n"] * np.outer(mx, my)) / s[:, None])
    return [(W, my - mx @ W) for W in (V @ (VC / (lam + a * m["n"])[:, None]) / s[:, None] for a in alphas)]


def _effective_rank(m):
    """Participation ratio of the standardized inputs' eigenvalues: how many coordinates the variance spreads over."""
    lam = _standardized(m)[2]
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-300))


def _error(m, W, b):
    """Summed squared error of the prediction x @ W + b on the rows behind moments m."""
    return (m["yy"] - 2 * float((W * m["C"]).sum()) - 2 * float(b @ m["y"]) + float(((m["G"] @ W) * W).sum())
            + 2 * float(b @ (W.T @ m["z"])) + m["n"] * float(b @ b))


def _select(pairs):
    """(penalty, relative errors): the lowest cross-validated error over [(total, [fold moments])]."""
    if not any(len(folds) > 1 for _, folds in pairs):
        raise ValueError("Cross-validation needs recordings of at least two movies")
    errors = np.zeros(len(ALPHAS))
    for total, folds in pairs:
        for held in folds:
            if held["n"]:
                for i, (W, b) in enumerate(_solutions(_minus(total, held), ALPHAS)):
                    errors[i] += _error(held, W, b)
    return ALPHAS[int(np.argmin(errors))], {f"{a:g}": float(e / errors.min()) for a, e in zip(ALPHAS, errors)}


# ----------------------------------------------------------------------------------------------- probe

def _fold(timeline):
    """Fold of a whole recording by its stimulus: the same movie falls in the same fold for every participant."""
    return zlib.crc32(stimulus_key(timeline).encode()) % FOLDS


def _group(dataset, subject):
    """One probe per participant on Algonauts, one pooled probe on HCP (as references._group)."""
    return "all" if dataset == "hcp_movies" else subject


def _states(model, batch, size=128):
    """[B, H, D] predicted states, in chunks of `size` queries."""
    return np.concatenate([np.asarray(model.predict_states(batch.take(slice(a, a + size)))[1], dtype=np.float32)
                           for a in range(0, len(batch), size)])


def _weights_digest(model):
    """Digest of a model's weights (its torch modules), or None when it has none to identify it by."""
    import torch
    h, found = hashlib.sha256(), False
    for part in (getattr(model, "model", None), getattr(model, "readout", None)):
        if isinstance(part, torch.nn.Module):
            found = True
            for key, value in part.state_dict().items():
                h.update(key.encode() + value.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16] if found else None


class Probe:
    """The frozen linear probe of one latent model (module docstring): {group: (W [D, 1000], b [1000])}, stored with
    its provenance and reused while the model and the benchmark code match."""

    def __init__(self, data, model, output="artifacts/benchmark", log=print, open_loop_only=False):
        self.dataset, self.open_loop_only = data.dataset, open_loop_only
        self.path = Path(output) / data.dataset / "probes" / f"{model.name}{' open-loop' if open_loop_only else ''}.npz"
        cfg = data.cfg["data"]
        digest = _weights_digest(model)
        settings = json.dumps({"model": digest, "code": code_digest(), "bundle": str(cfg["bundle_path"]),
                               "splits": {k: cfg.get(k) for k in ("train_runs", "val_runs", "heldout_subjects")},
                               "split_frames": cfg.get("split_frames", {}).get("train")}, sort_keys=True, default=str)
        if digest and self.path.is_file():
            with np.load(self.path, allow_pickle=False) as saved:
                if str(saved["settings"]) == settings:
                    self.meta = json.loads(str(saved["meta"]))
                    self.weights = {g: (saved[f"W {i}"], saved[f"b {i}"]) for i, g in enumerate(self.meta["groups"])}
            if hasattr(self, "weights") and log:
                log(f"probe for {model.name}: loaded {self.path}")
        if not hasattr(self, "weights"):
            self._fit(data, model, log)
            if digest:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.path, settings=settings, meta=json.dumps(self.meta),
                         **{f"{k} {i}": v for i, g in enumerate(self.meta["groups"]) for k, v in zip("Wb", self.weights[g])})
        self.alpha = self.meta["penalty"]
        weights = hashlib.sha256(b"".join(a.tobytes() for g in self.meta["groups"] for a in self.weights[g]))
        self.provenance = {"probe": str(self.path) if digest else None, "weights sha256": weights.hexdigest()[:16],
                           "model weights": digest, "penalty": self.alpha,
                           "latent dimension": self.meta["latent dimension"],
                           "effective rank": round(float(np.median(list(self.meta["effective rank"].values()))), 1)}

    def _fit(self, data, model, log):
        from neuroworld.benchmark import updating as U
        started = time.time()
        grid, H = data.grid, data.grid.horizon_steps
        every = max(1, frames_within(STRIDE_SECONDS, grid.tr) // grid.stride_trs)
        totals, folds, runs = {}, {}, data.runs("train")
        for first in range(0, len(runs), U.RUNS):  # the queries of several runs asked together, as in Updating
            group, pieces = runs[first:first + U.RUNS], []
            open_loop = Q.episodes(group, lambda run: Q.forecast_origins(run, grid)[::every])
            if open_loop:
                batch = Q.joint_batch(data.dataset, grid.tr, open_loop, grid.history_frames(), H)
                pieces.append((open_loop, _states(model, batch), *Q.joint_targets(open_loop, H)))
            updated = [] if self.open_loop_only else Q.episodes(group, lambda run: U.origins(data, run)[::2])
            if updated:
                pieces.append((updated, *U.rollout(data, lambda b: _states(model, b), updated, UPDATE_BUDGET)))
            for episodes, states, y, ok in pieces:
                for run, rows in Q.spans(episodes):
                    key, keep = _group(data.dataset, run.subject_id), ok[rows]
                    z, t = states[rows][keep].astype(np.float64), y[rows][keep].astype(np.float64)
                    _add(totals.setdefault(key, _moments()), z, t)
                    _add(folds.setdefault((key, _fold(run.timeline_id)), _moments()), z, t)
            for run in group:
                run.__dict__.pop("brain", None)
        groups = sorted(totals)
        alpha, validation = _select([(totals[g], [folds[(g, f)] for f in range(FOLDS) if (g, f) in folds]) for g in groups])
        self.weights = {g: tuple(a.astype(np.float32) for a in _solutions(totals[g], [alpha])[0]) for g in groups}
        self.meta = {
            "groups": groups, "penalty": alpha, "validation error by penalty (relative to the best)": validation,
            "latent dimension": int(totals[groups[0]]["G"].shape[0]),
            "effective rank": {g: _effective_rank(totals[g]) for g in groups},
            "samples": {g: int(totals[g]["n"]) for g in groups},
            "fold samples": {g: {str(f): int(folds[(g, f)]["n"]) for f in range(FOLDS) if (g, f) in folds} for g in groups},
            "training recordings": [run.run_id for run in runs],
            "fitting states": f"open-loop rollouts from forecast origins every {STRIDE_SECONDS:g} s and Updating "
                              f"rollouts with {UPDATE_BUDGET} measurements from every other Updating origin; all steps",
            "target": "measured fMRI at the predicted frames (1,000 parcels), valid frames only",
            "preprocessing": "each latent coordinate standardized by its training statistics (inside each "
                             "cross-validation fold too), folded into W",
            "seconds": round(time.time() - started)}
        if log:
            log(f"probe for {getattr(model, 'name', 'model')}: {len(groups)} probe(s), latent dimension "
                f"{self.meta['latent dimension']}, effective rank {np.median(list(self.meta['effective rank'].values())):.1f}, "
                f"penalty {alpha:g} ({self.meta['seconds']}s)")

    def __call__(self, states, subjects):
        out = np.empty((*states.shape[:-1], 1000), dtype=np.float32)
        keys = np.array([_group(self.dataset, s) for s in subjects])
        for key in np.unique(keys):
            rows = np.flatnonzero(keys == key)
            W, b = self.weights[key]
            out[rows] = states[rows] @ W + b
        return out


class LatentProbe:
    """A latent model read out by its frozen probe: an ordinary benchmark model (module docstring)."""

    def __init__(self, model, probe, name):
        self.model, self.probe, self.name, self.provenance = model, probe, name, probe.provenance

    def predict(self, batch):
        return self.probe(_states(self.model, batch), batch.subject_ids)
