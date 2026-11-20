"""IPSID baseline (docs/benchmark.md#tasks), scored like any submission.

IPSID is an input-driven linear state-space model identified by subspace identification (Vahidi, Sani & Shanechi,
PNAS 2024; the PSID package), without a behavioural signal (n1 = 0: the non-preferential input-driven case). One
model per participant on Algonauts, one shared by all participants on HCP (new participants use it unchanged), on
brain and stimulus principal components. Its steady-state Kalman filter estimates the state
over the prefix, skipping the measurement update at missing frames; the state is then propagated open loop under
the stimulus. It reads the 30-s prefix and the stimulus rows of the chosen features at the HRF lag (remapped per
modality for history-swapped queries), predicts in a pooled spatial PCA basis and returns Schaefer-1000 fMRI.
Fits are cached under BASELINES_ROOT/<bundle name>/<model>.

    python -m neuroworld.benchmark suite --model ipsid --features projections
"""
import json
from pathlib import Path
import time

import numpy as np

from neuroworld.benchmark.grid import LONGEST_PREFIX
from neuroworld.benchmark.interface import stimulus_source
from neuroworld.benchmark.references import FEATURES, StimulusRows, feature_tag
from neuroworld.paths import DATA
import os

BASELINES_ROOT = DATA / "neuralworld-checkpoints/baselines"
PYLIB = os.environ.get("PSID_PYLIB", str(DATA / "pylib"))  # where PSID is installed (pip --target), if not in the environment
BRAIN_COMPONENTS = 256


def _lag(data):
    return int(np.floor(data.cfg["data"]["hrf_delay_seconds"] / data.grid.tr + 0.5))


class BrainBasis:
    """Pooled (unwhitened) spatial principal components of training frames: an orthonormal map, so squared
    error in components equals parcel-space squared error within the basis."""

    def __init__(self, data, components=BRAIN_COMPONENTS, mean=None, vectors=None):
        if vectors is None:
            total, square, count = np.zeros(1000), np.zeros((1000, 1000)), 0
            for run in data.runs("train"):
                frames = np.flatnonzero(run.brain_ok & run.allowed)[::4]
                x = np.asarray(run.brain[frames], dtype=np.float64)
                total += x.sum(0); square += x.T @ x; count += len(x)
                run.__dict__.pop("brain", None)
            mean = total / count
            values, vectors = np.linalg.eigh(square / count - np.outer(mean, mean))
            vectors = vectors[:, np.argsort(values)[::-1][:components]]
        self.mean, self.vectors = np.asarray(mean, dtype=np.float32), np.asarray(vectors, dtype=np.float32)

    def to(self, y):
        return (np.asarray(y, dtype=np.float32) - self.mean) @ self.vectors

    def back(self, z):
        return np.asarray(z, dtype=np.float32) @ self.vectors.T + self.mean


class StimulusInputs:
    """Stimulus rows at the HRF lag, optionally reduced to principal components fitted on training rows."""

    def __init__(self, data, features, components=None, cache_root="artifacts/benchmark", projection=None):
        self.features = features or FEATURES[data.dataset]
        self.rows = StimulusRows(data, [_lag(data)], cache=Path(cache_root) / data.dataset / f"stimulus-rows-{self.features}.npz",
                                 features=self.features)
        self.projection = projection
        if components and projection is None:
            stack = np.concatenate([self.run_rows(r) for r in data.runs("train")[::3]])[::2]
            mean = stack.mean(0)
            values, vectors = np.linalg.eigh(np.cov(stack - mean, rowvar=False))
            self.projection = (mean.astype(np.float32), vectors[:, np.argsort(values)[::-1][:components]].astype(np.float32))

    def _reduce(self, x):
        return x if self.projection is None else (x - self.projection[0]) @ self.projection[1]

    def run_rows(self, run):
        return self.rows.lagged(run.timeline_id, np.arange(run.n)).astype(np.float32)

    def run_inputs(self, run):
        return self._reduce(self.run_rows(run))

    def query(self, batch, i, frames):
        source = (lambda f, m: stimulus_source(batch, i, f, m)) if batch.remapped(i) else None
        return self._reduce(self.rows.lagged(batch.timeline_ids[i], np.asarray(frames), source).astype(np.float32))

    @property
    def width(self):
        return self.rows.width if self.projection is None else self.projection[1].shape[1]


def _query_inputs(batch, stimulus):
    """([B, P, width] stimulus at the prefix frames, [B, H, width] at the target frames)."""
    width = batch.prefix.shape[1]
    u_pre, u_fut = [], []
    for i, t0 in enumerate(batch.origins):
        u = stimulus.query(batch, i, np.arange(t0 - width, t0 + batch.horizon))
        u_pre.append(u[:width]); u_fut.append(u[width:])
    return np.stack(u_pre), np.stack(u_fut)


def _prefix_components(batch, basis):
    prefix = np.where(batch.prefix_valid[..., None], batch.prefix, basis.mean)
    return basis.to(prefix)


# ----------------------------------------------------------------------------------------------- IPSID

class IPSIDBaseline:
    """Input-driven linear state-space model (module docstring)."""

    STATES = (16, 32)
    BLOCK_ROWS = 10
    # Subspace identification builds (2 x block rows x channels) x samples matrices: 64 brain and 32 input
    # components and every other training run keep a fit to minutes per participant.
    INPUT_COMPONENTS = 32
    BRAIN_COMPONENTS = 64
    RUN_EVERY = 2

    def __init__(self, data, features=None, log=print, refit=False, output="artifacts/benchmark"):
        import sys
        if PYLIB not in sys.path:
            sys.path.append(PYLIB)
        self.data = data
        self.features = features or FEATURES[data.dataset]
        self.name = "IPSID" + feature_tag(data, self.features)
        folder = BASELINES_ROOT / Path(data.cfg["data"]["bundle_path"]).name / self.name
        path = folder / "model.npz"
        settings = self._settings()
        if path.is_file() and not refit:
            saved = np.load(path, allow_pickle=False)
            if "settings" in saved.files and str(saved["settings"]) == settings:
                self.basis = BrainBasis(data, mean=saved["basis_mean"], vectors=saved["basis_vectors"])
                self.stimulus = StimulusInputs(data, self.features, cache_root=output,
                                               projection=(saved["u_mean"], saved["u_vectors"]))
                self.params = json.loads(str(saved["params"]))
                self.params = {s: {k: np.asarray(v) for k, v in p.items()} for s, p in self.params.items()}
                if log:
                    log(f"{self.name}: loaded {path}")
                return
            if log:
                log(f"{self.name}: {path} was fitted with other settings (or none recorded); refitting")
        started = time.time()
        self.basis = BrainBasis(data, components=self.BRAIN_COMPONENTS)
        self.stimulus = StimulusInputs(data, self.features, components=self.INPUT_COMPONENTS, cache_root=output)
        self.params, choice = {}, {}
        by_subject = {}
        for run in data.runs("train"):
            by_subject.setdefault(self._group(run.subject_id), []).append(run)
        for subject, runs in sorted(by_subject.items()):
            ys, us = self._segments(runs[::self.RUN_EVERY])
            vals = [r for r in data.runs("val") if self._group(r.subject_id) == subject]
            best = None
            ws = {}
            for nx in self.STATES:
                model, ws = self._identify(ys, us, nx, ws)
                p = self._extract(model)
                score = self._validation_error(p, vals)
                if log:
                    log(f"{self.name} {subject}: nx {nx} validation 20-step error {score:.4f}")
                if best is None or score < best[0]:
                    best = (score, nx, p)
            self.params[subject], choice[subject] = best[2], best[1]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(path, basis_mean=self.basis.mean, basis_vectors=self.basis.vectors, u_mean=self.stimulus.projection[0],
                 u_vectors=self.stimulus.projection[1], settings=settings,
                 params=json.dumps({s: {k: v.tolist() for k, v in p.items()} for s, p in self.params.items()}))
        (folder / "fit.json").write_text(json.dumps({"states": choice, "seconds": time.time() - started}) + "\n")
        if log:
            log(f"{self.name}: fitted ({choice}) in {time.time() - started:.0f}s")

    def _settings(self):
        """What the fit depends on (the full bundle path, the TR and stimulus lag, the splits, the features and the model
        sizes), stored with the fit: a cached fit is reused only when they match."""
        cfg = self.data.cfg["data"]
        return json.dumps({"bundle": str(cfg["bundle_path"]), "dataset": self.data.dataset, "features": self.features,
                           "tr": self.data.grid.tr, "hrf_delay_seconds": cfg.get("hrf_delay_seconds"), "lag": _lag(self.data),
                           "splits": {k: cfg.get(k) for k in ("train_runs", "val_runs", "heldout_subjects")},
                           "split_frames": {k: v for k, v in cfg.get("split_frames", {}).items() if k in ("train", "val")},
                           "states": list(self.STATES), "brain": self.BRAIN_COMPONENTS, "inputs": self.INPUT_COMPONENTS,
                           "run_every": self.RUN_EVERY, "block_rows": self.BLOCK_ROWS}, sort_keys=True, default=str)

    def _group(self, subject):
        """Algonauts: one model per participant; HCP: one model for every participant."""
        return "all" if self.data.dataset == "hcp_movies" else subject

    def _segments(self, runs):
        """Contiguous valid stretches (>= 4 block-rows) of brain components and inputs."""
        ys, us = [], []
        for run in runs:
            ok = run.brain_ok & run.allowed
            y = self.basis.to(np.asarray(run.brain[:], dtype=np.float32))
            u = self.stimulus.run_inputs(run)
            edges = np.diff(np.r_[0, ok.astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if b - a >= 4 * self.BLOCK_ROWS:
                    ys.append(y[a:b].astype(np.float64)); us.append(u[a:b].astype(np.float64))
            run.__dict__.pop("brain", None)
        return ys, us

    def _identify(self, ys, us, nx, ws):
        import PSID
        model, ws = PSID.IPSID(ys, None, us, nx=nx, n1=0, i=self.BLOCK_ROWS, WS=ws, return_WS=True)
        return model, ws

    @staticmethod
    def _extract(model):
        p = {k: np.asarray(getattr(model, k), dtype=np.float64) for k in ("A", "B", "C", "D", "K")}
        y_mean = getattr(getattr(model, "YPrepModel", None), "dataMean", None)
        u_mean = getattr(getattr(model, "UPrepModel", None), "dataMean", None)
        p["y_mean"] = np.zeros(p["C"].shape[0]) if y_mean is None else np.asarray(y_mean, dtype=np.float64).ravel()
        p["u_mean"] = np.zeros(p["B"].shape[1]) if u_mean is None else np.asarray(u_mean, dtype=np.float64).ravel()
        return p

    @staticmethod
    def _filter_and_roll(p, y_pre, u_pre, u_fut, use_input=True, valid=None):
        """Steady-state Kalman filter over the prefix (one-step predicted state), then open-loop propagation. A frame
        that is not valid gets no measurement update (the state is only propagated through it).
        Returns (component predictions [B, H, K], states [B, H, nx], origin state [B, nx])."""
        A, B, C, D, K = p["A"], p["B"], p["C"], p["D"], p["K"]
        y_pre = y_pre - p["y_mean"]
        u_pre = (u_pre - p["u_mean"]) if use_input else np.zeros_like(u_pre)
        u_fut = u_fut - p["u_mean"]
        x = np.zeros((len(y_pre), A.shape[0]))
        for t in range(y_pre.shape[1]):
            innovation = y_pre[:, t] - x @ C.T - (u_pre[:, t] @ D.T if D.size else 0.0)
            if valid is not None:
                innovation = innovation * valid[:, t, None]
            x = x @ A.T + u_pre[:, t] @ B.T + innovation @ K.T
        origin = x.copy()
        out = np.zeros((len(x), u_fut.shape[1], C.shape[0])); states = np.zeros((len(x), u_fut.shape[1], A.shape[0]))
        for k in range(u_fut.shape[1]):
            states[:, k] = x
            out[:, k] = x @ C.T + (u_fut[:, k] @ D.T if D.size else 0.0) + p["y_mean"]
            x = x @ A.T + u_fut[:, k] @ B.T
        return out, states, origin

    def _validation_error(self, p, runs, steps=20, stride=10):
        width = self.data.grid.prefixes()[LONGEST_PREFIX]
        err, count = 0.0, 0
        for run in runs[::2]:
            ok = run.brain_ok & run.allowed
            y = self.basis.to(np.asarray(run.brain[:], dtype=np.float32)).astype(np.float64)
            u = self.stimulus.run_inputs(run).astype(np.float64)
            starts = [t for t in range(width, run.n - steps, stride) if ok[t - width:t + steps].all()]
            if starts:
                idx = np.array(starts)
                y_pre = np.stack([y[t - width:t] for t in idx]); u_pre = np.stack([u[t - width:t] for t in idx])
                u_fut = np.stack([u[t:t + steps] for t in idx]); y_fut = np.stack([y[t:t + steps] for t in idx])
                pred, _, _ = self._filter_and_roll(p, y_pre, u_pre, u_fut)
                err += float(((pred - y_fut) ** 2).sum()); count += y_fut.size
            run.__dict__.pop("brain", None)
        return err / max(count, 1)

    def _run(self, batch, use_input=True):
        groups = [self._group(s) for s in batch.subject_ids]
        if len(set(groups)) > 1:  # queries of several participants: each with its own model
            parts = [(rows, self._run(batch.take(rows), use_input)) for rows in
                     ([i for i, g in enumerate(groups) if g == key] for key in dict.fromkeys(groups))]
            # Each participant's model has its own state width (identified per participant), so latent outputs are
            # zero-padded to the widest; the fMRI forecasts have the same width for everyone.
            widths = [np.max([np.shape(values[i]) for _, values in parts], axis=0)[1:] for i in range(len(parts[0][1]))]
            out = tuple(np.zeros((len(batch), *w)) for w in widths)
            for rows, values in parts:
                for whole, v in zip(out, values):
                    whole[(rows, *[slice(0, n) for n in np.shape(v)[1:]])] = v
            return out
        p = self.params[groups[0]]
        y_pre = _prefix_components(batch, self.basis).astype(np.float64)
        u_pre, u_fut = _query_inputs(batch, self.stimulus)
        return self._filter_and_roll(p, y_pre, u_pre.astype(np.float64), u_fut.astype(np.float64), use_input,
                                     valid=batch.prefix_valid.astype(np.float64))

    def predict(self, batch):
        return self.predict_states(batch)[0]

    def predict_states(self, batch):
        out, states, _ = self._run(batch)
        return self.basis.back(out).astype(np.float32), states.astype(np.float32)

    def state(self, batch):
        """Brain-only state: the filter over the prefix with the inputs held at their mean."""
        return self._run(batch, use_input=False)[2].astype(np.float32)
