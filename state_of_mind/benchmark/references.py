"""Benchmark reference models, fitted on the training split.

stimulus-only ridge    fMRI at frame t from the stimulus rows at t - d for the dataset's delays; no brain input.
direct linear          per prefix length and per step k, the stimulus-only prediction plus a ridge from the
                       prefix frames to its residual at step k (two-stage additive fit). It never consumes
                       its own predictions; with no prefix it is the stimulus-only ridge.

Stimulus rows are the dataset's fixed-input rows, z-scored on training rows, zero (the training mean) before a
run starts and where the stimulus is missing. Algonauts: brain-free, each modality's frozen extractor features
reduced to principal components fitted on the training stimuli (TRIBEv2 was trained on these recordings).
HCP: TRIBEv2's frozen stimulus projections, trained on other datasets' brain data and never on HCP; HCP has
only about 45 minutes of distinct training movie, and 384 brain-free components per modality reach r 0.32
on fitting clips but 0.02 on held-out clips, against 0.09 for the projections (the features of the HCP fMRI forecaster and Transformer world model). Prefix frames enter as whitened spatial principal
components (fitted on training frames), concatenated over the prefix. Algonauts fits one model per
participant; HCP one pooled model (about 150 participants, each with little data). Penalties are chosen
on the validation split.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from state_of_mind.benchmark.metrics import step_scores
from state_of_mind.benchmark import queries as Q
from state_of_mind.benchmark.interface import stimulus_source
from state_of_mind.eval.intersubject import stimulus_key
from state_of_mind.eval.linear_baselines import _raw_rows, _stimulus_rows
from state_of_mind.paths import DATA

DELAYS = {"algonauts": range(3, 9), "hcp_movies": range(3, 9)}  # TRs before the target frame
FEATURES = {"algonauts": "raw", "hcp_movies": "projections"}
ALPHAS = (1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)
BETAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)  # large values let long-horizon history weights vanish
SPATIAL_COMPONENTS, HISTORY_WIDTH = 256, 2048
FIT_STRIDE = 2  # every other training origin for the history moments


def _group(data, subject):
    return "all" if data.dataset == "hcp_movies" else subject


class StimulusRows:
    """Brain-free, z-scored stimulus rows per stimulus, with lagged access."""

    def __init__(self, data, delays, cache=None, features=None):
        """features: "raw" (brain-free extractor PCA), "projections" (TRIBEv2's frozen stimulus projections) or
        "context" (TRIBEv2's frozen causal contextual rows); default per dataset (FEATURES)."""
        self.features = features or FEATURES[data.dataset]
        self.delays = list(delays)
        self.pad = max(self.delays)
        # Rows for every test recording too (no split is locked); standardized on the training recordings alone.
        runs = {split: data.runs(split) for split in ("train", "val") + tuple(s for s in ("dev", "test") if data.cfg["data"][f"{s}_runs"])}
        support = {stimulus_key(r.timeline_id): r.support for rs in runs.values() for r in rs}
        if cache is not None and Path(cache).is_file():
            with np.load(cache) as saved:
                z_rows = {key: saved[key] for key in saved.files}
        else:
            records = {r.run_id: data.records[r.run_id] for rs in runs.values() for r in rs}
            train_keys = {stimulus_key(r.timeline_id) for r in runs["train"]}
            if self.features == "raw":
                rows = _raw_rows(data.bundle, list(records.values()), train_keys)
            else:
                from copy import deepcopy
                cfg = deepcopy(data.cfg)
                cfg["model"]["stimulus_features"] = "tribev2_context" if self.features == "context" else "projections"
                rows = _stimulus_rows(cfg, data.bundle, list(records.values()), features="run")
            stack = np.concatenate([rows[k][support[k]] for k in sorted(train_keys)])
            mean, std = stack.mean(0), stack.std(0) + 1e-8
            z_rows = {}
            for key, value in rows.items():
                z = ((value - mean) / std).astype(np.float32)
                z[~support[key]] = 0.0
                z_rows[key] = z
            if cache is not None:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, **z_rows)
        missing = set(support) - set(z_rows)
        if missing:
            raise ValueError(f"Stimulus-row cache {cache} lacks {sorted(missing)[:3]}; delete it to rebuild")
        self.width = next(iter(z_rows.values())).shape[1] * len(self.delays)
        # Column blocks per modality, for per-modality stimulus histories (contextual rows fuse modalities).
        columns = next(iter(z_rows.values())).shape[1]
        order = {"raw": ("audio", "text", "video"), "projections": ("text", "audio", "video")}.get(self.features)
        self.blocks = ({m: slice(k * columns // 3, (k + 1) * columns // 3) for k, m in enumerate(order)}
                       if order and columns % 3 == 0 else {None: slice(0, columns)})
        self.rows = {key: np.concatenate([np.zeros((self.pad, z.shape[1]), dtype=np.float32), z])
                     for key, z in z_rows.items()}

    def _rows(self, timelines, frames):
        """Rows of (timeline, frame) pairs; zero before and after each recording."""
        out = np.zeros((len(frames), next(iter(self.rows.values())).shape[1]), dtype=np.float32)
        for timeline in dict.fromkeys(timelines):
            x = self.rows[stimulus_key(timeline)]
            rows = np.flatnonzero(timelines == timeline)
            index = frames[rows] + self.pad
            inside = (index >= 0) & (index < len(x))
            out[rows[inside]] = x[index[inside]]
        return out

    def lagged(self, timeline_id, frames, source=None):
        """[len(frames), width] rows at frames - d for every delay. ``source(stimulus_frames, modality)`` maps
        stimulus frames to (timelines, frames) per modality block for a remapped query; by default the query's
        own timeline."""
        frames = np.asarray(frames)
        if source is None:
            own = lambda f: (np.full(len(f), timeline_id, dtype=object), f)
            return np.concatenate([self._rows(*own(frames - d)) for d in self.delays], axis=1)
        columns_total = next(iter(self.rows.values())).shape[1]
        blocks = getattr(self, "blocks", None) or {None: slice(0, columns_total)}
        out = []
        for d in self.delays:
            block = np.zeros((len(frames), columns_total), dtype=np.float32)
            for modality, columns in blocks.items():
                block[:, columns] = self._rows(*source(frames - d, modality))[:, columns]
            out.append(block)
        return np.concatenate(out, axis=1)


def _fit_targets(run):
    """Training frames that may be fitting targets."""
    return np.flatnonzero(run.brain_ok & run.target_ok)


def _pearson_runs(data, runs, predict_frames):
    """Mean over runs of the per-run parcel Pearson on scoreable frames (encoder selection)."""
    from state_of_mind.benchmark.metrics import _parcel_pearson
    values = []
    for run in runs:
        frames = _fit_targets(run)
        if len(frames) > 20:
            values.append(_parcel_pearson(predict_frames(run, frames), np.asarray(run.brain[frames], dtype=np.float64)))
    return float(np.nanmean(values))


def feature_tag(data, features):
    """Name suffix for references fitted on non-default stimulus rows, e.g. " [context]"."""
    return "" if not features or features == FEATURES[data.dataset] else f" [{features}]"


class StimulusRidge:
    name = "stimulus-only ridge"

    def __init__(self, data, rows=None, cache=None, log=print, features=None):
        self.data = data
        self.rows = rows or StimulusRows(data, DELAYS[data.dataset], cache, features)
        self.name = "stimulus-only ridge" + feature_tag(data, self.rows.features)
        started = time.time()
        moments = {}
        for run in data.runs("train"):
            t = _fit_targets(run)
            X = torch.from_numpy(self.rows.lagged(run.timeline_id, t).astype(np.float64))
            Y = torch.from_numpy(np.asarray(run.brain[t], dtype=np.float64))
            m = moments.setdefault(_group(data, run.subject_id), {
                "G": torch.zeros(self.rows.width, self.rows.width, dtype=torch.float64),
                "C": torch.zeros(self.rows.width, 1000, dtype=torch.float64),
                "x": torch.zeros(self.rows.width, dtype=torch.float64), "y": torch.zeros(1000, dtype=torch.float64), "n": 0})
            m["G"] += X.T @ X; m["C"] += X.T @ Y; m["x"] += X.sum(0); m["y"] += Y.sum(0); m["n"] += len(t)
        self.solvers = {}
        for group, m in moments.items():
            n = m["n"]
            mx, my = m["x"] / n, m["y"] / n
            G = m["G"] - n * torch.outer(mx, mx)
            C = m["C"] - n * torch.outer(mx, my)
            lam, V = torch.linalg.eigh(G)
            self.solvers[group] = (mx.numpy(), my.numpy(), lam.numpy(), V.numpy(), (V.T @ C).numpy())
        val = data.runs("val")[::4]
        scores = {}
        for alpha in ALPHAS:
            self._set(alpha)
            scores[alpha] = _pearson_runs(data, val, lambda run, frames: self.predict_frames(run.subject_id, run.timeline_id, frames))
        self.alpha = max(scores, key=lambda a: scores[a])
        self._set(self.alpha)
        self.selection = {"alpha": self.alpha, "validation_pearson": scores}
        if log:
            log(f"{self.name}: alpha {self.alpha:g} ({time.time() - started:.0f}s) {scores}")

    def _set(self, alpha):
        self.weights = {g: (mx, my, V @ (VC / (lam + alpha)[:, None])) for g, (mx, my, lam, V, VC) in self.solvers.items()}

    def predict_frames(self, subject, timeline_id, frames, source=None):
        mx, my, W = self.weights[_group(self.data, subject)]
        return (self.rows.lagged(timeline_id, frames, source) - mx) @ W + my

    def predict(self, batch):
        steps = np.arange(batch.horizon)
        out = np.empty((len(batch), batch.horizon, 1000), dtype=np.float32)
        plain = [i for i in range(len(batch)) if not batch.remapped(i)]
        keys = [(batch.subject_ids[i], batch.timeline_ids[i]) for i in plain]
        for key in dict.fromkeys(keys):
            rows = [i for i, k in zip(plain, keys) if k == key]
            frames = (batch.origins[rows][:, None] + steps).reshape(-1)
            out[rows] = self.predict_frames(*key, frames).reshape(len(rows), batch.horizon, 1000)
        for i in range(len(batch)):
            if batch.remapped(i):
                source = lambda f, m, i=i: stimulus_source(batch, i, f, m)
                out[i] = self.predict_frames(batch.subject_ids[i], batch.timeline_ids[i], batch.origins[i] + steps, source)
        return out


class SpatialBasis:
    """Whitened spatial principal components of training frames, pooled over participants."""

    def __init__(self, data, components=SPATIAL_COMPONENTS):
        total, square, count = np.zeros(1000), np.zeros((1000, 1000)), 0
        for run in data.runs("train"):
            frames = _fit_targets(run)[::4]
            x = np.asarray(run.brain[frames], dtype=np.float64)
            total += x.sum(0); square += x.T @ x; count += len(x)
        self.mean = total / count
        values, vectors = np.linalg.eigh(square / count - np.outer(self.mean, self.mean))
        order = np.argsort(values)[::-1][:components]
        self.projection = (vectors[:, order] / np.sqrt(values[order])).astype(np.float32)

    def features(self, frames_by_query, width):
        """[B, F * C] from [B, F, 1000] prefix frames, C = min(components, width // F)."""
        b, f, _ = frames_by_query.shape
        c = min(self.projection.shape[1], width // f)
        return ((frames_by_query - self.mean.astype(np.float32)) @ self.projection[:, :c]).reshape(b, f * c)


def _moments(width, horizon, outputs):
    """Empty sufficient statistics of the per-step ridges (_accumulate)."""
    return {"hh": np.zeros((width, width)), "miss": np.zeros((horizon, width, width)), "hr": np.zeros((horizon, width, outputs)),
            "hk": np.zeros((horizon, width)), "rk": np.zeros((horizon, outputs)), "m": np.zeros(horizon)}


def _accumulate(g, h, r, ok):
    """Add anchors h [N, D] with step targets r [N, H, outputs] (zero where missing) and their mask ok [N, H]. Each
    step's ridge uses only the anchors that have its target: the Gram matrix of all anchors minus that of the anchors
    missing the step (few, near run and segment ends), so every step solves its own normal equations."""
    g["hh"] += h.T @ h
    for k in np.flatnonzero((~ok).any(0)):
        missing = h[~ok[:, k]]
        g["miss"][k] += missing.T @ missing
    g["hr"] += (h.T @ r.reshape(len(h), -1)).reshape(h.shape[1], ok.shape[1], -1).transpose(1, 0, 2)
    g["hk"] += ok.T.astype(np.float64) @ h
    g["rk"] += r.sum(0)
    g["m"] += ok.sum(0)


def _solve(g):
    """Per step: eigendecomposition of the centered Gram matrix of its own anchors, and the rotated cross moments."""
    m = np.maximum(g["m"], 1)
    mean_h, mean_r = g["hk"] / m[:, None], g["rk"] / m[:, None]
    gram = g["hh"][None] - g["miss"] - m[:, None, None] * mean_h[:, :, None] * mean_h[:, None, :]
    lam, V = np.linalg.eigh(gram)  # [H, D], [H, D, D]
    cross = g["hr"] - m[:, None, None] * mean_h[:, :, None] * mean_r[:, None, :]
    return lam, V, np.matmul(np.swapaxes(V, 1, 2), cross), mean_h, mean_r


class LinearForecaster:
    """Direct linear forecaster: a stimulus model plus per-step ridges from the prefix to its residual."""

    def __init__(self, data, stimulus, basis=None, log=print, windows=None):
        """windows: {name: prefix frames} to fit (default: the grid's 30-s prefix)."""
        self.data, self.stimulus = data, stimulus
        if hasattr(stimulus, "rows"):
            self.name = "direct linear" + feature_tag(data, stimulus.rows.features)
        else:  # a pretrained encoder's predictions as the stimulus part (e.g. TRIBEv2 plus a residual forecaster)
            self.name = f"{stimulus.name} + residual linear"
        self.basis = basis or SpatialBasis(data)
        self.horizon = data.grid.horizon_steps
        self.levels = {name: frames for name, frames in (windows or data.grid.prefixes()).items() if frames > 0}
        self.longest = max(self.levels.values())
        # Fitting targets and their residuals after the stimulus prediction, computed once for all prefix lengths.
        self.targets = {}
        for run in data.runs("train"):
            frames = _fit_targets(run)
            self.targets[run.run_id] = (frames, self._residual(run, frames).astype(np.float32))
        self.fits, self.selection = {}, {}
        for name, frames in self.levels.items():
            started = time.time()
            self.fits[frames] = self._fit(frames)
            self.selection[name] = self._select(frames)
            if log:
                chosen = self.selection[name]["beta_by_step"]
                log(f"{self.name} {name}: per-step beta, median {self.selection[name]['beta']:g}, first/last steps "
                    f"{chosen[0]:g}/{chosen[-1]:g} ({time.time() - started:.0f}s)")

    def _residual(self, run, frames):
        y = np.asarray(run.brain[frames], dtype=np.float64)
        return y - self.stimulus.predict_frames(run.subject_id, run.timeline_id, frames)

    def _fit(self, frames):
        """Per group: eigendecomposition of the centered history moments and per-step cross moments."""
        H = self.horizon
        stats = {}
        for run in self.data.runs("train"):
            anchors, prefix = self._prefixes(run, frames)
            if not len(anchors):
                continue
            h = self._features(prefix).astype(np.float64)
            frames_ok, values = self.targets[run.run_id]
            target = np.zeros(run.n, dtype=bool)
            target[frames_ok] = True
            residual = np.zeros((run.n, 1000))
            residual[frames_ok] = values
            g = stats.setdefault(_group(self.data, run.subject_id), _moments(h.shape[1], H, residual.shape[1]))
            # Every step at once: targets t0 + k for k < H, zero where not a fitting target.
            t = anchors[:, None] + np.arange(H)[None]
            ok = t < run.n
            ok[ok] = target[t[ok]]
            r = residual[np.minimum(t, run.n - 1)] * ok[..., None]  # [anchors, H, 1000]
            _accumulate(g, h, r, ok)
        return {group: _solve(g) for group, g in stats.items()}

    def _prefixes(self, run, frames):
        """Training anchors whose whole prefix is valid training frames, and their prefixes [anchors, frames, 1000]."""
        ok_prefix = run.brain_ok & run.allowed
        bad = np.r_[0, np.cumsum(~ok_prefix)]
        anchors = np.arange(frames, run.n, FIT_STRIDE)
        anchors = anchors[bad[anchors] == bad[anchors - frames]]
        return anchors, np.stack([np.asarray(run.brain[t0 - frames:t0], dtype=np.float32) for t0 in anchors]) if len(anchors) else None

    def _features(self, prefix):
        return self.basis.features(prefix, HISTORY_WIDTH)

    def _weights(self, frames, beta):
        out = {}
        for group, (lam, V, VC, mean_h, mean_r) in self.fits[frames].items():
            W = np.matmul(V, VC / (lam + beta)[:, :, None])  # [H, D, 1000]
            out[group] = (W.astype(np.float32), (mean_r - np.matmul(mean_h[:, None, :], W)[:, 0]).astype(np.float32))
        return out

    def _select(self, frames):
        """Penalty per step: the lowest per-step normalized MSE on a quarter of the validation runs. History weights
        must shrink toward zero at long horizons, where one penalty for every step adds noise. MSE, not Pearson:
        on whitened prefix features a penalty mostly rescales the prediction, which Pearson ignores."""
        runs = self.data.runs("val")[::4]
        name = next(n for n, f in self.levels.items() if f == frames)
        candidates, scores = {}, {}
        for beta in BETAS:
            candidates[beta] = self._weights(frames, beta)
            self.weights = {**getattr(self, "weights", {}), frames: candidates[beta]}
            per_step = []
            for run in runs:
                origins = Q.forecast_origins(run, self.data.grid)
                if len(origins):
                    observed, ok = Q.targets(run, origins, self.horizon)
                    predicted = self.predict(Q.batch(self.data.dataset, self.data.grid.tr, run, origins, frames, self.horizon))
                    per_step.append(step_scores(predicted, observed, ok, origins, self.data.grid.tr)["nmse"])
            scores[beta] = np.nanmean(np.stack(per_step), axis=0)
        table = np.stack([scores[b] for b in BETAS])  # [betas, H]
        best = [BETAS[i] for i in np.nanargmin(np.where(np.isfinite(table), table, np.inf), axis=0)]
        self.weights[frames] = {group: (np.stack([candidates[b][group][0][k] for k, b in enumerate(best)]),
                                        np.stack([candidates[b][group][1][k] for k, b in enumerate(best)]))
                                for group in candidates[BETAS[0]]}
        return {"prefix": name, "beta": float(np.median(best)), "beta_by_step": best,
                "validation_nmse": {str(b): float(np.nanmean(scores[b])) for b in BETAS},
                "selected_nmse": float(np.nanmean(np.nanmin(table, axis=0)))}

    def predict(self, batch):
        width = batch.prefix.shape[1]
        horizon = batch.horizon
        if horizon > self.horizon:
            raise ValueError(f"{self.name} was fitted to {self.horizon} steps")
        base = self.stimulus.predict(batch)
        if width == 0:
            return base
        # The longest fitted window serves longer histories, so memory ends there (30 s by default).
        frames = min(width, self.longest) if width not in self.weights else width
        if frames not in self.weights:
            frames = max(f for f in self.weights if f <= frames)
        prefix = batch.prefix[:, width - frames:]
        valid = batch.prefix_valid[:, width - frames:]
        # A missing frame enters as the training mean (Updating reveals only some frames).
        prefix = np.where(valid[..., None], np.nan_to_num(prefix), self.basis.mean.astype(np.float32))
        h = self._features(prefix)
        out = base.copy()
        groups = np.array([_group(self.data, s) for s in batch.subject_ids])
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            W, b = self.weights[frames][group]
            out[rows] += np.tensordot(h[rows], W[:horizon], axes=([1], [1])) + b[:horizon]
        return out


def direct_linear(data, features=None, cache=None, log=print, refit=False):
    """The direct linear forecaster (the Forecast and State baseline) with its stimulus ridge. The fit is a shared asset,
    like IPSID's: stored in BASELINES_ROOT/<bundle>/<name>/model.npz with what it depends on (the settings and this
    module's code), reused while they match and refitted otherwise. cache: the stimulus-row cache."""
    import hashlib
    from state_of_mind.benchmark.baselines import BASELINES_ROOT
    features = features or FEATURES[data.dataset]
    name = "direct linear" + feature_tag(data, features)
    path = BASELINES_ROOT / Path(data.cfg["data"]["bundle_path"]).name / name / "model.npz"
    cfg = data.cfg["data"]
    settings = json.dumps({"bundle": str(cfg["bundle_path"]), "dataset": data.dataset, "features": features,
                           "tr": data.grid.tr, "hrf_delay_seconds": cfg.get("hrf_delay_seconds"),
                           "splits": {k: cfg.get(k) for k in ("train_runs", "val_runs", "dev_runs", "heldout_subjects")},
                           "split_frames": {k: v for k, v in cfg.get("split_frames", {}).items() if k in ("train", "val")},
                           "prefixes": data.grid.prefixes(), "horizon": data.grid.horizon_steps,
                           "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]}, sort_keys=True, default=str)
    rows = StimulusRows(data, DELAYS[data.dataset], cache, features)
    if path.is_file() and not refit:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["settings"]) == settings:
                model = _restore(data, rows, saved)
                if log:
                    log(f"{name}: loaded {path}")
                return model
        if log:
            log(f"{name}: {path} was fitted with other settings or code; refitting")
    model = LinearForecaster(data, StimulusRidge(data, rows=rows, log=log), log=log)
    stimulus, groups = model.stimulus, sorted(model.stimulus.weights)
    arrays = {f"stimulus {i} {k}": v for i, g in enumerate(groups) for k, v in zip(("mx", "my", "W"), stimulus.weights[g])}
    arrays.update({f"history {frames} {i} {k}": v for frames, by_group in model.weights.items()
                   for i, g in enumerate(groups) for k, v in zip(("W", "b"), by_group[g])})
    meta = {"groups": groups, "levels": model.levels, "stimulus": stimulus.selection, "history": model.selection}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, settings=settings, meta=json.dumps(meta, default=float), basis_mean=model.basis.mean,
             basis_projection=model.basis.projection, **arrays)
    if log:
        log(f"{name}: stored {path}")
    return model


def _restore(data, rows, saved):
    """A LinearForecaster with its StimulusRidge from a stored fit (direct_linear)."""
    meta = json.loads(str(saved["meta"]))
    groups = meta["groups"]
    stimulus = StimulusRidge.__new__(StimulusRidge)
    stimulus.data, stimulus.rows, stimulus.name = data, rows, "stimulus-only ridge" + feature_tag(data, rows.features)
    stimulus.selection, stimulus.alpha = meta["stimulus"], meta["stimulus"]["alpha"]
    stimulus.weights = {g: tuple(saved[f"stimulus {i} {k}"] for k in ("mx", "my", "W")) for i, g in enumerate(groups)}
    model = LinearForecaster.__new__(LinearForecaster)
    model.data, model.stimulus, model.name = data, stimulus, "direct linear" + feature_tag(data, rows.features)
    model.basis = SpatialBasis.__new__(SpatialBasis)
    model.basis.mean, model.basis.projection = saved["basis_mean"], saved["basis_projection"]
    model.horizon, model.levels, model.selection = data.grid.horizon_steps, meta["levels"], meta["history"]
    model.longest = max(model.levels.values())
    model.weights = {frames: {g: tuple(saved[f"history {frames} {i} {k}"] for k in ("W", "b")) for i, g in enumerate(groups)}
                     for frames in model.levels.values()}
    return model


TRIBEV2_PREDICTIONS = {"algonauts": DATA / "algonauts/derivatives/neuroworld-schaefer1000-v2-tribev2-predictions",
                       "hcp_movies": DATA / "hcp_movies/derivatives/neuroworld-schaefer1000-v2-tribev2-predictions"}


class SavedEncoder:
    """Per-stimulus encoder predictions cached at the bundle's TR (eval/tribev2_reference.py): TRIBEv2 zero-shot,
    average-subject head. Open track and non-causal (bidirectional attention within 100-s segments); on Algonauts
    it was trained on these recordings. It ignores the prefix; frames without a prediction are zero."""
    name = "TRIBEv2 zero-shot"

    def __init__(self, data, directory=None):
        root = Path(directory or TRIBEV2_PREDICTIONS[data.dataset])
        index = json.loads((root / "index.json").read_text())
        self.paths = {key: root / entry["path"] for key, entry in index["timelines"].items()}
        self._arrays = {}

    def _array(self, timeline):
        # Algonauts caches are keyed by the full timeline (sub-XX:...), HCP's by the shared stimulus.
        key = timeline if timeline in self.paths else stimulus_key(timeline)
        if key not in self._arrays:
            self._arrays[key] = np.nan_to_num(np.load(self.paths[key]).astype(np.float32))
        return self._arrays[key]

    def predict_frames(self, subject, timeline_id, frames, source=None):
        """Cached predictions at the frames (zero outside the cache), for residual fitting."""
        array = self._array(timeline_id)
        frames = np.asarray(frames)
        out = np.zeros((len(frames), array.shape[1]), dtype=np.float32)
        inside = (frames >= 0) & (frames < len(array))
        out[inside] = array[frames[inside]]
        return out

    def predict(self, batch):
        """For a spliced query, frames whose response window (5 s earlier) lies after the splice come from the
        other timeline's predictions, which were computed in that recording's own context (approximate)."""
        out = np.zeros((len(batch), batch.horizon, 1000), dtype=np.float32)
        offset = int(round(5.0 / batch.tr))
        for i, t0 in enumerate(batch.origins):
            frames = np.arange(t0, t0 + batch.horizon)
            timelines, mapped = stimulus_source(batch, i, frames - offset)
            mapped = mapped + offset
            for timeline in dict.fromkeys(timelines):
                array = self._array(timeline)
                rows = np.flatnonzero((timelines == timeline) & (mapped < len(array)))
                out[i, rows] = array[mapped[rows]]
        return out
