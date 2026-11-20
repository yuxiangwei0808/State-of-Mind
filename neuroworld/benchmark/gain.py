"""Gain: how strongly a brain series follows the movie (the State task's gain readout, docs/benchmark.md#tasks).

The gain of a series in a window is sum(y * s~) / sum(s~^2) over a parcel set, where s~ is a fixed reference
prediction (the stimulus-only ridge on TRIBEv2 projections, the own participant's weights) with its mean and linear
trend removed per parcel within the window. A constant series has gain 0. Parcel sets: the stimulus-driven parcels
(top quarter of the reference's fit on training runs) and, per Yeo network, that network's parcels above the median.
A window whose detrended reference is flat (energy below FLOOR of the typical) has no gain.
"""
from pathlib import Path

import numpy as np

from neuroworld.benchmark.grid import frames_within
from neuroworld.benchmark.references import DELAYS, StimulusRidge, StimulusRows
from neuroworld.paths import DATA

WINDOW_SECONDS = 30.0
MIN_VALID = 0.8      # a window needs this fraction of valid frames
MIN_PARCELS = 10     # smaller network sets are dropped
FLOOR = 1e-4         # a window whose detrended reference energy per cell is below FLOOR x the typical one has no gain
NETWORKS_TSV = Path(DATA / "templateflow/tpl-MNI152NLin2009cAsym/"
                    "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.tsv")


class Reference:
    """The fixed reference prediction and parcel sets, fitted once per dataset and cached."""

    def __init__(self, data, output="artifacts/benchmark", log=print):
        self.data = data
        folder = Path(output) / data.dataset
        self.rows = StimulusRows(data, DELAYS[data.dataset], cache=folder / "stimulus-rows-projections.npz",
                                 features="projections")
        cache = folder / "state" / "gain-reference.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as z:
                groups = [str(g) for g in z["groups"]]
                self.weights = {g: (z[f"mx_{i}"], z[f"my_{i}"], z[f"W_{i}"]) for i, g in enumerate(groups)}
                names = [str(n) for n in z["set_names"]]
                self.sets = {n: z[f"set_{i}"] for i, n in enumerate(names)}
            self.floor = self._floor()
            return
        ridge = StimulusRidge(data, rows=self.rows, log=log)
        self.weights = {g: tuple(np.asarray(a, dtype=np.float32) for a in w) for g, w in ridge.weights.items()}
        self.sets = self._parcel_sets()
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"groups": np.array(list(self.weights)), "set_names": np.array(list(self.sets))}
        for i, (mx, my, W) in enumerate(self.weights.values()):
            arrays.update({f"mx_{i}": mx, f"my_{i}": my, f"W_{i}": W})
        arrays.update({f"set_{i}": p for i, p in enumerate(self.sets.values())})
        partial = cache.with_suffix(".partial.npz")  # other tasks may be waiting for the cache: write, then rename
        np.savez(partial, **arrays)
        partial.replace(cache)
        self.floor = self._floor()

    def predict(self, subject, timeline_id, frames, source=None):
        group = "all" if self.data.dataset == "hcp_movies" else subject
        mx, my, W = self.weights[group]
        return ((self.rows.lagged(timeline_id, np.asarray(frames), source) - mx) @ W + my).astype(np.float32)

    def _floor(self):
        """Per set, FLOOR x the median detrended reference energy per cell over training windows: windows without
        a stimulus (a run's lead-in, missing features) give a flat reference and no gain."""
        W = frames_within(WINDOW_SECONDS, self.data.grid.tr)
        t = np.linspace(-1.0, 1.0, W)
        X = np.stack([np.ones(W), t], 1)
        energy = []
        for run in self.data.runs("train")[::20]:
            s = self.predict(run.subject_id, run.timeline_id, np.arange(run.n)).astype(np.float64)
            for start in range(0, run.n - W + 1, W):
                if (run.support & run.allowed)[start:start + W].all():
                    w = s[start:start + W]
                    w = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]
                    energy.append([(w[:, p] ** 2).mean() for p in self.sets.values()])
        return FLOOR * np.median(np.array(energy), 0)

    def _parcel_sets(self):
        """Stimulus-driven parcels (top quarter of the reference's fit on training runs) and, per Yeo network,
        that network's parcels above the median."""
        r = []
        for run in self.data.runs("train")[::10]:
            f = np.flatnonzero(run.brain_ok & run.support & run.allowed)  # training frames only
            y, s = np.asarray(run.brain[f], dtype=np.float64), self.predict(run.subject_id, run.timeline_id, f)
            yc, sc = y - y.mean(0), s - s.mean(0)
            r.append((yc * sc).sum(0) / np.sqrt((yc ** 2).sum(0) * (sc ** 2).sum(0) + 1e-12))
            run.__dict__.pop("brain", None)
        r = np.nanmean(r, 0)
        sets = {"stimulus-driven": np.flatnonzero(r >= np.quantile(r, 0.75))}
        if NETWORKS_TSV.is_file():
            names = np.array([line.split("\t")[1].split("_")[2] for line in NETWORKS_TSV.read_text().splitlines()[1:]])
            for name in dict.fromkeys(names):
                parcels = np.flatnonzero((names == name) & (r > np.median(r)))
                if len(parcels) >= MIN_PARCELS:
                    sets[name] = parcels
        return sets


def window_gains(series, reference, ok, sets, floor=None):
    """[B, len(sets)] gains of series [B, T, 1000] on reference [B, T, 1000] over frames ok [B, T]; NaN where fewer
    than MIN_VALID of the window's frames are valid, or where the detrended reference's energy per cell is below
    floor[set]. The reference is detrended (mean and slope) per parcel."""
    B, T = ok.shape
    out = np.full((B, len(sets)), np.nan)
    t = np.linspace(-1.0, 1.0, T)
    for b in range(B):
        f = np.flatnonzero(ok[b])
        if len(f) < MIN_VALID * T:
            continue
        X = np.stack([np.ones(len(f)), t[f]], 1)
        s = reference[b, f].astype(np.float64)
        s = s - X @ np.linalg.lstsq(X, s, rcond=None)[0]
        y = series[b, f].astype(np.float64)
        for j, parcels in enumerate(sets):
            den = (s[:, parcels] ** 2).sum()
            least = 0.0 if floor is None else floor[j] * len(f) * len(parcels)
            out[b, j] = (y[:, parcels] * s[:, parcels]).sum() / den if den > max(least, 0.0) else np.nan
    return out
