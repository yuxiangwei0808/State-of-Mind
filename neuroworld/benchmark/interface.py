"""What a submission sees and returns (docs/benchmark.md#how-a-model-is-queried).

A model receives a QueryBatch: for each query a run, subject, origin frame t0 (the first target, step 1),
the observed prefix of brain frames before t0, and a horizon. It returns predicted Schaefer-1000 fMRI for
frames t0 .. t0+horizon-1. The stimulus is public: a model reads it for the query's timeline itself, up to
frame t0+horizon-1 (a causal model, for each target only up to that target). No brain frame at or after
t0 is ever passed in.

A counterfactual query carries a splice: from stimulus frame t0 on (or t0 + k), the stimulus comes from another
timeline, starting at its frame s (stimulus_source maps each frame). A query may also replace the stimulus history before
t0, per modality: histories[i] = {modality: (timeline, frame)} takes that modality's frames before t0 from the
other timeline, aligned so that t0 maps to its frame. The key "all" serves models whose rows fuse modalities.
"""
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np


@dataclass
class QueryBatch:
    dataset: str
    tr: float
    run_ids: list
    subject_ids: list
    timeline_ids: list
    origins: np.ndarray       # [B] frame index of step 1
    prefix: np.ndarray        # [B, P, 1000] float32; frames t0-P .. t0-1; NaN where invalid or before the run
    prefix_valid: np.ndarray  # [B, P] bool
    horizon: int
    splices: list = None      # per query: None, or (timeline, frame[, k]) supplying the stimulus from t0 (+ k) on
    histories: list = None    # per query: None, or {modality or "all": (timeline, frame)} replacing frames before t0

    def __len__(self):
        return len(self.origins)

    def splice(self, i):
        return None if self.splices is None else self.splices[i]

    def history(self, i):
        return None if self.histories is None else self.histories[i]

    def remapped(self, i):
        return self.splice(i) is not None or bool(self.history(i))

    def take(self, rows):
        """The queries at `rows` (a slice or a list of indices)."""
        def pick(values):
            if values is None:
                return None
            return values[rows] if isinstance(rows, slice) else [values[i] for i in rows]
        return replace(self, run_ids=pick(self.run_ids), subject_ids=pick(self.subject_ids),
                       timeline_ids=pick(self.timeline_ids), origins=self.origins[rows], prefix=self.prefix[rows],
                       prefix_valid=self.prefix_valid[rows], splices=pick(self.splices), histories=pick(self.histories))


def stimulus_source(batch, i, frames, modality=None):
    """(timelines, frames) supplying query i's stimulus (one modality's, or all fused rows when modality is
    None) at the given stimulus frames: spliced from t0 on, with any replaced history before t0."""
    frames = np.asarray(frames)
    timelines = np.full(len(frames), batch.timeline_ids[i], dtype=object)
    mapped = frames.copy()
    after = frames >= batch.origins[i]
    splice = batch.splice(i)
    if splice is not None:  # (timeline, frame) from t0 on, or (timeline, frame, k) from t0 + k on
        start = batch.origins[i] + (splice[2] if len(splice) > 2 else 0)
        switched = frames >= start
        timelines[switched] = splice[0]
        mapped[switched] = frames[switched] - start + splice[1]
    history = (batch.history(i) or {}).get("all" if modality is None else modality)
    if history is not None:
        timelines[~after] = history[0]
        mapped[~after] = frames[~after] - batch.origins[i] + history[1]
    return timelines, mapped


class Model(Protocol):
    name: str

    def predict(self, batch: QueryBatch) -> np.ndarray:
        """[B, horizon, 1000] predicted fMRI."""


def last_valid(batch):
    """[B, 1000] most recent valid prefix frame, or zeros (the z-scored mean) with no valid frame."""
    out = np.zeros((len(batch), batch.prefix.shape[2] if batch.prefix.ndim == 3 else 1000), dtype=np.float32)
    if batch.prefix.shape[1] == 0:
        return out
    valid = batch.prefix_valid
    has = valid.any(1)
    index = batch.prefix.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    out[has] = batch.prefix[np.flatnonzero(has), index[has]]
    return out


class Persistence:
    """Repeat the last observed frame at every step (the run mean, zero, without a prefix)."""
    name = "persistence"

    def predict(self, batch):
        return np.repeat(last_valid(batch)[:, None], batch.horizon, axis=1)
