"""Queries: forecast origins, query batches and measured targets.

Origins depend only on the bundle's masks, split definitions and the grid, never on responses.
"""
import numpy as np

from state_of_mind.benchmark.grid import LONGEST_PREFIX
from state_of_mind.benchmark.interface import QueryBatch


def forecast_origins(run, grid):
    """Origins every stride with the last 30 s complete and the next minute of targets inside one supported
    segment of the split."""
    longest, horizon = grid.prefixes()[LONGEST_PREFIX], grid.horizon_steps
    return np.array([t0 for t0 in range(longest, run.n - horizon + 1, grid.stride_trs)
                     if run.prefix_complete(t0, longest) and run.target_span_ok(t0, horizon)], dtype=np.int64)


def batch(dataset, tr, run, origins, prefix_frames, horizon, splices=None, histories=None):
    """QueryBatch for origins of one run; splices give each query's (timeline, frame) supplying the stimulus from
    its origin on, histories its replaced stimulus history per modality (interface.QueryBatch)."""
    origins = np.asarray(origins, dtype=np.int64)
    prefix = np.full((len(origins), prefix_frames, 1000), np.nan, dtype=np.float32)
    valid = np.zeros((len(origins), prefix_frames), dtype=bool)
    for i, t0 in enumerate(origins):
        if prefix_frames:
            frames = np.arange(t0 - prefix_frames, t0)
            ok = np.zeros(prefix_frames, dtype=bool)  # frames before the run start are invalid
            inside = frames >= 0
            ok[inside] = run.brain_ok[frames[inside]]
            if inside.any():
                first = int(frames[inside][0])
                block = np.asarray(run.brain[first:t0], dtype=np.float32)
                prefix[i][inside] = np.where(ok[inside][:, None], block, np.nan)
            valid[i] = ok
    return QueryBatch(dataset=dataset, tr=tr, run_ids=[run.query_id] * len(origins),
                      subject_ids=[run.subject_id] * len(origins), timeline_ids=[run.stimulus_id] * len(origins),
                      origins=origins, prefix=prefix, prefix_valid=valid, horizon=int(horizon),
                      splices=None if splices is None else list(splices),
                      histories=None if histories is None else list(histories))


def joint_batch(dataset, tr, episodes, prefix_frames, horizon):
    """One QueryBatch for the origins of several runs, episodes = [(run, origins)], in that order."""
    parts = [batch(dataset, tr, run, origins, prefix_frames, horizon) for run, origins in episodes]
    return QueryBatch(dataset=dataset, tr=tr, run_ids=sum((p.run_ids for p in parts), []),
                      subject_ids=sum((p.subject_ids for p in parts), []),
                      timeline_ids=sum((p.timeline_ids for p in parts), []),
                      origins=np.concatenate([p.origins for p in parts]),
                      prefix=np.concatenate([p.prefix for p in parts]),
                      prefix_valid=np.concatenate([p.prefix_valid for p in parts]), horizon=int(horizon))


def episodes(runs, origins):
    """[(run, origins(run))] for the runs with at least one origin."""
    return [(run, o) for run, o in ((run, origins(run)) for run in runs) if len(o)]


def spans(episodes):
    """(run, rows) of each run's queries in a joint batch of episodes = [(run, origins)]."""
    end = 0
    for run, origins in episodes:
        end += len(origins)
        yield run, slice(end - len(origins), end)


def joint_targets(episodes, horizon):
    """targets() of episodes = [(run, origins)] of several runs, concatenated in that order."""
    y, ok = zip(*(targets(run, origins, horizon) for run, origins in episodes))
    return np.concatenate(y), np.concatenate(ok)


def targets(run, origins, horizon, stop=None):
    """([n, horizon, 1000] measured frames, [n, horizon] scoreable mask). Frames past the run, outside the
    split's supported targets or (with `stop`) at or after a segment end are not scoreable."""
    origins = np.asarray(origins, dtype=np.int64)
    frames = origins[:, None] + np.arange(horizon)[None]
    inside = frames < (run.n if stop is None else np.minimum(run.n, np.asarray(stop)[..., None]))
    clipped = np.minimum(frames, run.n - 1)
    ok = inside & run.brain_ok[clipped] & run.target_ok[clipped]
    return np.asarray(run.brain[clipped.reshape(-1)], dtype=np.float32).reshape(*frames.shape, run.brain.shape[1]), ok


def run_variance(run):
    """[1000] the run's measured variance over its valid, stimulus-supported frames, averaged over parcels (the same
    value for every parcel): the nMSE normalizer of every task. 1 - nMSE is then the share of the run's activity,
    summed over parcels, that a forecast explains; a parcel without signal in some run cannot dominate it."""
    ok = run.brain_ok & run.support
    variance = np.asarray(run.brain[np.flatnonzero(ok)], dtype=np.float64).var(0).mean() + 1e-8
    return np.full(run.brain.shape[1], variance)


def run_mean(run):
    """[1000] the run's mean per parcel over its valid, stimulus-supported frames: the reference predictor of every
    "% of measured BOLD variance predicted" score (skill 0 = predicting the run's average at every scored frame)."""
    ok = run.brain_ok & run.support
    return np.asarray(run.brain[np.flatnonzero(ok)], dtype=np.float64).mean(0)


def band_slices(grid, horizon):
    """{band: 0-based step indices} available within `horizon` steps (bands with >= 3 steps)."""
    return {band: np.arange(steps.start - 1, steps.stop - 1) for band, steps in grid.bands(horizon).items()}


__all__ = ["forecast_origins", "batch", "joint_batch", "episodes", "spans", "targets", "joint_targets",
           "run_variance", "run_mean", "band_slices"]
