"""Forecast task (docs/benchmark.md#tasks): score one model's forecasts on one dataset split.

At every origin (each 3 s), the model gets the measured fMRI before it (up to two minutes, the last 30 s complete)
and the movie, and forecasts the next minute. Per run and step, the drift-robust Pearson r and nMSE against the
measured fMRI are kept with the run's bootstrap unit and exposure tag, so summaries and paired comparisons resample
the same units (benchmark/report.py). CPU models can score runs in forked worker processes (``workers``); GPU models use one.
"""
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from neuroworld.benchmark import queries as Q
from neuroworld.benchmark.metrics import step_scores


def predict(model, dataset, tr, run, origins, prefix_frames, horizon, batch_size=256, splices=None):
    out = []
    for start in range(0, len(origins), batch_size):
        chunk = None if splices is None else splices[start:start + batch_size]
        query = Q.batch(dataset, tr, run, origins[start:start + batch_size], prefix_frames, horizon, splices=chunk)
        value = np.asarray(model.predict(query), dtype=np.float32)
        if value.shape != (len(query), horizon, 1000) or not np.isfinite(value).all():
            raise ValueError(f"{model.name} returned {value.shape} or non-finite values for {run.run_id}")
        out.append(value)
    return np.concatenate(out) if out else np.zeros((0, horizon, 1000), dtype=np.float32)


def predict_batch(model, batch, size=128):
    """[B, horizon, 1000] forecasts for a prepared QueryBatch, in chunks of `size` queries."""
    out = []
    for a in range(0, len(batch), size):
        part = batch.take(slice(a, a + size))
        value = np.asarray(model.predict(part), dtype=np.float32)
        if value.shape != (len(part), batch.horizon, 1000) or not np.isfinite(value).all():
            raise ValueError(f"{model.name} returned {value.shape} or non-finite values")
        out.append(value)
    return np.concatenate(out) if out else np.zeros((0, batch.horizon, 1000), dtype=np.float32)


def _meta(data, run):
    return {"run_id": run.run_id, "subject": run.subject_id, "content": run.group, "unit": data.bootstrap_unit(run),
            "exposure": run.exposure}


def forecast_rows(data, models, run):
    """The Forecast rows of one run for several models on the same queries, one after the other: two readouts of one latent model
    (its probe and its own decoder) then share its rollouts through the adapter's memory."""
    grid = data.grid
    origins = Q.forecast_origins(run, grid)
    if not len(origins):
        return [None] * len(models)
    observed, ok = Q.targets(run, origins, grid.horizon_steps)
    mean = Q.run_mean(run)
    out = []
    for model in models:
        forecast = predict(model, data.dataset, grid.tr, run, origins, grid.history_frames(), grid.horizon_steps)
        scores = {name: step_scores(forecast, observed, ok, origins, grid.tr, mean=mean)
                  for name in grid.prefixes()}  # one entry: the last 30 s complete, up to two minutes supplied
        out.append({**_meta(data, run), "origins": int(len(origins)), "scores": scores})
    return out


_CONTEXT = {}  # the job a forked worker runs (set by _map before forking)


def _release(item):
    """Close a scored run's brain memory map (each open map holds a file descriptor)."""
    if hasattr(item, "__dict__"):
        item.__dict__.pop("brain", None)


def _call(index):
    function, items, args = _CONTEXT["job"]
    try:
        return function(*args, items[index])
    finally:
        _release(items[index])


def _map(function, items, args, workers, log=None, label=""):
    """[function(*args, item) for item in items], in forked workers when workers > 1."""
    if workers <= 1 or len(items) < 2:
        out = []
        for i, item in enumerate(items):
            try:
                out.append(function(*args, item))
            finally:
                _release(item)
            if log and (i + 1) % 50 == 0:
                log(f"{label} {i + 1}/{len(items)}")
        return out
    _CONTEXT["job"] = (function, items, args)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            return pool.map(_call, range(len(items)), chunksize=1)
    finally:
        _CONTEXT.pop("job", None)


def run(data, model, split, output=None, workers=1, log=print):
    """Score `model` (or a list of models, on the same queries) on `split`; returns the results (a list for a list)
    and writes them under output/<dataset>/<split>/<model>."""
    started = time.time()
    models = model if isinstance(model, list) else [model]
    runs = data.runs(split)
    from neuroworld.benchmark.grid import stamp
    rows = _map(forecast_rows, runs, (data, models), workers, log, "forecast")
    out = []
    for i, m in enumerate(models):
        results = {"model": m.name, "dataset": data.dataset, "split": split, "tr": data.grid.tr, "stamp": stamp(m),
                   "grid": {"prefixes": data.grid.prefixes(), "horizon_steps": data.grid.horizon_steps,
                            "stride_trs": data.grid.stride_trs},
                   "config": str(data.config_path), "forecast": [r[i] for r in rows if r[i]],
                   "seconds": time.time() - started}
        if output is not None:
            save(results, Path(output) / data.dataset / split / m.name)
        if log:
            log(f"{m.name} {data.dataset} {split}: {results['seconds']:.0f}s")
        out.append(results)
    return out if isinstance(model, list) else out[0]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def save(results, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(_jsonable(results)) + "\n")
    return folder


def load(folder):
    return json.loads((Path(folder) / "results.json").read_text())
