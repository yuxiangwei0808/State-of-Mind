"""Nonanticipation check (docs/benchmark.md#nonanticipation-check): a forecast must not depend on input that comes
later than the forecast frame.

Two queries share the origin, the fMRI prefix and the movie up to frame t0 + k; from t0 + k on, the second takes
its movie from another moment. Forecasts for the frames before t0 + k must agree. The check splices precomputed
feature rows, so it tests the model's use of its inputs; lookahead inside the feature extractors themselves (for
example TRIBEv2's bidirectional context) cannot show up here and is flagged by the data notes instead.

    python -m neuroworld.benchmark nonanticipation --model run:<dir> --device cuda
"""
import json
from pathlib import Path

import numpy as np

from neuroworld.benchmark import queries as Q
from neuroworld.benchmark.forecast import predict_batch

DELAYS = (5, 10, 20)   # frames after the origin at which the two movies diverge
QUERIES = 256          # origins sampled from the split
TOLERANCE = 1e-4       # largest allowed difference before the divergence, relative to the forecast's SD


def check(data, model, split="val", seed=0):
    rng = np.random.default_rng(seed)
    grid = data.grid
    prefix = grid.history_frames()
    runs = [r for r in data.runs(split) if len(Q.forecast_origins(r, grid))]
    out = {"model": model.name, "split": split, "delays": {}}
    for k in DELAYS:
        horizon = k + 5
        worst, total = 0.0, 0
        for _ in range(QUERIES // 32):
            run, other = (runs[i] for i in rng.choice(len(runs), 2, replace=False))
            origins = rng.choice(Q.forecast_origins(run, grid), min(32, len(Q.forecast_origins(run, grid))), replace=False)
            theirs = rng.choice(Q.forecast_origins(other, grid), len(origins))
            same = Q.batch(data.dataset, grid.tr, run, origins, prefix, horizon)
            spliced = Q.batch(data.dataset, grid.tr, run, origins, prefix, horizon,
                              splices=[(other.timeline_id, int(t), k) for t in theirs])
            a, b = predict_batch(model, same), predict_batch(model, spliced)
            scale = a.std() + 1e-8
            worst = max(worst, float(np.abs(a[:, :k] - b[:, :k]).max() / scale))
            total += len(origins)
            for r in (run, other):
                r.__dict__.pop("brain", None)
        out["delays"][str(k)] = {"max_relative_difference": worst, "queries": total}
    out["pass"] = all(v["max_relative_difference"] <= TOLERANCE for v in out["delays"].values())
    return out


def run(data, model, output="artifacts/benchmark", split="val", log=print):
    from neuroworld.benchmark.adapt import calibrate
    from neuroworld.benchmark.grid import stamp
    calibrate(model, data, split, log=log)
    result = {**check(data, model, split), "stamp": stamp(model)}
    folder = Path(output) / data.dataset / "nonanticipation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{model.name}.json").write_text(json.dumps(result, indent=1) + "\n")
    if log:
        log(f"nonanticipation {model.name}: {'pass' if result['pass'] else 'FAIL'} "
            + ", ".join(f"k={k}: {v['max_relative_difference']:.2e}" for k, v in result["delays"].items()))
    return result
