"""Scores: per-step forecasting and pooled correlations.

Per-step Pearson is the correlation across origins within a run, for each parcel, averaged over parcels;
runs are averaged later. The drift-robust version first removes a DCT high-pass basis (cutoff 1/128 Hz,
over the span of the origins' target times) from both the predicted and the measured series.
"""
import numpy as np

HIGH_PASS_SECONDS = 128.0
MIN_ROWS = 10  # origins needed for a per-run, per-step correlation


def _parcel_pearson(x, y):
    """Mean over parcels of the Pearson of x and y along axis 0, for every remaining leading index:
    x, y [rows, ..., parcels] -> [...]; NaN where no parcel is defined."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    sxx, syy = (x * x).sum(0), (y * y).sum(0)
    ok = (sxx > 1e-12) & (syy > 1e-12)
    r = np.where(ok, (x * y).sum(0) / np.sqrt(np.where(ok, sxx * syy, 1.0)), 0.0)
    count = ok.sum(-1)
    return np.where(count > 0, r.sum(-1) / np.maximum(count, 1), np.nan)


def high_pass_basis(times, tr, cutoff=HIGH_PASS_SECONDS):
    """Constant plus DCT regressors slower than 1/cutoff Hz over the span of `times` (frame indices)."""
    times = np.asarray(times, dtype=np.float64)
    span = times.max() - times.min() + 1
    count = int(np.floor(2 * span * tr / cutoff)) + 1
    phase = (times - times.min() + 0.5) / span
    return np.column_stack([np.ones_like(times)] + [np.cos(np.pi * j * phase) for j in range(1, count)])


def _residual(basis, *arrays):
    q, _ = np.linalg.qr(basis)
    return [a - np.tensordot(q, np.tensordot(q.T, a, axes=1), axes=1) for a in arrays]


def step_scores(predicted, observed, ok, origins, tr, min_rows=MIN_ROWS, mean=None):
    """Per-step scores for one run: {'pearson', 'pearson_hp', 'nmse', 'err', 'ref', 'rows'} arrays of length horizon.

    predicted, observed: [n, H, P]; ok: [n, H] scoreable targets; origins: [n] frame of step 1. Steps with the
    same scoreable origins share one high-pass basis (times shift by a constant), so they are scored together.
    nmse divides by the measured variance across these origins. err and ref (with mean, the run's mean per parcel,
    queries.run_mean) are the summed squared errors of the forecast and of predicting the run's average on the same
    targets: pooled over runs, 1 - err / ref is the card's "% of measured BOLD variance predicted". They cover every
    scoreable target; min_rows applies only to the correlations and nmse, which need several origins."""
    n, horizon, _ = predicted.shape
    out = {key: np.full(horizon, np.nan) for key in ("pearson", "pearson_hp", "nmse", "err", "ref")}
    out["rows"] = ok.sum(0).astype(np.int64)
    if mean is not None:  # squared errors need no minimum number of origins
        out["err"] = np.where(ok, ((predicted.astype(np.float64) - observed) ** 2).sum(-1), 0.0).sum(0)
        out["ref"] = np.where(ok, ((observed.astype(np.float64) - mean) ** 2).sum(-1), 0.0).sum(0)
    groups = {}
    for k in range(horizon):
        if out["rows"][k] >= min_rows:
            groups.setdefault(ok[:, k].tobytes(), []).append(k)
    for key, steps in groups.items():
        rows = np.flatnonzero(np.frombuffer(key, dtype=bool))
        steps = np.asarray(steps)
        x = predicted[rows][:, steps].astype(np.float64)
        y = observed[rows][:, steps].astype(np.float64)
        out["pearson"][steps] = _parcel_pearson(x, y)
        variance = ((y - y.mean(0)) ** 2).sum(axis=(0, 2))
        error = ((x - y) ** 2).sum(axis=(0, 2))
        out["nmse"][steps] = np.where(variance > 0, error / np.where(variance > 0, variance, 1.0), np.nan)
        basis = high_pass_basis(origins[rows], tr)
        if len(rows) >= basis.shape[1] + min_rows:
            xr, yr = _residual(basis, x, y)
            out["pearson_hp"][steps] = _parcel_pearson(xr, yr)
    return out


def pooled_correlation(sums):
    """Mean over parcels of the Pearson from summed [n, Sx, Sy, Sxy, Sxx, Syy] rows [6, P]."""
    n, sx, sy, sxy, sxx, syy = sums
    cov = sxy - sx * sy / np.maximum(n, 1)
    vx, vy = sxx - sx ** 2 / np.maximum(n, 1), syy - sy ** 2 / np.maximum(n, 1)
    ok = (vx > 1e-12) & (vy > 1e-12)
    return float((cov[ok] / np.sqrt(vx[ok] * vy[ok])).mean()) if ok.any() else np.nan
