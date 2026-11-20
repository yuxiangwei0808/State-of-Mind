"""Bootstrap units, forecast summaries, paired model differences and the forecast scorecard.

Every statistic is a ratio of per-unit sums (a mean of run values, a skill), so a bootstrap draw reweights units
by their resampling counts; a paired difference uses the same draws for both models on their common units.
Units are content parts (Algonauts) or participants (HCP), never origins.
"""
import numpy as np

from neuroworld.benchmark.grid import GRIDS, LONGEST_PREFIX, PREFIXES

DRAWS = 2000
METRICS = ("pearson_hp", "pearson", "nmse")  # per-run means; "nmse_pooled" pools err / ref over frames


def _band_steps(dataset, horizon):
    """{band: 0-based steps}: the detail bands, and "minute", every step of the next minute (the headline)."""
    out = {band: np.arange(s.start - 1, s.stop - 1) for band, s in GRIDS[dataset].bands(horizon).items()}
    out["minute"] = np.arange(min(horizon, GRIDS[dataset].horizon_steps))
    return out


class Units:
    """Bootstrap resampling counts [draws, units] over a fixed, sorted unit list."""

    def __init__(self, units, draws=DRAWS, seed=0):
        self.units = sorted(set(units))
        self.index = {u: i for i, u in enumerate(self.units)}
        rng = np.random.default_rng(seed)
        n = len(self.units)
        picks = rng.integers(n, size=(draws, n))
        self.counts = np.stack([np.bincount(row, minlength=n) for row in picks]).astype(np.float64)

    def collect(self, pairs):
        """Sum (unit, numerator, denominator) triples into per-unit arrays (missing units contribute zero)."""
        num, den = np.zeros(len(self.units)), np.zeros(len(self.units))
        for unit, a, b in pairs:
            if unit in self.index and np.isfinite(a) and np.isfinite(b):
                num[self.index[unit]] += a
                den[self.index[unit]] += b
        return num, den

    def ratio(self, num, den):
        """(estimate, low, high, samples) of sum(num) / sum(den)."""
        estimate = num.sum() / den.sum() if den.sum() > 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = (self.counts @ num) / (self.counts @ den)
        return estimate, samples

    @staticmethod
    def interval(estimate, samples):
        finite = samples[np.isfinite(samples)]
        if not np.isfinite(estimate) or len(finite) < 0.9 * len(samples):
            return {"value": None if not np.isfinite(estimate) else float(estimate), "low": None, "high": None}
        return {"value": float(estimate), "low": float(np.quantile(finite, 0.025)),
                "high": float(np.quantile(finite, 0.975))}


def _run_band(row, prefix, metric, steps):
    if metric not in row["scores"][prefix]:  # results scored before the metric existed
        return np.nan
    values = np.asarray(row["scores"][prefix][metric], dtype=np.float64)[steps]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def cells(results):
    """{(prefix, band, metric): [(unit, numerator, denominator)]}: each run's mean over the band's steps, and
    "nmse_pooled", the summed squared error over the summed run-mean reference error (every scored frame weighs the
    same, in every task)."""
    steps = _band_steps(results["dataset"], results["grid"]["horizon_steps"])
    out = {}
    for row in results["forecast"]:
        for prefix in (p for p in results["grid"]["prefixes"] if p in PREFIXES):  # older results also hold P7.5
            for band, s in steps.items():
                for metric in METRICS:
                    out.setdefault((prefix, band, metric), []).append((row["unit"], _run_band(row, prefix, metric, s), 1.0))
                scores = row["scores"][prefix]
                if "err" in scores:
                    err, ref = (np.nansum(np.asarray(scores[k], dtype=np.float64)[s]) for k in ("err", "ref"))
                    out.setdefault((prefix, band, "nmse_pooled"), []).append((row["unit"], err, ref))
    return out


def _all_units(results):
    return {row["unit"] for row in results["forecast"]}


def summarize(results, units=None):
    """{cell: {"value", "low", "high", "units"}} with 95% bootstrap intervals."""
    units = units or Units(_all_units(results))
    summary = {}
    for cell, triples in cells(results).items():
        num, den = units.collect(triples)
        summary[cell] = {**Units.interval(*units.ratio(num, den)), "units": int((den > 0).sum())}
    return summary


def paired(results_a, results_b):
    """{cell: a - b with a 95% interval} over the units both models were scored on."""
    common = _all_units(results_a) & _all_units(results_b)
    units = Units(common)
    a, b = cells(results_a), cells(results_b)
    out = {}
    for cell in sorted(set(a) & set(b), key=str):
        na, da = units.collect(a[cell])
        nb, db = units.collect(b[cell])
        ea, sa = units.ratio(na, da)
        eb, sb = units.ratio(nb, db)
        out[cell] = Units.interval(ea - eb, sa - sb)
    return out


# Scorecard columns: drift-robust r and nMSE in B1 (0-7.5 s), B2 (7.5-30 s) and B3 (30-90 s), from the 30-s prefix.
HEADLINE = [(f"{band} {label}", (LONGEST_PREFIX, band, metric)) for band in ("B1", "B2", "B3")
            for label, metric in (("r", "pearson_hp"), ("nMSE", "nmse_pooled"))]


def _fmt(cell, digits=3):
    if cell is None or cell.get("value") is None:
        return "n/a"
    text = f"{cell['value']:.{digits}f}"
    if cell.get("low") is not None:
        text += f" [{cell['low']:.{digits}f}, {cell['high']:.{digits}f}]"
    return text


def scorecard(summaries, names=None):
    """Markdown table: one row per model, the headline cells as columns."""
    names = names or list(summaries)
    lines = ["| Model | " + " | ".join(label for label, _ in HEADLINE) + " |",
             "|---|" + "---|" * len(HEADLINE)]
    for name in names:
        s = summaries[name]
        lines.append(f"| {name} | " + " | ".join(_fmt(s.get(cell)) for _, cell in HEADLINE) + " |")
    return "\n".join(lines)


__all__ = ["Units", "cells", "summarize", "paired", "scorecard", "HEADLINE"]
