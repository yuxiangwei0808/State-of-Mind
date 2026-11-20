"""neuroworld/paper/ablations.py (paper, Sec. 5.4 and its supplement): the ablation table, four small tables in one table*,
and the same rows with Forecast split by seconds ahead (supplement).

    python -m neuroworld.paper.ablations      prints both tables' LaTeX (stdout) and the evaluation sets of each row (stderr),
                                              and writes paper-analysis/results-ablations.json

Each row is one change from Spectator, all with 180 s of fMRI, paired against its twin on every Algonauts evaluation set the two
share (val = Friends S6, dev and test = the four Movie10 movies; State on the Movie10 repeats). Twins: the rows retrained through
the release recipe (random init, state size, fMRI window of 3 frames, no EMA, mean slow weights, the modality rows;
checkpoints/algonauts/ablations_20261113, scores in benchmark-ablations) against the control run there, which is Spectator
unchanged through the same scripts; the Transformer counterpart against Spectator; the other rows against the previous final
fit of Spectator (D180-win5), unmarked in the table. Of the fMRI windows the table keeps only 4.5 s.
Main table (the columns of the main table): Forecast over the minute, State, and Updating with 4 new fMRI frames, as paired
differences in points, and the paired difference in Pearson r of the forecast (Table 2's r: per recording, mean over the 40
frames ahead of the parcel-averaged correlation, averaged over recordings; as neuroworld/paper/history_table.py). Supplement:
Forecast over the minute, its first 7.5 s and its last 30 s. Labels use the symbols and terms of Sec. 3 (fast state f_t, slow
state u_t, windows in seconds). Run folders are in docs/reproduce.md.
"""
import json
import sys

import numpy as np

from neuroworld.benchmark import forecast as F, profile as P, report as R, updating as U
from neuroworld.benchmark.grid import frames_within
from neuroworld.paper.analysis import A, ROOTS as _ROOTS, TR
from neuroworld.paths import OUTPUT

ROOTS = [*_ROOTS, OUTPUT / "benchmark-ablations"]

SPLITS = ("val", "dev", "test")
BANDS = {"0-7.5 s": (0, frames_within(7.5, TR)), "30-60 s": (frames_within(30.0, TR), frames_within(60.0, TR))}
TWIN = "step2 D180-win5 q180 (latent)"
SPECTATOR = "ridge-jepa-180s-std-partial (latent)"
NEW = "ridge-jepa-180s-std-partial-{} (latent)"  # the November 13 runs and the reviewer controls (ablations_20261113)
CONTROL = NEW.format("control")  # Spectator unchanged through the same script (run_variant4.sh): twin of the run_variant3/4 rows
COUNTERPART = "transformer-tfforecast-linpath-jepa-180s-2x1024 (latent)"

# (table title, [(row label, benchmark row, twin row)]): each row removes or replaces one part, and the default is in its label.
# Rows run on November 13 (random init., fMRI window, state size; run_variant3.sh) are paired with the control run, the reviewer
# controls (run_variant4.sh: no EMA and stop-gradient, mean slow weights) and the Transformer counterpart with Spectator, the rest
# with the previous final fit (TWIN).
TABLES = [
    ("(a) Training", [("w/o pretraining (random init.)", NEW.format("randinit"), CONTROL),
                      ("Frozen pretrained encoder", "ridge-frozen-180s (latent)", TWIN),
                      ("w/o EMA and stop-gradient", NEW.format("online180"), SPECTATOR)]),
    ("(b) Architecture", [(r"w/o slow state $u_t$", "ridge-jepa-180s-noslow (latent)", TWIN),
                          (r"Slow weights: learned $\to$ mean", NEW.format("meanslow"), SPECTATOR),
                          (r"Transformer encoder $g_\phi$", "ridge-tfforecast-jepa-180s (latent)", TWIN),
                          (r"Transformer predictor $p_\theta$", "transformer-jepa-180s-2x1024 (latent)", TWIN),
                          (r"Transformer $g_\phi$ and $p_\theta$", COUNTERPART, SPECTATOR)]),
    ("(c) Inputs", [(r"fMRI window: 7.5 $\to$ 4.5\,s", NEW.format("win3"), CONTROL),
                           (r"Movie window: 7.5 $\to$ 1.5\,s", "ridge-jepa-180s-m1 (latent)", TWIN),
                           (r"Movie window: 7.5 $\to$ 15\,s", "ridge-jepa-180s-m10 (latent)", TWIN),
                           ("w/o video features", NEW.format("novideo"), SPECTATOR),
                           ("w/o text features", NEW.format("notext"), SPECTATOR),
                           ("w/o audio features", NEW.format("noaudio"), SPECTATOR)]),
    ("(d) Hyperparameters", [(r"State size: 2,048 $\to$ 1,024", NEW.format("dim512"), CONTROL),
                             (r"State size: 2,048 $\to$ 4,096", NEW.format("dim2048"), CONTROL),
                             (r"Rollout $K$: 40 $\to$ 80", "ridge-jepa-180s-roll80 (latent)", TWIN)]),
]


def forecast(name, splits, band=None):
    """Per-viewer (err, ref) sums of Forecast over the given splits, or None if a split is missing."""
    out = {}
    for split in splits:
        path = next((r / "algonauts" / split / name for r in ROOTS if (r / "algonauts" / split / name / "results.json").is_file()), None)
        if path is None:
            return None
        for row in F.load(path)["forecast"]:
            s = row["scores"]["P30"]
            err, ref = np.asarray(s["err"], float), np.asarray(s["ref"], float)
            if band is not None:
                err, ref = err[slice(*band)], ref[slice(*band)]
            a, b = out.get(row["unit"], (0.0, 0.0))
            out[row["unit"]] = (a + np.nansum(err), b + np.nansum(ref))
    return out


def updating(name, splits):
    """Per-viewer (err, ref) sums of Updating with the headline number of new fMRI frames."""
    out = {}
    for split in splits:
        sums = None
        for r in ROOTS:
            f = r / "algonauts" / "updating" / f"{name}.npz"
            if f.is_file():
                loaded = U.load(f.parent, name)
                if split in loaded:
                    sums = loaded[split]
                    break
        if sums is None:
            return None
        for unit, v in sums.get(P.HEADLINE_BUDGET, {}).items():
            a, b = out.get(unit, (0.0, 0.0))
            out[unit] = (a + float(v["se"]), b + float(v["ref"]))
    return out


def pearson(name, splits):
    """{content unit: (sum over its recordings of the forecast's Pearson r over the 40 frames ahead, recordings)}."""
    out = {}
    for split in splits:
        path = next((r / "algonauts" / split / name for r in ROOTS if (r / "algonauts" / split / name / "results.json").is_file()), None)
        if path is None:
            return None
        for row in F.load(path)["forecast"]:
            r = np.nanmean(np.asarray(row["scores"]["P30"]["pearson"], float)[:40])
            if np.isfinite(r):
                a, b = out.get(row["unit"], (0.0, 0))
                out[row["unit"]] = (a + r, b + 1)
    return out


def diff_r(a, b):
    """Mean r of a minus mean r of b, with its paired 95% interval over their common content units."""
    if not a or not b:
        return None
    common = sorted(set(a) & set(b))
    units = R.Units(common)
    (ea, sa), (eb, sb) = (units.ratio(np.array([x[u][0] for u in common], float), np.array([x[u][1] for u in common], float))
                          for x in (a, b))
    d = R.Units.interval(ea - eb, sa - sb)
    return {"value": d["value"], "low": d.get("low"), "high": d.get("high")}


def state(name):
    """Per-viewer sums of State on the Movie10 repeats."""
    for r in ROOTS:
        if (r / "algonauts" / "state" / f"{name}.npz").is_file():
            P.ROOT = r
            u = P._unit_sums("state", "algonauts", ["dev"], name)
            if u:
                return u
    return None


def diff(a, b):
    """Paired difference a - b in points, with its 95% interval."""
    if not a or not b:
        return None
    d = P._paired(a, b)
    return {"value": d["value"], "low": d.get("low"), "high": d.get("high")}


def row_cells(row, twin):
    have = [s for s in SPLITS if forecast(row, [s]) is not None and forecast(twin, [s]) is not None]
    cells = {"Forecast": diff(forecast(row, have), forecast(twin, have))}
    for key, band in BANDS.items():
        cells[f"Forecast {key}"] = diff(forecast(row, have, band), forecast(twin, have, band))
    cells["State"] = diff(state(row), state(twin))
    cells["Updating"] = diff(updating(row, have), updating(twin, have))
    cells["r"] = diff_r(pearson(row, have), pearson(twin, have))
    return have, cells


def num(d, digits=1):
    if d is None:
        return "n/a"
    s = f"{d['value']:+.{digits}f}"
    return s[1:] if float(s) == 0 else "$" + s + "$"


MAIN = ("Forecast", "State", "Updating", "r")
BY_HORIZON = ("Forecast", "Forecast 0-7.5 s", "Forecast 30-60 s")


def tabular(title, rows):
    out = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}", r"\toprule",
           rf"\textbf{{{title}}} & Forecast & State & Updating & $r$ \\", r"\midrule"]
    for label, cells in rows:
        out.append(label + " & " + " & ".join(num(cells[k], 3 if k == "r" else 1) for k in MAIN) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular*}"])


def latex(blocks):
    col = lambda top, bottom: (r"\begin{minipage}[t]{0.49\linewidth}" + "\n" + top + "\n\n" + r"\vspace{8pt}" + "\n" + bottom
                               + "\n" + r"\end{minipage}")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        "% Paired differences on all evaluation data (Friends S6 and the four movies). Twins: the November 13 runs (random init.,",
        "% fMRI window, state size) against the control run (Spectator unchanged through the same script); the reviewer controls and",
        "% the Transformer counterpart against Spectator; the other rows against the previous final fit (D180-win5).",
        "% From neuroworld/paper/ablations.py; run folders in docs/reproduce.md.",
        col(blocks[0], blocks[2]) + r"\hfill", col(blocks[1], blocks[3]),
        r"\caption{\textbf{Ablations.} Change in each metric of \cref{tab:main} when one part of \method is removed or replaced, "
        r"in points for Forecast, State and Updating and as a difference for $r$. Negative values are below \method. "
        r"The four panels ablate the training procedure~(a), the architecture~(b), the input windows~(c) and two hyperparameters~(d), "
        r"and \cref{sec:ablations} describes each change. The value before each arrow is the one \method uses.}",
        r"\label{tab:ablations}", r"\end{table*}"])


def latex_by_horizon(groups):
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           "% The rows of tab:ablations, Forecast only, over the minute and two parts of it. From neuroworld/paper/ablations.py.",
           r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{Forecast, seconds ahead} \\", r"\cmidrule(l){2-4}",
           r" & 0--60 & 0--7.5 & 30--60 \\", r"\midrule"]
    for k, (title, rows) in enumerate(groups):
        if k:
            out.append(r"\addlinespace")
        out.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        for label, cells in rows:
            out.append(label + " & " + " & ".join(num(cells[c]) for c in BY_HORIZON) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}",
                            r"\caption{\textbf{Ablations by seconds ahead.} Change in Forecast points for the rows of \cref{tab:ablations}, "
                            r"over the whole minute (as in \cref{tab:ablations}), its first 7.5\,s and its last 30\,s.}",
                            r"\label{tab:app-ablations}", r"\end{table}"])


if __name__ == "__main__":
    blocks, groups, results = [], [], {}
    for title, rows in TABLES:
        done = []
        for label, row, twin in rows:
            have, cells = row_cells(row, twin)
            done.append((label, cells))
            results[f"{title} | {label}"] = {"row": row, "twin": twin, "splits": have, **cells}
            print(f"{label}: {'+'.join(have)}", file=sys.stderr)
        blocks.append(tabular(title, done))
        groups.append((title, done))
    (A / "results-ablations.json").write_text(json.dumps(results, indent=1))
    print(latex(blocks))
    print()
    print("% ---- supplement (sec/8_appendix.tex) ----")
    print(latex_by_horizon(groups))
