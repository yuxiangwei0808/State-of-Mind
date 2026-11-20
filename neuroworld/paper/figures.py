"""neuroworld/paper/figures.py: the paper's Sec. 5 figures from neuroworld/paper/analysis.py's results, in $NEUROWORLD_OUTPUT/paper-analysis/figs/paper/,
and the data helpers (TABLE2, LOOK, AT_180, with_external, with_ownfill) whose results Figures 5 and 6 snapshot (cvpr2027-paper/viz/history,
viz/properties: draw_figure5.py / draw_figure6.py --refresh).

    python -m neuroworld.paper.figures
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroworld.paper.analysis import A, TR, H, ROWS, forecast_steps, state_steps, updating_budgets, surface_values
from neuroworld.paths import EXTERNAL

OUT = A / "figs/paper"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
# Names the paper uses for the stored rows (Table 2 and the appendix figures).
SHOWN = {"Linear forecaster, 180 s": "Linear forecaster", "Kalman filter": "IPSID", "Prior latent world model": "NeuroWorld",
         "Stimulus only (TRIBE v2)": "TRIBE v2"}
STYLE = {"Spectator": dict(color="#C0392B", lw=2.0, zorder=5), "Transformer counterpart": dict(color="#2E86C1"),
         "Linear forecaster, 180 s": dict(color="#555555"),
         "Linear forecaster, 7.5 s": dict(color="#999999", ls="--"), "Kalman filter": dict(color="#B9770E"),
         "Prior latent world model": dict(color="#229954"), "Stimulus only (TRIBE v2)": dict(color="#AAB7B8", ls=":"),
         "No fMRI (movie only)": dict(color="#E67E22", ls="--"), "No movie (fMRI only)": dict(color="#16A085", ls="--")}
sec = (np.arange(H) + 1) * TR


def curve(ax, label, x, y):
    ax.plot(x, y, label=SHOWN.get(label, label), **STYLE.get(label, {}))


# The models of Table 2, in its order, and one look each, read by the appendix figures' scripts (Figure 5: cvpr2027-paper/viz/history,
# Figure 6: cvpr2027-paper/viz/properties), which draw them. Colour-blind-safe hues (Okabe-Ito / Paul Tol), a marker per model;
# the models that receive no fMRI (TRIBE v2, MIRAGE) are dashed. Palette reviewed with GPT Astra, Nov 14.
TABLE2 = ["Linear forecaster", "IPSID", "NeuroWorld", "BrainVista", "TRIBE v2", "MIRAGE", "Spectator"]
LOOK = {"Linear forecaster": dict(color="#3B3B3B", marker="s"),
        "IPSID": dict(color="#4477AA", marker="v"),
        "NeuroWorld": dict(color="#009E73", marker="P"),
        "BrainVista": dict(color="#AA3377", marker="D", ms=2.5),
        "TRIBE v2": dict(color="#56B4E9", marker=None, ls=(0, (4, 3))),
        "MIRAGE": dict(color="#C99A2E", marker=None, ls=(3.5, (4, 3))),
        "Spectator": dict(color="#C0392B", marker="o", lw=1.7, ms=3.4, alpha=1.0, zorder=5)}
NO_FMRI = ("TRIBE v2", "MIRAGE")
# Benchmark rows of the models at 180 s (Table 2); BrainVista comes from neuroworld/paper/brainvista_crossfit.py, MIRAGE from the external storess.
AT_180 = {"Linear forecaster": "linear [context] 180 s", "IPSID": "IPSID [context] 180 s", "NeuroWorld": "NeuroWorld k1 180s (latent)",
          "TRIBE v2": "TRIBEv2 zero-shot", "MIRAGE": "MIRAGE zero-shot", "Spectator": "ridge-jepa-180s-std-partial (latent)"}
EXTERNAL_STORES = [EXTERNAL / "baselines/benchmark", EXTERNAL / "baselines/test-split/benchmark"]
BRAINVISTA_STEPS = A / "results-brainvista-steps-180s.npz"


def with_external(fn, *args):
    """fn(*args) with the external stores after the paper's, for the rows only they hold (MIRAGE)."""
    from neuroworld.paper import analysis
    analysis.ROOTS.extend(EXTERNAL_STORES)
    try:
        return fn(*args)
    finally:
        del analysis.ROOTS[-len(EXTERNAL_STORES):]


with_zwu25 = with_external  # the name the paper repository's figure scripts call


# The linear forecaster's Updating with the frames it is not shown filled by its own forecast, as the latent models fill them
# (paper-analysis/linear-updating-ownfill, scored with `benchmark updating --own-fill`; the older stores fill them with the training
# mean). Same model and fits.
# budgets180: 180 s at every number of new frames (Figure 6); valdev, test: every length at 0, 4, 16 and gap (Figure 5, Table 2)
OWNFILL = [A / "linear-updating-ownfill/budgets180", A / "linear-updating-ownfill/valdev", A / "linear-updating-ownfill/test"]



def with_ownfill(fn, *args):
    """fn(*args) with the own-fill stores searched first."""
    from neuroworld.paper import analysis
    analysis.ROOTS[:0] = OWNFILL
    try:
        return fn(*args)
    finally:
        del analysis.ROOTS[:len(OWNFILL)]


def panels(height):
    """Three task panels with room on the right of each for the names of the no-fMRI lines."""
    fig, axes = plt.subplots(1, 3, figsize=(6.9, height))
    fig.subplots_adjust(left=0.06, right=0.925, top=0.9, bottom=0.27, wspace=0.45)
    return fig, axes


def inputs():
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7), gridspec_kw={"width_ratios": [1.6, 1]})
    rows = ["Spectator", "No movie (fMRI only)", "No fMRI (movie only)", "Linear forecaster, 180 s"]
    bars = {}
    for label in rows:
        f = forecast_steps(ROWS[label])
        if f is not None:
            curve(axes[0], label, sec, 100 * (1 - f[0] / f[1]))
        s, u = state_steps(ROWS[label]), updating_budgets(ROWS[label])
        bars[label] = (None if s is None else 100 * (1 - s[0].sum() / s[1].sum()), None if u is None else u.get("4"))
    axes[0].set_xlabel("seconds ahead"); axes[0].set_ylabel("Forecast skill (%)"); axes[0].legend(frameon=False)
    labs = [l for l in rows if l != "Linear forecaster, 180 s"]
    x = np.arange(2)
    for i, label in enumerate(labs):
        v = [np.nan if b is None else b for b in bars[label]]
        axes[1].bar(x + (i - 1) * 0.27, v, 0.25, color=STYLE[label]["color"], label=label)
        for xi, vi in zip(x + (i - 1) * 0.27, v):
            if np.isfinite(vi):
                axes[1].text(xi, max(vi, 0) + 0.3, f"{vi:.1f}", ha="center", va="bottom", fontsize=5)
    axes[1].set_xticks(x, ["State", "Updating"]); axes[1].axhline(0, color="#cccccc", lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "inputs.pdf"); fig.savefig(OUT / "inputs.png", dpi=200)


def inside():
    res = json.loads((A / "results-collected.json").read_text())
    share = np.array(list(res["state_share_by_seconds"].values()))
    eig = np.load(A / "spectator/eigenvalues.npy")
    tau = -TR / np.log(np.clip(np.abs(eig), 1e-9, 0.999999))
    imp = res["impulse_response"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.6))
    axes[0].plot(sec, 100 * share, color=STYLE["Spectator"]["color"]); axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("seconds ahead"); axes[0].set_ylabel("variance from the state (%)"); axes[0].set_title("What the inferred state carries")
    c = np.array(imp["top-50 parcels, signed mean"]); c = c / np.abs(c).max()
    axes[1].plot(imp["seconds"], c, color=STYLE["Spectator"]["color"], marker="o", ms=2); axes[1].axhline(0, color="#cccccc", lw=0.5)
    axes[1].set_xlabel("seconds after a movie change"); axes[1].set_ylabel("predicted response (a.u.)"); axes[1].set_title("Response to the movie")
    axes[2].hist(tau[tau > 0.5], bins=np.logspace(0, 2.6, 30), color="#888888"); axes[2].set_xscale("log")
    axes[2].set_xlabel("time constant (s)"); axes[2].set_ylabel("modes"); axes[2].set_title("Time constants of the dynamics")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "inside.pdf"); fig.savefig(OUT / "inside.png", dpi=200)


def cortex():
    from nilearn import datasets, plotting
    m = np.load(A / "cortex-maps.npz")
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    fig = plt.figure(figsize=(6.9, 1.75))
    panels = [("Share of the forecast from the inferred state", m["state_share"], "viridis"), ("Agreement between viewers (ISC)", m["isc"], "magma")]
    for p, (title, values, cmap) in enumerate(panels):
        sv = surface_values(values)
        lo, hi = np.nanpercentile(values, 2), np.nanpercentile(values, 98)
        for v, view in enumerate(("lateral", "medial")):
            ax = fig.add_axes([0.0 + p * 0.385 + v * 0.19, 0.08, 0.18, 0.74], projection="3d")
            plotting.plot_surf(fs["infl_left"], sv["L"], hemi="left", view=view, cmap=cmap, vmin=lo, vmax=hi, bg_map=fs["sulc_left"],
                               bg_on_data=True, axes=ax, colorbar=False, darkness=0.5)
        fig.text(0.02 + p * 0.385, 0.92, title, fontsize=7)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cax = fig.add_axes([0.07 + p * 0.385, 0.06, 0.24, 0.03])
        fig.colorbar(sm, cax=cax, orientation="horizontal").ax.tick_params(labelsize=5.5)
    ax = fig.add_axes([0.83, 0.22, 0.16, 0.66])
    r = np.corrcoef(m["isc"], m["state_share"])[0, 1]
    ax.scatter(m["isc"], 100 * m["state_share"], s=1.5, alpha=0.4, color="#444444", lw=0)
    ax.set_xlabel("ISC"); ax.set_ylabel("state share (%)"); ax.set_title(f"r = {r:.2f}")
    fig.savefig(OUT / "cortex.pdf", bbox_inches="tight"); fig.savefig(OUT / "cortex.png", dpi=200, bbox_inches="tight")



if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (sys.argv[1:] or ["inputs", "inside", "cortex"]):
        globals()[f]()
        print("drew", f, flush=True)
