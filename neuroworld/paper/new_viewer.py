"""neuroworld/paper/new_viewer.py: how much of a new viewer's fMRI Spectator needs on HCP (the synthetic-data proposal's "a new viewer
with little data"). Spectator's encoder and predictor stay frozen; only the readout changes.

    python -m neuroworld.paper.new_viewer select --out <dir>        the shrinkage, chosen on training participants
    python -m neuroworld.paper.new_viewer select-run1 --out <dir>   the shrinkage, chosen on the new viewers' run 1
    python -m neuroworld.paper.new_viewer fit --out <dir>           every new viewer's readout for each k and selection
    python -m neuroworld.paper.new_viewer score --out <dir> --k <k> --tag <training|run1>   Forecast and Updating for one k
    python -m neuroworld.paper.new_viewer report --out <dir>        table, paired intervals against k = 0, figure

Readout of a new viewer from the first k minutes of their run 1 (k = 0, 1, 2, 5, 15, all 15.35 min):
  the pooled probe of Spectator on HCP (fitted on the 148 training participants, frozen release spectator-v1/hcp/model/probe.npz),
  plus a ridge fit of the viewer's residual (their fMRI minus the pooled probe's reading) on Spectator's predicted states.
  The fitting pairs are Spectator's open-loop rollouts from every other frame of those k minutes (180 s of history, frames before
  the run start empty, as in scoring), each predicted state with the fMRI frame it predicts, only frames inside the k minutes.
  The residual map has a ridge penalty lam on standardized states; its intercept, the viewer's mean residual, is shrunk by
  n / (n + kappa) with n the number of distinct target frames. lam and kappa are fixed across k (so more minutes shrink less) and
  are chosen once on 12 training participants: calibrated on the first k minutes of their run 1, scored on open-loop forecasts
  of their run 2, summed over k = 1, 2, 5, 15. The scored recordings never enter the choice.
  k = 0 is the pooled probe unchanged, the benchmark's row of Spectator on new participants.
Scored as the benchmark scores Spectator's HCP row (neuroworld/spectator/score.py): Forecast and Updating (4 frames revealed in the
minute, the headline) on the new participants' runs 2-3 (heldout-adapt) and run 4 (heldout-test). The new viewer's readout is
used everywhere the probe is: in the scores and where Updating fills hidden frames.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
from neuroworld.paths import OUTPUT

RELEASE = OUTPUT / "spectator-v1/hcp/model"
CONFIG = "neuroworld/configs/hcp_movies/base.yaml"
MINUTES = (0, 1, 2, 5, 15, "all")
SELECT_MINUTES = (1, 2, 5, 15)
LAMS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf)
KAPPAS = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, np.inf)
SELECT_PARTICIPANTS = 12
STRIDE = 2   # calibration origins every other frame
FIRST = 5    # first calibration origin: the fast window's 5 frames
SCORED = ("heldout-adapt", "heldout-test")
log = lambda text: print(text, flush=True)


def label(k):
    return "all run 1" if k == "all" else f"{k} min"


def row_name(k, tag=""):
    return f"spectator-hcp new viewer{' ' + tag if tag else ''}, {label(k)}"


class ViewerProbe:
    """The pooled probe, plus a residual map and intercept per new viewer (subject id -> (dW [D, 1000], db [1000]))."""

    def __init__(self, W, b, viewers=None, provenance=None):
        self.W, self.b, self.viewers = W, b, viewers or {}
        self.provenance = provenance or {"probe": str(RELEASE / "probe.npz")}

    def __call__(self, states, subjects):
        out = states @ self.W + self.b
        for i, subject in enumerate(subjects):
            if subject in self.viewers:
                dW, db = self.viewers[subject]
                out[i] += states[i] @ dW + db
        return out.astype(np.float32)


def setup(device="cuda"):
    """(data, adapter, pooled ViewerProbe) as score.latent_row builds Spectator's HCP row, with the release probe."""
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    from neuroworld.benchmark import grid as G
    from neuroworld.benchmark.adapters import RolloutAdapter
    from neuroworld.benchmark.data import BenchmarkData
    from neuroworld.spectator.score import Head
    from neuroworld.train.readout import load_world_model
    from neuroworld.utils.config import load_config
    G.HISTORY_SECONDS = 180.0
    data = BenchmarkData(CONFIG)
    cfg = load_config(RELEASE / "config.yaml")
    model, _ = load_world_model(cfg, RELEASE / "best.pt", device)
    Head.subjects = sorted({run.subject_id for split in ("train", "val", "heldout-fit", "heldout-adapt", "heldout-test")
                            for run in data.runs(split)})
    with np.load(RELEASE / "probe.npz", allow_pickle=False) as saved:
        pooled = ViewerProbe(saved["W 0"], saved["b 0"])  # float32, as the benchmark's Probe reads it
    states = getattr(model, "read_states", cfg["model"]["context_states"]) * getattr(model, "state_step", 1)
    adapter = RolloutAdapter("spectator-hcp", model, Head(pooled), cfg, device, states)
    return data, adapter, pooled


def forget(adapter):
    adapter._memory, adapter._memory_bytes = OrderedDict(), 0


def pairs(data, adapter, pooled, run, limit=None, origins=None):
    """(states [m, D], residual [m, 1000], frames [m]) of open-loop rollouts on run: origins every STRIDE frames from FIRST (or
    the given ones), targets only at valid frames before `limit`; the residual is the fMRI minus the pooled probe's reading."""
    from neuroworld.benchmark import queries as Q
    from neuroworld.benchmark.latent import _states
    H, n = data.grid.horizon_steps, run.n if limit is None else min(run.n, limit)
    if origins is None:
        ok = run.brain_ok
        origins = np.array([t for t in range(FIRST, n, STRIDE) if ok[t - FIRST:t].all()], dtype=np.int64)
    if not len(origins):
        return None
    batch = Q.batch(data.dataset, data.grid.tr, run, origins, data.grid.history_frames(), H)
    z = _states(adapter, batch)                                # [o, H, D]
    y, ok = Q.targets(run, origins, H)                         # [o, H, 1000], [o, H]
    frames = origins[:, None] + np.arange(H)[None]
    ok &= frames < n
    z, y, frames = z[ok].astype(np.float64), y[ok].astype(np.float64), frames[ok]
    return z, y - (z @ pooled.W.astype(np.float64) + pooled.b), frames


def moments(z, r, frames, scale):
    """Sufficient statistics of one viewer's calibration pairs, the states standardized by `scale`."""
    zs = z / scale
    mz, mr = zs.mean(0), r.mean(0)
    zc = zs - mz
    lam, V = np.linalg.eigh(zc.T @ zc)
    return {"lam": np.clip(lam, 0, None), "V": V, "VC": V.T @ (zc.T @ (r - mr)), "mz": mz, "mr": mr,
            "frames": int(len(np.unique(frames))), "pairs": int(len(z))}


def solve(m, scale, lam, kappa):
    """(dW [D, 1000], db [1000]) in the original state coordinates: ridge penalty lam on the standardized states, the intercept
    shrunk by n / (n + kappa) over n distinct target frames."""
    dWs = np.zeros_like(m["VC"]) if np.isinf(lam) else m["V"] @ (m["VC"] / (m["lam"] + lam)[:, None])
    shrink = 0.0 if np.isinf(kappa) else m["frames"] / (m["frames"] + kappa)
    db = shrink * (m["mr"] - m["mz"] @ dWs)
    return dWs / scale[:, None], db


def scale_of(states):
    s = states.std(0)
    return np.where(s > 1e-8, s, 1.0)


def select(args):
    """lam and kappa on SELECT_PARTICIPANTS training participants: calibrated on their run 1, scored on their run 2."""
    data, adapter, pooled = setup(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_subject = {}
    for run in data.runs("train"):
        by_subject.setdefault(run.subject_id, {})[run.run_id.split("/")[3]] = run
    subjects = [s for s in sorted(by_subject) if {"run-01", "run-02"} <= set(by_subject[s])][:SELECT_PARTICIPANTS]
    started, calib, evals = time.time(), {}, {}
    for s in subjects:
        calib[s] = pairs(data, adapter, pooled, by_subject[s]["run-01"])
        from neuroworld.benchmark import queries as Q
        run2 = by_subject[s]["run-02"]
        evals[s] = pairs(data, adapter, pooled, run2, origins=Q.forecast_origins(run2, data.grid)[::2])
        for run in (by_subject[s]["run-01"], run2):
            run.__dict__.pop("brain", None)
        log(f"select: {s} calibration pairs {len(calib[s][0])}, evaluation pairs {len(evals[s][0])} ({time.time() - started:.0f}s)")
    scale = scale_of(np.concatenate([c[0][::7] for c in calib.values()]))
    errors = np.zeros((len(LAMS), len(KAPPAS)))
    by_k = {k: np.zeros((len(LAMS), len(KAPPAS))) for k in SELECT_MINUTES}
    for s in subjects:
        z, r, frames = calib[s]
        ze, re, _ = evals[s]
        for k in SELECT_MINUTES:
            keep = frames < 60 * k
            m = moments(z[keep], r[keep], frames[keep], scale)
            for i, lam in enumerate(LAMS):
                for j, kappa in enumerate(KAPPAS):
                    dW, db = solve(m, scale, lam, kappa)
                    e = float(((re - ze @ dW - db) ** 2).sum())
                    errors[i, j] += e
                    by_k[k][i, j] += e
    base = errors[LAMS.index(np.inf), KAPPAS.index(np.inf)]
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    chosen = {"lam": float(LAMS[i]), "kappa": float(KAPPAS[j]), "participants": subjects,
              "error relative to the pooled probe (all k)": {f"lam {LAMS[a]:g}, kappa {KAPPAS[b]:g}": float(errors[a, b] / base)
                                                             for a in range(len(LAMS)) for b in range(len(KAPPAS))},
              "chosen, relative to the pooled probe, by k": {str(k): float(by_k[k][i, j] / by_k[k][LAMS.index(np.inf), KAPPAS.index(np.inf)])
                                                            for k in SELECT_MINUTES}}
    np.save(out / "scale.npy", scale)
    (out / "selection.json").write_text(json.dumps(chosen, indent=1, default=float))
    log(f"select: lam {chosen['lam']:g}, kappa {chosen['kappa']:g}; error relative to the pooled probe by k "
        f"{chosen['chosen, relative to the pooled probe, by k']} ({time.time() - started:.0f}s)")


def select_run1(args):
    """lam and kappa on the new viewers' run 1 (never scored), as the benchmark's adapt.ridge_head chooses a new participant's
    penalty on the last fifth of run 1: calibrated on the first k minutes (k = 1, 2, 5), scored on open-loop forecasts of
    the last fifth of the same run. Fixed across k, like the training-participant choice."""
    data, adapter, pooled = setup(args.device)
    out = Path(args.out)
    scale = np.load(out / "scale.npy")
    errors = np.zeros((len(LAMS), len(KAPPAS)))
    by_k = {k: np.zeros((len(LAMS), len(KAPPAS))) for k in (1, 2, 5)}
    started, viewers = time.time(), 0
    for run in data.runs("heldout-fit"):
        found = pairs(data, adapter, pooled, run)
        run.__dict__.pop("brain", None)
        if found is None:
            continue
        z, r, frames = found
        last = int(0.8 * run.n)
        held = frames >= last
        if held.sum() < 2:
            continue
        viewers += 1
        ze, re = z[held], r[held]
        for k in by_k:
            keep = frames < 60 * k
            if keep.sum() < 2:
                continue
            m = moments(z[keep], r[keep], frames[keep], scale)
            for i, lam in enumerate(LAMS):
                for j, kappa in enumerate(KAPPAS):
                    dW, db = solve(m, scale, lam, kappa)
                    e = float(((re - ze @ dW - db) ** 2).sum())
                    errors[i, j] += e
                    by_k[k][i, j] += e
        log(f"select-run1: {run.subject_id} ({time.time() - started:.0f}s)")
    base = errors[LAMS.index(np.inf), KAPPAS.index(np.inf)]
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    chosen = {"lam": float(LAMS[i]), "kappa": float(KAPPAS[j]), "viewers": viewers,
              "validation": "first k minutes of each new viewer's run 1 against open-loop forecasts of its last fifth",
              "error relative to the pooled probe (k = 1, 2, 5)": {f"lam {LAMS[a]:g}, kappa {KAPPAS[b]:g}": float(errors[a, b] / base)
                                                                  for a in range(len(LAMS)) for b in range(len(KAPPAS))},
              "chosen, relative to the pooled probe, by k": {str(k): float(by_k[k][i, j] / by_k[k][LAMS.index(np.inf), KAPPAS.index(np.inf)])
                                                            for k in by_k}}
    (out / "selection-run1.json").write_text(json.dumps(chosen, indent=1, default=float))
    log(f"select-run1: lam {chosen['lam']:g}, kappa {chosen['kappa']:g}; error relative to the pooled probe by k "
        f"{chosen['chosen, relative to the pooled probe, by k']} ({time.time() - started:.0f}s)")


def fit(args):
    """Every new viewer's readout for each k, for both selections, from one pass over their run 1: probes/<tag>-<k>.npz."""
    data, adapter, pooled = setup(args.device)
    out = Path(args.out)
    scale = np.load(out / "scale.npy")
    selections = {"training": json.loads((out / "selection.json").read_text()), "run1": json.loads((out / "selection-run1.json").read_text())}
    calib = {}
    for run in data.runs("heldout-fit"):
        calib[run.subject_id] = pairs(data, adapter, pooled, run)
        run.__dict__.pop("brain", None)
    log(f"calibration pairs from {len(calib)} new viewers' run 1")
    (out / "probes").mkdir(exist_ok=True)
    for tag, chosen in selections.items():
        for k in MINUTES[1:]:
            arrays, frames_used = {}, {}
            for subject, found in calib.items():
                if found is None:
                    continue
                z, r, frames = found
                keep = np.ones(len(z), bool) if k == "all" else frames < 60 * k
                if keep.sum() < 2:
                    continue
                m = moments(z[keep], r[keep], frames[keep], scale)
                dW, db = solve(m, scale, chosen["lam"], chosen["kappa"])
                i = len(frames_used)
                arrays[f"dW {i}"], arrays[f"db {i}"] = dW.astype(np.float32), db.astype(np.float32)
                frames_used[subject] = m["frames"]
            meta = {"subjects": list(frames_used), "target frames": frames_used, "lam": chosen["lam"], "kappa": chosen["kappa"],
                    "minutes": str(k), "selection": tag}
            np.savez(out / "probes" / f"{tag}-{label(k).replace(' ', '-')}.npz", meta=json.dumps(meta), **arrays)
            log(f"fit {tag} {label(k)}: {len(frames_used)} viewers, median target frames {np.median(list(frames_used.values())):.0f}")


def score(args):
    """One k: Forecast on runs 2-3 and run 4, then Updating, with the readout of probes/<tag>-<k>.npz (k = 0: the pooled probe)."""
    from neuroworld.benchmark import forecast as F, updating as U
    from neuroworld.benchmark.latent import LatentProbe
    from neuroworld.spectator.score import Head
    data, adapter, pooled = setup(args.device)
    out = Path(args.out)
    store = out / "benchmark"
    k = args.k if args.k == "all" else int(args.k)
    viewers, meta = {}, {"minutes": "0"}
    if k != 0:
        with np.load(out / "probes" / f"{args.tag}-{label(k).replace(' ', '-')}.npz", allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"]))
            viewers = {s: (saved[f"dW {i}"], saved[f"db {i}"]) for i, s in enumerate(meta["subjects"])}
    probe = ViewerProbe(pooled.W, pooled.b, viewers, {"probe": str(RELEASE / "probe.npz"), "new viewers": len(viewers),
                                                     **{key: meta.get(key) for key in ("minutes", "lam", "kappa", "selection")}})
    adapter.readout = Head(probe)
    forget(adapter)
    tag = "" if k == 0 or args.tag == "training" else args.tag
    adapter.name = row_name(k, tag)
    row = LatentProbe(adapter, probe, f"{row_name(k, tag)} (latent)")
    started = time.time()
    for split in SCORED:
        F.run(data, [row], split, output=store, log=log)
    forget(adapter)
    U.run(data, [row], splits=SCORED, output=store, log=log)
    log(f"{row.name}: scored in {time.time() - started:.0f}s")


def report(args):
    """Both selections: "" (lam, kappa chosen on training participants) and "run1" (chosen on the new viewers' run 1); k = 0 is
    the pooled probe for both."""
    from neuroworld.benchmark import profile as P
    out = Path(args.out)
    P.ROOT = out / "benchmark"
    skill = lambda u: None if not u else 100 * (1 - sum(v[0] for v in u.values()) / sum(v[1] for v in u.values()))
    fmt = lambda d: None if d is None or d.get("low") is None else [round(d["value"], 2), round(d["low"], 2), round(d["high"], 2)]
    def cells(name):
        return {"Forecast": P._unit_sums("forecast", "hcp_movies", list(SCORED), name),
                "Forecast, runs 2-3": P._unit_sums("forecast", "hcp_movies", ["heldout-adapt"], name),
                "Forecast, run 4": P._unit_sums("forecast", "hcp_movies", ["heldout-test"], name),
                "Updating": P._unit_sums("updating", "hcp_movies", list(SCORED), name),
                "Updating, runs 2-3": P._unit_sums("updating", "hcp_movies", ["heldout-adapt"], name),
                "Updating, run 4": P._unit_sums("updating", "hcp_movies", ["heldout-test"], name)}
    zero = cells(f"{row_name(0)} (latent)")
    tables = {}
    for tag, title in (("", "shrinkage chosen on 12 training participants"), ("run1", "shrinkage chosen on the new viewers' run 1")):
        table = {}
        for k in MINUTES:
            c = zero if k == 0 else cells(f"{row_name(k, tag)} (latent)")
            if not c["Forecast"]:
                continue
            row = {key: None if v is None else round(skill(v), 2) for key, v in c.items()}
            if k != 0:
                for key in ("Forecast", "Updating"):
                    if c[key] and zero[key]:
                        row[f"{key} minus 0 min"] = fmt(P._paired(c[key], zero[key]))
            table[label(k)] = row
            log(f"[{title}] {label(k)}: {json.dumps(row)}")
        tables[title] = table
    (out / "results.json").write_text(json.dumps(tables, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    for ax, key in zip(axes, ("Forecast", "Updating")):
        for (title, table), style in zip(tables.items(), ("--", "-")):
            xs = [k for k in MINUTES if label(k) in table and table[label(k)][key] is not None]
            ax.plot([15.35 if k == "all" else k for k in xs], [table[label(k)][key] for k in xs], style, marker="o", ms=3,
                    color="#C0392B", label=title)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 5, 15], ["0", "1", "2", "5", "15"])
        ax.set_xlabel("minutes of the new viewer's fMRI"); ax.set_title(key, fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("skill (%)"); axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "new-viewer.pdf"); fig.savefig(out / "new-viewer.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "select-run1", "fit", "score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda")
        if name == "score":
            p.add_argument("--k", required=True, help="0, 1, 2, 5, 15 or all")
            p.add_argument("--tag", default="run1", help="the selection: training or run1")
    args = parser.parse_args()
    {"select": select, "select-run1": select_run1, "fit": fit, "score": score, "report": report}[args.command](args)
