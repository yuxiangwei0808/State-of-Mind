"""Declared adaptation for new participants: fit only a linear readout head.

Our readouts are a shared trunk plus one linear head per participant (models/readout.py). For a held-out
participant, the world model and trunk stay frozen and a new head is fitted by ridge regression on that
participant's adaptation run (HCP run 1), never on scored frames:
  latent world model (the Transformer world model)  trunk(encoded state ending at frame t) -> fMRI at t
  fMRI forecaster (the fMRI forecaster and stage-0 forecaster) trunk(predicted next state from states up to t) -> fMRI at t + 1
The penalty is chosen on the last fifth of the adaptation frames, then the head is refitted on all of them.
"""
import numpy as np
import torch

from state_of_mind.models.readout import Readout

ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
WINDOW_STATES = 16


def _features(adapter, run, starts):
    """(trunk features [N, H], target frames [N]) for windows starting at the given frames."""
    model, device = adapter.model, adapter.device
    k, states = adapter.state_frames, WINDOW_STATES
    length = k + states - 1
    head = model.readout if adapter.forecaster else adapter.readout
    feats, frames = [], []
    for start in starts:
        brain = torch.from_numpy(np.asarray(run.brain[start:start + length], dtype=np.float32))[None].to(device)
        with torch.no_grad():
            z = model.encode(brain)  # [1, states, D]
            if adapter.forecaster:
                # Teacher-forced predicted states: state r (ending at start+k-1+r) predicts frame start+k+r.
                first = start + k - adapter.stimulus.lag - (adapter.stimulus.context - 1)
                rows = adapter.stimulus.rows(run.timeline_id, int(first), states - 1 + adapter.stimulus.context - 1)
                stimuli = {name: torch.from_numpy(v)[None].to(device) for name, v in rows.items()}
                conditions = model.encode_stimulus(stimuli) if model.mode == "stimulus" else None
                hidden = model.predict(z[:, :-1], conditions)
                target = start + k + np.arange(states - 1)
            else:
                hidden = z
                target = start + k - 1 + np.arange(states)
            feats.append(head.trunk(hidden)[0].float().cpu().numpy())
        frames.append(target)
    return np.concatenate(feats), np.concatenate(frames)


def _windows(run, length, stride):
    """Window starts over the whole run (rest included). Invalid target frames are dropped from the fit; inputs
    keep the bundle's stored values, so motion-heavy participants still get a head."""
    return np.arange(0, run.n - length + 1, stride)


def ridge_head(x, y, alphas=ALPHAS):
    """(weight [H, P], bias [P], alpha): penalty chosen on the last fifth, refitted on everything."""
    split = int(len(x) * 0.8)
    def fit(xs, ys, alpha):
        mx, my = xs.mean(0), ys.mean(0)
        xc = xs - mx
        w = np.linalg.solve(xc.T @ xc + alpha * len(xs) * np.eye(xs.shape[1]), xc.T @ (ys - my))
        return w, my - mx @ w
    def score(w, b):
        p = x[split:] @ w + b
        t = y[split:]
        pc, tc = p - p.mean(0), t - t.mean(0)
        return float(((pc * tc).sum(0) / np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0) + 1e-12)).mean())
    scores = {a: score(*fit(x[:split], y[:split], a)) for a in alphas}
    alpha = max(scores, key=scores.get)
    return (*fit(x, y, alpha), alpha)


def adapt(adapter, data, fit_runs, log=print):
    """Add a fitted head for every participant in fit_runs to the adapter's readout; returns {subject: alpha}."""
    head = adapter.model.readout if adapter.forecaster else adapter.readout
    by_subject = {}
    for run in fit_runs:
        by_subject.setdefault(run.subject_id, []).append(run)
    new = sorted(set(by_subject) - set(head.subjects))
    if not new:
        return {}
    weights, biases, chosen = [], [], {}
    for subject in new:
        xs, ys = [], []
        for run in by_subject[subject]:
            length = adapter.state_frames + WINDOW_STATES - 1
            starts = _windows(run, length, WINDOW_STATES)
            x, frames = _features(adapter, run, starts)
            ok = run.brain_ok[frames]
            xs.append(x[ok])
            ys.append(np.asarray(run.brain[frames[ok]], dtype=np.float64))
            run.__dict__.pop("brain", None)
        w, b, alpha = ridge_head(np.concatenate(xs).astype(np.float64), np.concatenate(ys))
        weights.append(w), biases.append(b)
        chosen[subject] = alpha
    extended = Readout(head.trunk[0].in_features if hasattr(head.trunk, "__getitem__") else head.head_weight.shape[1],
                       head.head_weight.shape[1] if hasattr(head.trunk, "__getitem__") else 0,
                       head.subjects + new, n_parcels=head.head_weight.shape[2]).to(head.head_weight.device)
    with torch.no_grad():
        if hasattr(head.trunk, "__getitem__"):
            extended.trunk.load_state_dict(head.trunk.state_dict())
        extended.head_weight.copy_(torch.cat([head.head_weight, torch.tensor(np.stack(weights), dtype=head.head_weight.dtype,
                                                                            device=head.head_weight.device)]))
        extended.head_bias.copy_(torch.cat([head.head_bias, torch.tensor(np.stack(biases), dtype=head.head_bias.dtype,
                                                                        device=head.head_bias.device)]))
    extended.eval().requires_grad_(False)
    if adapter.forecaster:
        adapter.model.readout = extended
    else:
        adapter.readout = extended
    if log:
        log(f"adapted {len(new)} participants (linear heads; median alpha {np.median(list(chosen.values())):g})")
    return chosen


def calibrate(model, data, split, log=print):
    """Before scoring new participants (any held-out split), give a model with participant-specific heads a head
    for each of them, fitted on their run 1; participants that already have one are skipped. A latent row
    (latent.LatentProbe) calibrates the model it reads out: that model's rollouts decode through its heads."""
    from state_of_mind.benchmark.latent import LatentProbe
    if isinstance(model, LatentProbe):
        model = model.model
    if split.startswith("heldout") and hasattr(model, "state_frames"):
        adapt(model, data, data.runs("heldout-fit"), log=log)
