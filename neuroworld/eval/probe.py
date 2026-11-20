"""Closed-form ridge probe from rollout latents to fMRI, used to select stage-1 checkpoints.

Latent errors cannot rank epochs because the latent space itself changes during
training. The probe decodes rollouts to fMRI so every epoch (and every state
size) is compared on the primary fMRI metric. Stage 2 still trains the real
readout; the probe only ranks checkpoints.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from neuroworld.eval.rollout import move_batch


class RidgeReadout(nn.Module):
    """Per-subject affine map from latents to parcels, with the Readout call interface."""

    def __init__(self, subjects, weights):
        super().__init__()
        self.subjects = [str(subject) for subject in subjects]
        self.index = {subject: position for position, subject in enumerate(self.subjects)}
        self.register_buffer("weights", weights)  # [subjects, D + 1, parcels]

    def subject_indices(self, subject_ids, device=None):
        unknown = sorted({str(s) for s in subject_ids} - set(self.index))
        if unknown:
            raise KeyError(f"Probe has no weights for subjects {unknown[:3]}")
        return torch.tensor([self.index[str(s)] for s in subject_ids], dtype=torch.long, device=device)

    def forward(self, latents, subject_index):
        inputs = torch.cat((latents, latents.new_ones(*latents.shape[:-1], 1)), dim=-1)
        return torch.einsum("btd,bdp->btp", inputs.float(), self.weights[subject_index])


def probe_subset(dataset, windows, seed=0):
    """A fixed random subset of training windows, identical for every epoch."""
    if len(dataset) <= windows:
        return dataset
    chosen = np.sort(np.random.default_rng(seed).choice(len(dataset), windows, replace=False))
    return Subset(dataset, chosen.tolist())


@torch.no_grad()
def fit_ridge_probe(model, loader, device, *, prefix_trs, ridge=0.1):
    """Fit fMRI = [latent, 1] @ W per subject on rollouts of the loader's windows.

    The penalty is ``ridge`` times the mean latent energy per sample, so it does not
    depend on the latent scale; the bias is not penalized.
    """
    model.eval()
    k = model.encoder.state_window_trs
    gram, cross = {}, {}
    for raw in loader:
        batch = move_batch(raw, device)
        brain = batch["brain"]
        horizon = brain.shape[1] - (k + prefix_trs - 1)
        latents = model.rollout(brain[:, :k + prefix_trs - 1], batch["stimuli"], horizon_trs=horizon)
        inputs = torch.cat((latents, latents.new_ones(*latents.shape[:-1], 1)), dim=-1).double()
        targets = brain[:, k + prefix_trs - 1:].double()
        subjects = [str(subject) for subject in raw["subject_id"]]
        for subject in set(subjects):
            rows = torch.tensor([s == subject for s in subjects], device=inputs.device)
            x, y = inputs[rows].flatten(0, 1), targets[rows].flatten(0, 1)
            gram[subject] = gram.get(subject, 0) + x.T @ x
            cross[subject] = cross.get(subject, 0) + x.T @ y
    if not gram:
        raise ValueError("The probe loader yielded no windows")
    weights = []
    for subject in sorted(gram):
        g = gram[subject]
        penalty = torch.eye(len(g), dtype=g.dtype, device=g.device) * ridge * g.diagonal()[:-1].mean()
        penalty[-1, -1] = 0
        weights.append(torch.linalg.solve(g + penalty, cross[subject]).float())
    return RidgeReadout(sorted(gram), torch.stack(weights)).to(device)
