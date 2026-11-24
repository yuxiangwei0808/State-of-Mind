"""Frozen-latent readout: shared trunk plus one linear head per subject.

Follows the readout structure described by the NeuroWorld preprint (Dong et
al., 2026); the trunk size is our choice. ``hidden_dim=0`` removes the trunk, leaving
one linear map per subject. It never updates the world model.
"""
import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self, state_dim, hidden_dim, subjects, n_parcels=1000):
        super().__init__()
        self.subjects = [str(subject) for subject in subjects]
        if len(set(self.subjects)) != len(self.subjects) or not self.subjects:
            raise ValueError("subjects must be a non-empty list of unique IDs")
        self.index = {subject: position for position, subject in enumerate(self.subjects)}
        if hidden_dim:
            self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU())
        else:
            self.trunk, hidden_dim = nn.Identity(), state_dim
        weight = torch.empty(len(self.subjects), hidden_dim, n_parcels)
        for head in weight:
            nn.init.kaiming_uniform_(head.T, a=5 ** 0.5)
        self.head_weight = nn.Parameter(weight)
        self.head_bias = nn.Parameter(torch.zeros(len(self.subjects), n_parcels))

    def subject_indices(self, subject_ids, device=None):
        unknown = sorted({str(s) for s in subject_ids} - set(self.index))
        if unknown:
            raise KeyError(f"Readout has no head for subjects {unknown[:3]}")
        return torch.tensor([self.index[str(s)] for s in subject_ids], dtype=torch.long, device=device)

    def forward(self, latents, subject_index):
        """latents [B, T, D], subject_index [B] -> predicted fMRI [B, T, parcels]."""
        if latents.ndim != 3 or subject_index.shape != (latents.shape[0],):
            raise ValueError("Expected latents [B, T, D] and one subject index per row")
        hidden = self.trunk(latents)
        weight = self.head_weight[subject_index]
        return torch.einsum("bth,bhp->btp", hidden, weight) + self.head_bias[subject_index][:, None]
