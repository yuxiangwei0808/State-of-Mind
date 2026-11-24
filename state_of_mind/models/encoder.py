"""Spectator's fMRI encoders: a linear fast encoder (BrainEncoder) and a two-timescale encoder (HistoryEncoder)."""
import torch
from torch import nn


class BrainEncoder(nn.Module):
    def __init__(self, state_window_trs=1, state_dim=1024, hidden_dim=0):
        super().__init__()
        self.state_window_trs = state_window_trs
        self.projection = nn.Linear(1000 * state_window_trs, state_dim) if not hidden_dim else nn.Sequential(
            nn.Linear(1000 * state_window_trs, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim))

    def forward(self, brain):
        if brain.ndim != 3 or brain.shape[-1] != 1000:
            raise ValueError("brain must have shape [batch, time, 1000]")
        k = self.state_window_trs
        count = brain.shape[1] - k + 1
        if count < 1:
            raise ValueError("Not enough brain frames to encode a state")
        # [B, state time, K, 1000]; oldest frame first, no temporal pooling.
        windows = torch.stack([brain[:, offset:offset + count] for offset in range(k)], dim=2)
        return self.projection(windows.flatten(2))

    def zero_point(self):
        """[D] what a window of zero frames (unmeasured, or before a run's start) encodes to."""
        if isinstance(self.projection, nn.Linear):
            return self.projection.bias
        return self.projection(self.projection[0].weight.new_zeros(self.projection[0].in_features))


def frame_encoder(m, state_window_trs, state_dim):
    """Build the fast frame encoder from model settings m."""
    if m.get("encoder_kind", "linear") != "linear":
        raise ValueError(f"Unsupported encoder_kind {m.get('encoder_kind')!r}; only 'linear' is supported")
    return BrainEncoder(state_window_trs, state_dim, m.get("encoder_hidden_dim", 0))


class HistoryEncoder(nn.Module):
    """A state of two latents from the last state_window_trs frames: the fast latent, a BrainEncoder of the newest fast_window_trs
    frames, then a slow latent over the same encoder's states at every lag the window holds. lag_weights (oldest first,
    [lags], or with per_dimension [D, lags], each latent dimension its own) weight them about what zero frames (unmeasured or
    before a run's start) encode to, and scale sets the slow latent's spread about that point. Both are fixed; with
    trainable_lags the weights train with the encoder, since they read the coordinates the encoder moves. extra_branches k > 0 adds
    k more slow latents after it, each with its own lag weights and scale (extra_lag_weights, extra_scale), such as sums fitted to
    other horizons."""

    def __init__(self, state_window_trs, fast_window_trs, fast_dim, per_dimension=False, trainable_lags=False, extra_branches=0, fast=None):
        super().__init__()
        self.state_window_trs = state_window_trs
        self.fast = BrainEncoder(fast_window_trs, fast_dim) if fast is None else fast
        lags = state_window_trs - fast_window_trs + 1
        weights = torch.full((fast_dim, lags) if per_dimension else (lags,), float("nan"))
        if trainable_lags:
            self.lag_weights = nn.Parameter(weights)
        else:
            self.register_buffer("lag_weights", weights)
        self.register_buffer("scale", torch.tensor(float("nan")))
        self.extra_branches = extra_branches
        if extra_branches:
            extra = torch.full((extra_branches, *weights.shape), float("nan"))
            if trainable_lags:
                self.extra_lag_weights = nn.Parameter(extra)
            else:
                self.register_buffer("extra_lag_weights", extra)
            self.register_buffer("extra_scale", torch.full((extra_branches,), float("nan")))

    def forward(self, brain):
        # In 32-bit floats even under bfloat16 training: the slow latent changes by a few percent per frame, near bfloat16's resolution.
        with torch.autocast(brain.device.type, enabled=False):
            fast = self.fast(brain.float())
            lags = self.lag_weights.shape[-1]
            if fast.shape[1] < lags:
                raise ValueError("Not enough brain frames to encode a state")
            bias = self.fast.zero_point()
            windows = (fast - bias).unfold(1, lags, 1)  # [B, state time, D, lags]
            slows = [bias + self.scale * (windows * self.lag_weights).sum(-1)]
            for weights, scale in zip(self.extra_lag_weights if self.extra_branches else (), self.extra_scale if self.extra_branches else ()):
                slows.append(bias + scale * (windows * weights).sum(-1))
            return torch.cat((fast[:, lags - 1:], *slows), dim=-1)
