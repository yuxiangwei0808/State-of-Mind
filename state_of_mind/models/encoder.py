"""Our fMRI encoders: linear (or one hidden layer), or a small Transformer over the window's frames; trained from scratch, or loaded
from a stage-0 run in the Transformer world model."""
import torch
from torch import nn


class BrainEncoder(nn.Module):
    def __init__(self, state_window_trs=1, state_dim=1024, hidden_dim=0):
        super().__init__()
        self.state_window_trs = state_window_trs
        # hidden_dim > 0: a nonlinear encoder, one hidden layer in place of the linear map.
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


class FrameTransformerEncoder(nn.Module):
    """A state from the last state_window_trs frames through a small Transformer: each frame's 1000 parcels are one token (a shared
    linear embedding plus a learned embedding of its place in the window), a learned readout token joins them, the tokens attend to
    each other through depth pre-norm blocks, and the readout token's output, normalized (or not, with output_norm False), maps to the
    state. With linear_path, a linear encoder of the same frames (a BrainEncoder) is added to that map, so the encoder nests the linear
    one; the readout's map then starts at zero, so training starts from the linear path alone."""

    def __init__(self, state_window_trs=5, state_dim=1024, width=512, depth=2, heads=8, mlp_dim=1024, dropout=0.1, output_norm=True,
                 linear_path=False):
        super().__init__()
        self.state_window_trs = state_window_trs
        self.embed = nn.Linear(1000, width)
        self.place = nn.Parameter(torch.randn(state_window_trs, width) * 0.02)
        self.readout_token = nn.Parameter(torch.randn(width) * 0.02)
        # Unused: the token of the removed masked stage 0's hidden frames, kept so saved encoders load and new ones initialize as before.
        self.mask_token = nn.Parameter(torch.randn(width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, mlp_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width) if output_norm else nn.Identity()
        self.projection = nn.Linear(width, state_dim)
        self.linear = BrainEncoder(state_window_trs, state_dim) if linear_path else None
        if linear_path:
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, brain):
        if brain.ndim != 3 or brain.shape[-1] != 1000:
            raise ValueError("brain must have shape [batch, time, 1000]")
        k = self.state_window_trs
        count = brain.shape[1] - k + 1
        if count < 1:
            raise ValueError("Not enough brain frames to encode a state")
        tokens = self.embed(brain)
        windows = tokens.unfold(1, k, 1).transpose(-1, -2).flatten(0, 1) + self.place  # [B * state time, K, width], oldest first
        # In chunks of windows: PyTorch's inference path for these layers fails on a CUDA launch limit beyond about 8k windows.
        states = torch.cat([self._read(part) for part in windows.split(self.chunk)]).unflatten(0, (brain.shape[0], count))
        return states if self.linear is None else states + self.linear(brain)

    chunk = 4096

    body_fixed = False

    def fix_body(self):
        """Fix the part before the output map (embedding, place and readout and mask tokens, blocks, the LayerNorm): no gradients and
        no dropout from now on, in training too, so it computes what a cache of its outputs would hold."""
        for part in (self.embed, self.blocks, self.norm):
            part.requires_grad_(False)
        for parameter in (self.place, self.readout_token, self.mask_token):
            parameter.requires_grad_(False)
        self.body_fixed = True
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.body_fixed:
            self.blocks.eval()
        return self

    def _read(self, windows):
        h = torch.cat((self.readout_token.expand(windows.shape[0], 1, -1).to(windows.dtype), windows), 1)
        return self.projection(self.norm(self.blocks(h)[:, 0]))

    def zero_point(self):
        """[D] what a window of zero frames encodes to, with no dropout."""
        training = self.training
        self.eval()
        try:
            return self(self.place.new_zeros(1, self.state_window_trs, 1000))[0, 0]
        finally:
            self.train(training)


def frame_encoder(m, state_window_trs, state_dim):
    """The encoder of one state's frames that model settings m choose (model.encoder_kind)."""
    if m.get("encoder_kind", "linear") == "transformer":
        return FrameTransformerEncoder(state_window_trs, state_dim, m["encoder_width"], m["encoder_depth"], m["encoder_heads"],
                                       m["encoder_mlp_dim"], m["encoder_dropout"], m.get("encoder_output_norm", True),
                                       m.get("encoder_linear_path", False))
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
