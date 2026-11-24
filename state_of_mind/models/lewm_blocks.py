"""AdaLN-zero blocks adapted from lucas-maes/le-wm, commit 8edfeb336732.

Uses torch reshapes instead of einops and fixes dim_head to state_dim / heads.
Attention accepts an explicit additive mask for causality and relative-position
bias. The predictor bounds history by slicing independent input windows before
applying these blocks. This is code reuse, not pretrained weights.
"""
import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation (le-wm/module.py::modulate)."""
    return x * (1 + scale) + shift


def banded_causal_mask(length, window, bias=None, *, device=None, dtype=torch.float32):
    """Additive [length, length] mask: position i attends to j only when 0 <= i-j < window.

    ``bias`` optionally holds one learned value per offset ``i-j`` that is added
    to the allowed logits. Disallowed positions receive ``-inf``. ``window=1``
    makes every position attend to itself only (a Markov transition).
    """
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    index = torch.arange(length, device=device)
    offsets = index[:, None] - index[None, :]
    allowed = (offsets >= 0) & (offsets < window)
    if bias is None:
        values = torch.zeros(length, length, device=device, dtype=dtype)
    else:
        if bias.shape != (window,):
            raise ValueError("bias must hold one value per offset inside the window")
        values = bias.to(dtype)[offsets.clamp(0, window - 1)]
    return values.masked_fill(~allowed, float("-inf"))


def _heads(dim: int) -> int:
    return 4 if dim % 4 == 0 else 1


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        heads = heads or _heads(dim)
        self.heads = heads
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * self.dim_head * heads, bias=False)
        inner = self.dim_head * heads
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout)) if heads != 1 else nn.Identity()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.heads, self.dim_head).transpose(1, 2) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype), dropout_p=drop)
        out = out.transpose(1, 2).reshape(B, T, self.heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (le-wm/module.py::ConditionalBlock)."""

    def __init__(self, dim: int, heads: int = 4, mlp_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim or 4 * dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
