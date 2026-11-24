"""Stimulus context: one condition per transition from the last N projected stimulus rows.

Configured by ``model.stimulus_context`` (none | concat | linear) and
``model.stimulus_context_trs`` (N). The newest row of each window is the frame
``data.hrf_delay_seconds`` before the predicted brain frame.

- ``none``: one row per transition (N = 1).
Each row is W values: 1152 projected values, or 2048 with ``model.stimulus_features: tribev2_context``.

- ``concat``: no parameters; the N rows are placed side by side (oldest first) and
  the predictor's existing condition layer reads N x W values.
- ``concat-audio``: rows hold contextual values and then raw audio values. The contextual values of all N rows are
  placed side by side, then the raw audio of the middle row only.
- ``linear``: one linear layer downsamples the N rows to W values before the
  predictor's condition layer. It starts as the identity on the newest row, so a
  new run begins from the one-row behaviour of ``none``.
"""
import torch
from torch import nn

KINDS = ("none", "concat", "linear", "concat-audio")
AUDIO_WIDTH = 1024  # the raw audio values at the end of each row, with concat-audio


class StimulusContext(nn.Module):
    def __init__(self, kind="none", rows=1, width=1152):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"stimulus_context must be one of {KINDS}")
        if type(rows) is not int or rows < 1:
            raise ValueError("stimulus_context_trs must be a positive integer")
        if kind == "none" and rows != 1:
            raise ValueError("stimulus_context: none uses one row; choose concat or linear for more")
        self.kind, self.rows = kind, rows
        # Width of each transition's condition as seen by the predictor.
        self.out_width = rows * width if kind == "concat" else rows * (width - AUDIO_WIDTH) + AUDIO_WIDTH if kind == "concat-audio" else width
        if kind == "linear":
            self.mix = nn.Linear(rows * width, width)
            with torch.no_grad():
                self.mix.weight.zero_()
                self.mix.weight[:, -width:] = torch.eye(width)
                self.mix.bias.zero_()

    def forward(self, projected):
        """[B, T + N - 1, W] projected rows -> [B, T, out_width], one condition per transition.

        Transition t uses rows t .. t + N - 1 (its newest row is t + N - 1), so no
        transition sees a row newer than its own.
        """
        if self.rows == 1:
            return projected
        count = projected.shape[1] - self.rows + 1
        if count < 1:
            raise ValueError(f"Stimulus context needs at least {self.rows} rows")
        if self.kind == "concat-audio":
            windows = projected.unfold(1, self.rows, 1).transpose(-1, -2)  # [B, T, N, W]
            return torch.cat((windows[..., :-AUDIO_WIDTH].flatten(2), windows[:, :, self.rows // 2, -AUDIO_WIDTH:]), dim=-1)
        windows = projected.unfold(1, self.rows, 1).transpose(-1, -2).flatten(2)  # [B, T, N * W]
        return windows if self.kind == "concat" else self.mix(windows)
