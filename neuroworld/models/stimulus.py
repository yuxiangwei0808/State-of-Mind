"""Stimulus inputs: per-frame TRIBEv2 projections, or precomputed contextual rows.

``StimulusEncoder`` (``model.stimulus_features: projections``) applies TRIBEv2's frozen per-modality
projections to native-TR features. Source: facebookresearch/tribev2, commit 38b3db073a7f,
tribev2/model.py::FmriEncoderModel.aggregate_features. Adaptation: inputs are already native-TR
[B,T,2,D] arrays; no temporal Transformer or fMRI readout.

``PrecomputedStimulus`` (``tribev2_context``) passes through rows prepared offline by
neuroworld/nbs_data/stimuli/tribev2_context.py: TRIBEv2's frozen projections, temporal Transformer
(causal attention) and low-rank head, pooled to the TR grid.
"""
from pathlib import Path
import torch
from torch import nn
from neuroworld.models.tribev2 import CONTEXT_DIM, MODALITY_DIMS

PROJECTED_DIM = 3 * 384  # the three projected modalities side by side
STIMULUS_DIMS = {"projections": PROJECTED_DIM, "tribev2_context": CONTEXT_DIM}


class StimulusEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, modality_dropout=0.0):
        super().__init__()
        # TRIBEv2's modality dropout: while training, each modality of each sample is dropped
        # (zero after projection, like a missing modality) with this probability.
        self.modality_dropout = modality_dropout
        self.projectors = nn.ModuleDict({name: nn.Linear(2 * width, 384)
                                         for name, width in MODALITY_DIMS.items()})
        self.requires_grad_(False)
        self.weights_loaded = False
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TRIBEv2 checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        dims = checkpoint.get("model_build_args", {}).get("feature_dims", {})
        if list(dims) != list(MODALITY_DIMS) or any(tuple(dims[name]) != (2, width) for name, width in MODALITY_DIMS.items()):
            raise ValueError("TRIBEv2 checkpoint feature dimensions/order do not match text, audio, video")
        state = checkpoint.get("state_dict", {})
        for name, layer in self.projectors.items():
            tensors = {}
            for parameter in ("weight", "bias"):
                key = f"model.projectors.{name}.{parameter}"
                tensor = state.get(key)
                if tensor is None or tensor.shape != getattr(layer, parameter).shape:
                    raise ValueError(f"Missing or incompatible TRIBEv2 projection: {key}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite TRIBEv2 projection: {key}")
                tensors[parameter] = tensor
            layer.load_state_dict(tensors, strict=True)
        self.weights_loaded = True

    def forward(self, features):
        if not self.weights_loaded:
            raise RuntimeError("TRIBEv2 projection weights have not been loaded")
        if not features or set(features) - set(MODALITY_DIMS):
            raise ValueError("Expected available text/audio/video feature tensors")
        reference = next(iter(features.values()))
        batch, time = reference.shape[:2]
        projected = []
        for name, width in MODALITY_DIMS.items():
            if name not in features:
                # Missing modalities are zero AFTER projection, so bias is excluded.
                projected.append(reference.new_zeros(batch, time, 384))
                continue
            values = features[name]
            if values.shape != (batch, time, 2, width):
                raise ValueError(f"Invalid {name} features: {tuple(values.shape)}")
            output = self.projectors[name](values.flatten(2))
            if self.training and self.modality_dropout > 0:
                keep = torch.rand(batch, 1, 1, device=output.device) >= self.modality_dropout
                output = output * keep
            projected.append(output)
        return torch.cat(projected, dim=-1)


class PrecomputedStimulus(nn.Module):
    """Rows prepared offline (``stimuli["context"]`` [B, T, CONTEXT_DIM]); nothing to train or load."""

    weights_loaded = True

    def __init__(self, modality_dropout=0.0, width=CONTEXT_DIM):
        super().__init__()
        self.width = width
        # The rows mix all modalities, so dropout removes a sample's whole row stream while
        # training, as if its stimulus were missing (TRIBEv2 drops each modality this way).
        self.modality_dropout = modality_dropout

    def forward(self, features):
        rows = features.get("context") if features else None
        if rows is None or rows.ndim != 3 or rows.shape[-1] != self.width:
            raise ValueError(f"Expected precomputed stimulus rows stimuli['context'] [B, T, {self.width}]")
        rows = rows.float()
        if self.training and self.modality_dropout > 0:
            keep = torch.rand(rows.shape[0], 1, 1, device=rows.device) >= self.modality_dropout
            rows = rows * keep
        return rows


class Standardize(nn.Module):
    """Each movie feature standardized with training-row statistics (set by the trainer); all-zero rows stay zero.
    With block, the condition is dim // block frames' rows of width block side by side (stimulus_context concat): one set
    of statistics serves every frame, and each frame's row is masked on its own, so a frame outside the recording stays zero."""

    def __init__(self, dim, block=None):
        super().__init__()
        self.block = block or dim
        if dim % self.block:
            raise ValueError("Standardize: dim must be a multiple of block")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, rows):
        if self.block == self.mean.shape[0]:
            return (rows - self.mean) / self.scale * (rows != 0).any(-1, keepdim=True)
        frames = rows.unflatten(-1, (-1, self.block))
        keep = (frames != 0).any(-1, keepdim=True)
        return ((frames - self.mean.view(-1, self.block)) / self.scale.view(-1, self.block) * keep).flatten(-2)

    @torch.no_grad()
    def fit(self, arrays):
        """Set the statistics from the rows (any row-major [n, block] arrays) that are not all zero."""
        total = torch.zeros(self.block, dtype=torch.float64)
        square, count = torch.zeros_like(total), 0
        for array in arrays:
            for start in range(0, len(array), 65536):
                part = torch.as_tensor(array[start:start + 65536], dtype=torch.float64)
                part = part[(part != 0).any(-1)]
                total, square, count = total + part.sum(0), square + (part ** 2).sum(0), count + len(part)
        mean = total / count
        copies = self.mean.shape[0] // self.block
        self.mean.copy_(mean.float().repeat(copies))
        self.scale.copy_((square / count - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6).float().repeat(copies))
