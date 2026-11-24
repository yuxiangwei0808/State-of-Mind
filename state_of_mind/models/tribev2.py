"""TRIBEv2 as released (d'Ascoli et al. 2026, arXiv 2605.04326), frozen.

Per-modality projections, the 8-block temporal Transformer, the low-rank head (2,048 values per
2-Hz sample) and, optionally, the released brain head: one "average subject" map to the 20,484
fsaverage5 vertices (the release keeps no per-subject heads). Predictions are for the fMRI
OFFSET_SECONDS after each stimulus sample.

Source: facebookresearch/tribev2, commit 38b3db073a7f, tribev2/model.py::FmriEncoderModel
(aggregate_features, transformer_forward, low_rank_head, predictor) with the released config.yaml
(hidden 1152, 8 blocks, 8 heads, low-rank head 2048, no combiner, bidirectional attention,
100-s segments). Adaptations: ``causal`` optionally masks attention to past and present samples;
the head's 2-Hz output is not pooled here (callers pool to their TR grid).
"""
from pathlib import Path

import torch
from torch import nn

MODALITY_DIMS = {"text": 3072, "audio": 1024, "video": 1408}  # TRIBEv2's feature order
HIDDEN, DEPTH, HEADS, CONTEXT_DIM, MAX_SEQ_LEN, N_VERTICES = 1152, 8, 8, 2048, 1024, 20484
SEGMENT_SAMPLES = 200  # TRIBEv2's 100-s inference and training segments
OFFSET_SECONDS = 5.0  # a sample at stimulus time t predicts the fMRI at t + 5 s


class TRIBEv2(nn.Module):
    def __init__(self, *, causal=False, brain_head=False, hidden=HIDDEN, depth=DEPTH, heads=HEADS,
                 context_dim=CONTEXT_DIM, max_seq_len=MAX_SEQ_LEN, n_vertices=N_VERTICES):
        super().__init__()
        from neuraltrain.models.transformer import TransformerEncoder  # TRIBEv2's dependency
        self.projectors = nn.ModuleDict({name: nn.Linear(2 * width, hidden // len(MODALITY_DIMS))
                                         for name, width in MODALITY_DIMS.items()})
        self.time_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden))
        self.encoder = TransformerEncoder(depth=depth, heads=heads, causal=causal).build(dim=hidden)
        self.low_rank_head = nn.Linear(hidden, context_dim, bias=False)
        self.head = nn.Linear(context_dim, n_vertices) if brain_head else None

    def load_tribev2(self, checkpoint_path):
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True, mmap=True)
        dims = checkpoint.get("model_build_args", {}).get("feature_dims", {})
        if list(dims) != list(MODALITY_DIMS) or any(tuple(dims[m]) != (2, w) for m, w in MODALITY_DIMS.items()):
            raise ValueError("TRIBEv2 checkpoint feature dimensions/order do not match text, audio, video")
        weights = checkpoint["state_dict"]
        state = {key[len("model."):]: value for key, value in weights.items()
                 if key.startswith("model.") and not key.startswith("model.predictor.")}
        if self.head is not None:
            # The released subject layer holds one map, [1, 2048, n_vertices]: the average subject.
            head, bias = weights["model.predictor.weights"], weights["model.predictor.bias"]
            if head.shape[0] != 1:
                raise ValueError(f"Expected the released average-subject head, found {head.shape[0]} subjects")
            state.update({"head.weight": head[0].T, "head.bias": bias[0]})
        self.load_state_dict(state, strict=True)
        return self

    def forward(self, features):
        """features: modality -> [B, T, 2, D] at 2 Hz (T <= max_seq_len); a missing modality is zero after
        projection, as in TRIBEv2. Returns the low-rank head's [B, T, context_dim]."""
        reference = next(iter(features.values()))
        parts = []
        for name, projector in self.projectors.items():
            if name in features:
                parts.append(projector(features[name].flatten(2).float()))
            else:
                parts.append(reference.new_zeros(*reference.shape[:2], projector.out_features, dtype=torch.float32))
        x = torch.cat(parts, dim=-1) + self.time_pos_embed[:, :reference.shape[1]]
        return self.low_rank_head(self.encoder(x))
