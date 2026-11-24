"""The Transformer fast encoder's options: no LayerNorm before its output map, and a linear path beside it."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from state_of_mind.models.encoder import BrainEncoder, FrameTransformerEncoder, frame_encoder
from state_of_mind.utils.config import DEFAULTS, validate_config


def _brain(frames=9):
    return torch.randn(2, frames, 1000, generator=torch.Generator().manual_seed(0))


def test_default_encoder_normalizes_its_readout():
    encoder = FrameTransformerEncoder(5, 32, width=16, depth=1, heads=4, mlp_dim=32, dropout=0.0).eval()
    assert isinstance(encoder.norm, nn.LayerNorm) and encoder.linear is None
    assert encoder(_brain()).shape == (2, 5, 32)


def test_without_output_norm_the_readout_maps_straight_to_the_state():
    encoder = FrameTransformerEncoder(5, 32, width=16, depth=1, heads=4, mlp_dim=32, dropout=0.0, output_norm=False).eval()
    assert isinstance(encoder.norm, nn.Identity)
    assert not any(name.startswith("norm.") for name in encoder.state_dict())
    h = torch.randn(3, 6, 16)
    assert torch.allclose(encoder._read(h), encoder.projection(encoder.blocks(
        torch.cat((encoder.readout_token.expand(3, 1, -1), h), 1))[:, 0]))


def test_linear_path_starts_as_the_linear_encoder_alone():
    encoder = FrameTransformerEncoder(5, 32, width=16, depth=1, heads=4, mlp_dim=32, dropout=0.0, linear_path=True).eval()
    assert isinstance(encoder.linear, BrainEncoder)
    brain = _brain()
    assert torch.allclose(encoder(brain), encoder.linear(brain), atol=1e-6)
    assert torch.allclose(encoder.zero_point(), encoder.linear.zero_point(), atol=1e-6)
    with torch.no_grad():
        encoder.projection.weight.normal_()
    assert not torch.allclose(encoder(brain), encoder.linear(brain), atol=1e-3)


def test_config_passes_the_options_and_checks_them():
    m = {**DEFAULTS["model"], "encoder_kind": "transformer", "encoder_width": 16, "encoder_heads": 4, "encoder_mlp_dim": 32,
         "encoder_depth": 1, "encoder_output_norm": False, "encoder_linear_path": True}
    encoder = frame_encoder(m, 5, 32)
    assert isinstance(encoder.norm, nn.Identity) and isinstance(encoder.linear, BrainEncoder)
    cfg = deepcopy(DEFAULTS)
    cfg["model"].update(encoder_output_norm=False)
    with pytest.raises(ValueError, match="encoder_kind transformer"):
        validate_config(cfg)


def test_a_fixed_body_keeps_no_dropout_and_no_gradients_while_the_output_map_and_linear_path_train():
    encoder = FrameTransformerEncoder(5, 32, width=16, depth=1, heads=4, mlp_dim=32, dropout=0.5, linear_path=True)
    with torch.no_grad():
        encoder.projection.weight.normal_()
    brain = _brain()
    reference = deepcopy(encoder).eval()(brain).detach()
    encoder.fix_body().train()
    assert encoder.training and not encoder.blocks.training
    torch.testing.assert_close(encoder(brain), reference)  # dropout off in training: the cached outputs
    encoder.zero_point()
    assert not encoder.blocks.training  # zero_point's switch back to training leaves the body fixed
    torch.testing.assert_close(encoder(brain), reference)
    encoder(brain).square().sum().backward()
    trained = {name for name, p in encoder.named_parameters() if p.grad is not None}
    assert trained == {"projection.weight", "projection.bias", "linear.projection.weight", "linear.projection.bias"}

