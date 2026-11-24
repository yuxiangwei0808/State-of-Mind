"""Spectator, the reported model, is frozen (state_of_mind/spectator/__init__.py): its configs keep their settings, and with the
frozen weights the current code must reproduce the stored reference rollout. A failure here means a code change altered
the reported model."""
from pathlib import Path

import pytest
import torch

from state_of_mind.utils.config import load_config
from state_of_mind.paths import OUTPUT

CONFIGS = Path(__file__).resolve().parents[1] / "state_of_mind" / "configs" / "spectator"
RELEASE = OUTPUT / "spectator-v1"


@pytest.mark.parametrize("name, window", [("algonauts", 120), ("hcp", 180)])
def test_final_configs_describe_spectator(name, window):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    m, t = cfg["model"], cfg["train"]
    assert cfg["design"] == "transformer_world_model" and t["freeze_encoder"]
    assert (m["encoder_kind"], m["fast_window_trs"], m["state_window_trs"], m["state_dim"]) == ("linear", 5, window, 2048)
    assert m["slow_per_dimension"] and m["state_tokens"] == 2
    assert (m["predictor_kind"], m["mlp_depth"]) == ("mlp", 0)  # the predictor is one matrix
    assert (m["stimulus_features"], m["stimulus_context"], m["stimulus_context_trs"], m["stimulus_scaling"]) == \
        ("tribev2_context", "concat", 5, "standardize")
    assert cfg["data"]["hrf_delay_seconds"] == 0.0 and not m["zero_brain"]
    assert t["encoder_checkpoint"].startswith(str(RELEASE)) and t["predictor_checkpoint"].startswith(str(RELEASE))


def test_training_configs_describe_spectator():
    stage0 = load_config(CONFIGS / "algonauts-stage0.yaml")
    assert (stage0["design"], stage0["model"]["encoder_kind"], stage0["model"]["state_window_trs"]) == ("stage0_forecaster", "linear", 5)
    assert stage0["train"]["patience"] is None and stage0["train"]["epochs"] == 100
    jepa = load_config(CONFIGS / "algonauts-jepa.yaml")
    t = jepa["train"]
    assert jepa["design"] == "transformer_world_model_finetuned" and t["target"] == "ema" and t["target_momentum"] == 0.998
    assert (t["lr"], t["encoder_lr_factor"], t["rollout_steps"], t["modality_dropout"], t["patience"]) == (5e-5, 0.2, 40, 0.3, 10)
    assert jepa["model"]["state_window_trs"] == 120 and jepa["data"]["hrf_delay_seconds"] == 3.0


@pytest.mark.skipif(not RELEASE.is_dir(), reason="the frozen release is not on this machine")
@pytest.mark.parametrize("dataset", ["algonauts", "hcp"])
def test_frozen_model_reproduces_its_reference_rollout(dataset):
    from state_of_mind.train.readout import load_world_model
    folder = RELEASE / dataset / "model"
    cfg = load_config(folder / "config.yaml")
    model, _ = load_world_model(cfg, folder / "best.pt", "cpu")
    reference = torch.load(folder / "reference-rollout.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        states = model.rollout(reference["brain"], reference["stimuli"], horizon_trs=reference["states"].shape[1])
    torch.testing.assert_close(states, reference["states"], rtol=1e-5, atol=1e-5)
    lags = model.encoder.lag_weights.shape
    assert tuple(lags) == (1024, cfg["model"]["state_window_trs"] - 4)
    assert tuple(model.predictor.linear.weight.shape) == (2048, 2048)
    assert tuple(model.predictor.condition.weight.shape) == (2048, 5 * 2048)
