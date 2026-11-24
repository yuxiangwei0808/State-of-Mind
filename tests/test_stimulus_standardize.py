import numpy as np
import pytest
import torch

from state_of_mind.models.stimulus import Standardize
from state_of_mind.utils.config import load_config


def test_standardize_blocks_share_statistics_and_mask_each_frame():
    rng = np.random.default_rng(0)
    rows = rng.normal(2.0, 3.0, size=(500, 4)).astype(np.float32)
    rows[:10] = 0  # rows outside the recording do not count
    scale = Standardize(12, block=4)
    scale.fit([rows])
    kept = rows[10:].astype(np.float64)
    assert torch.allclose(scale.mean, torch.tensor(kept.mean(0), dtype=torch.float32).repeat(3), atol=1e-5)
    assert torch.allclose(scale.scale, torch.tensor(kept.std(0), dtype=torch.float32).repeat(3), atol=1e-4)
    frames = torch.tensor(rows[10:13]).reshape(1, 12)  # three frames side by side
    frames[0, 4:8] = 0  # the middle frame is missing
    out = scale(frames).reshape(3, 4)
    assert torch.equal(out[1], torch.zeros(4))
    expected = (torch.tensor(rows[10]) - scale.mean[:4]) / scale.scale[:4]
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_standardize_without_blocks_is_unchanged():
    scale = Standardize(3)
    scale.fit([np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)])
    assert torch.allclose(scale.mean, torch.tensor([2.0, 2.0, 2.0]))
    out = scale(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(out[1], torch.zeros(3))


def test_config_accepts_concat_standardize_and_checks_zero_brain():
    config = "state_of_mind/configs/spectator/algonauts-jepa.yaml"
    settings = ("model.stimulus_features=tribev2_context", "model.stimulus_context=concat", "model.stimulus_context_trs=5",
                "model.stimulus_scaling=standardize")
    cfg = load_config(config, settings)
    assert cfg["model"]["stimulus_scaling"] == "standardize" and cfg["model"]["zero_brain"] is False
    with pytest.raises(ValueError, match="zero_brain"):
        load_config(config, settings + ("model.zero_brain=1",))
