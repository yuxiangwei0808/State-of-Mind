"""Load a frozen world model checkpoint for inference and scoring."""
from state_of_mind.models import build_model
from state_of_mind.train.checkpoint import read_checkpoint, restore_model


def load_world_model(cfg, checkpoint_path, device):
    checkpoint = read_checkpoint(checkpoint_path)
    model = build_model(cfg, checkpoint.get("subjects"), initialize_stimulus=False)
    restore_model(model, cfg, checkpoint)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint
