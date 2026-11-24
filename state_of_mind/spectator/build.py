"""Files between Spectator's training steps: the JEPA phase's encoder in the format later steps load, and the model's run
directory (encoder + closed-form matrix, no training)."""
from pathlib import Path

import torch
import yaml

from state_of_mind.models import build_model
from state_of_mind.train.checkpoint import FORMAT
from state_of_mind.train.stage_utils import build_dataset, load_encoder_init


def encoder_file(run_best, start_encoder, out):
    """The trained (online) encoder of a JEPA-phase run, written over the encoder file that run started from, which keeps
    that file's config. Returns the names of the weights that changed."""
    source = torch.load(start_encoder, map_location="cpu", weights_only=False)
    trained = torch.load(run_best, map_location="cpu", weights_only=False)["model"]
    changed = []
    for k in [k for k in source["model"] if k.startswith("encoder.")]:
        assert source["model"][k].shape == trained[k].shape, k
        if not torch.equal(source["model"][k], trained[k]):
            changed.append(k)
        source["model"][k] = trained[k].clone()
    torch.save(source, out)
    return changed


def assemble(cfg, matrix, run_dir):
    """A run directory (best.pt, config.yaml) for the model built from cfg: its encoder from train.encoder_checkpoint, its
    predictor (and movie-row statistics) from the closed-form matrix file."""
    model = build_model(cfg, build_dataset(cfg, "val").subject_ids, initialize_stimulus=True)
    load_encoder_init(model, cfg)
    state = model.state_dict()
    for k, v in torch.load(matrix, map_location="cpu", weights_only=False)["model"].items():
        assert state[k].shape == v.shape, k
        state[k] = v.clone()
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "phase": cfg["train"]["mode"], "config": cfg, "model": state, "optimizer": {}, "scheduler": {},
                "epoch": 0, "history": [], "encoder_frozen": True, "subjects": None}, out / "best.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    return out / "best.pt"
