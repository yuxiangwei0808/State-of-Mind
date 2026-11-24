"""Atomic checkpoints and epoch-boundary resume.

Only tensors and plain Python values are stored; loading uses weights_only=True.
World-model and fMRI-forecaster checkpoints use format neuroworld-latent-v2
(forecasters also store their subjects). Earlier formats are not supported.
"""
from pathlib import Path
import hashlib
import random
import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from state_of_mind.utils.config import DEFAULTS

FORMAT = "neuroworld-latent-v2"

MODEL_KEYS = ("state_window_trs", "context_states", "state_dim", "predictor_depth",
              "attention_heads", "predictor_dropout")


def read_checkpoint(path, *, expected=FORMAT):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != expected:
        raise ValueError(f"Expected a {expected} checkpoint at {path}")
    return checkpoint


def checkpoint_sha256(path):
    """Identify checkpoint contents independently of their filename."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_compatible(cfg, checkpoint, *, resume=False):
    source = checkpoint["config"]
    for section, keys in {"model": MODEL_KEYS, "data": ("tr_seconds", "hrf_delay_seconds"),
                          "train": ("mode",)}.items():
        for key in keys:
            if cfg[section][key] != source[section][key]:
                raise ValueError(f"Checkpoint mismatch: {section}.{key}")
    # Checkpoints predating these settings used one projected row per transition, each a 1-frame step.
    context_defaults = {"stimulus_context": "none", "stimulus_context_trs": 1, "stimulus_features": "projections",
                        "state_step": 1}
    for key, default in context_defaults.items():
        if cfg["model"].get(key, default) != source["model"].get(key, default):
            raise ValueError(f"Checkpoint mismatch: model.{key}")
    # Checkpoints predating train.objective are latent world models.
    if cfg["train"]["objective"] != source["train"].get("objective", "latent"):
        raise ValueError("Checkpoint mismatch: train.objective")
    # Checkpoints predating this setting used a feed-forward width of 4 * D.
    current_mlp = cfg["model"].get("predictor_mlp_dim", 4 * cfg["model"]["state_dim"])
    source_mlp = source["model"].get("predictor_mlp_dim", 4 * source["model"]["state_dim"])
    if current_mlp != source_mlp:
        raise ValueError("Checkpoint mismatch: model.predictor_mlp_dim")
    if resume:
        if checkpoint.get("world_size", 1) != PartialState().num_processes:
            raise ValueError("Resume requires the same number of training processes")
        if cfg["seed"] != source["seed"]:
            raise ValueError("Resume mismatch: seed")
        for section, keys in {
            "data": ("bundle_path", "train_runs", "val_runs", "heldout_subjects", "split_frames",
                     "sequence_states", "window_stride_trs", "batch_size"),
            "train": ("lr", "warmup_fraction", "schedule", "weight_decay", "grad_clip", "precision",
                      "sigreg_weight", "sigreg_knots", "sigreg_projections", "rollout_steps", "modality_dropout",
                      "rollout_weight", "freeze_encoder", "rollout_step_weights", "rollout_detach_every", "rollout_schedule",
                      "target", "target_momentum", "encoder_lr_factor", "freeze_fast_encoder", "freeze_encoder_body", "parcel_weight",
                      "latent_error", "l1_scale", "sigreg_ramp_epochs",
                      "teacher_run", "teacher_weight", "teacher_pair_weight"),
            "eval": ("prefix_trs", "horizon_trs", "window_stride_trs"),
        }.items():
            for key in keys:
                # Checkpoints predating a setting used its default.
                if cfg[section][key] != source[section].get(key, DEFAULTS[section][key]):
                    raise ValueError(f"Resume mismatch: {section}.{key}")
        if cfg["train"]["freeze_encoder"] != checkpoint["encoder_frozen"]:
            raise ValueError("Resume must preserve whether the encoder was frozen")


def restore_model(model, cfg, checkpoint, *, resume=False):
    check_compatible(cfg, checkpoint, resume=resume)
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.stimulus is not None:
        model.stimulus.weights_loaded = True
    if checkpoint["encoder_frozen"]:
        model.freeze_encoder()


def rng_state():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:])}


def write_atomic(path, payload, accelerator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    accelerator.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, cfg, *, accelerator, epoch, global_step, total_steps,
                    best_score, history, data_summary, wandb_run=None):
    rng_by_rank = gather_object([rng_state()])
    if not accelerator.is_main_process:
        return
    write_atomic(path, {
        "format": FORMAT, "phase": model.mode, "config": cfg, "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "global_step": global_step, "total_steps": total_steps,
        "best_score": best_score, "history": history, "encoder_frozen": model.encoder_frozen,
        "data_summary": data_summary, "world_size": accelerator.num_processes, "wandb_run": wandb_run,
        "subjects": list(model.readout.subjects) if hasattr(model, "readout") else None,
        "rng": rng_by_rank[0], "rng_by_rank": rng_by_rank}, accelerator)


def restore_rng(rng):
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])
    name, keys, position, has_gauss, cached = rng["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


def restore_optimizer_and_rng(optimizer, scheduler, checkpoint, device):
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(checkpoint["scheduler"])
    states = checkpoint.get("rng_by_rank", [checkpoint["rng"]])
    restore_rng(states[PartialState().process_index])
