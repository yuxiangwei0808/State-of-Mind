"""YAML settings: Spectator configs on top of a shared per-dataset base.

A config YAML (e.g. configs/spectator/algonauts-jepa.yaml) declares ``design`` and either
names a ``base`` or holds a fully resolved run configuration.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import re
import yaml

from state_of_mind.paths import expand

DEFAULTS = {
    "design": None,
    "seed": 0,
    "wandb": {"project": "state_of_mind", "entity": None, "name": None, "mode": "online"},
    "data": {"bundle_path": None, "tr_seconds": 1.49, "hrf_delay_seconds": 5.0,
             "sequence_states": 4, "window_stride_trs": 1, "train_runs": [], "val_runs": [],
             "dev_runs": [], "test_runs": [], "heldout_subjects": [], "split_frames": {},
             "batch_size": 128, "num_workers": 0, "preload": False, "context_features": None,
             "extra_movie": None, "extra_rest": [], "rest_store": None, "extra_windows": 0, "extra_sd": 0.61, "rest_only": False, "epoch_steps": 0,
             "history_fill": False},
    "model": {"state_window_trs": 3, "context_states": 3, "state_dim": 1024,
              "predictor_depth": 6, "attention_heads": 16, "predictor_mlp_dim": 2048,
              "predictor_dropout": 0.1, "history_blocks": 0, "history_block_states": 10,
              "history_measured_only": False, "history_spaced": False, "zero_brain": False,
              "stimulus_context": "none", "stimulus_context_trs": 1, "context_width": 2048,
              "stimulus_features": "projections", "tribev2_checkpoint": None, "stimulus_scaling": "layernorm",
              "conditioning": "adaln", "state_step": 1, "chunk_frames": 0,
              "state_tokens": 1, "fast_window_trs": 0, "own_drive": False, "movie_rank": 0,
              "measurement_token": False, "encoder_hidden_dim": 0, "slow_per_dimension": False,
              "predictor_kind": "transformer", "mlp_hidden_dim": 2048, "mlp_depth": 3,
              "predictor_output_norm": True, "movie_tokens": False, "token_positions_file": None,
              "predictor_width": 0, "slow_extra_branches": 0, "encoder_kind": "linear", "encoder_width": 512, "encoder_depth": 2,
              "encoder_heads": 8, "encoder_mlp_dim": 1024, "encoder_dropout": 0.1, "encoder_output_norm": True,
              "encoder_linear_path": False},
    "train": {"objective": "latent", "mode": "stimulus", "encoder_checkpoint": None, "freeze_encoder": False,
              "epochs": 1, "lr": 5e-5, "optimizer": "adamw", "warmup_fraction": 0.01, "schedule": "cosine",
              "weight_decay": 1e-3, "grad_clip": 1.0, "precision": "fp32", "accumulation_steps": 1,
              "sigreg_weight": 0.09, "sigreg_knots": 17, "sigreg_projections": 1024, "sigreg_scale_file": None, "sigreg_ramp_epochs": 0,
              "latent_error": "squared", "l1_scale": 1.0,
              "rollout_steps": 0, "rollout_weight": 1.0, "prediction_weight": 1.0, "modality_dropout": 0.0,
              "target": "encoder", "target_momentum": 0.998, "encoder_lr_factor": 1.0, "freeze_fast_encoder": False, "freeze_encoder_body": False,
              "rollout_step_weights": "uniform", "rollout_detach_every": 0, "rollout_schedule": None,
              "patience": None, "parcel_weight": 0.0, "predictor_checkpoint": None, "predictor_checkpoint_movie": False,
              "slow_loss_weight": 1.0, "reveal_readout": None,
              "teacher_run": None, "teacher_weight": 0.0, "teacher_pair_weight": 0.0,
              "masked_target": "latent", "mask_blocks": 2, "mask_lengths": [3, 6], "mask_suffix_probability": 0.5, "masked_predictor_depth": 2,
              "masked_target_norm": False,
              "device": "auto", "log_every_steps": 100, "output_dir": None, "run_tag": None},
    "eval": {"prefix_trs": 1, "horizon_trs": 20, "window_stride_trs": 20,
             "probe_windows": 4096, "probe_ridge": 0.1},
    "readout": {"hidden_dim": 1024, "epochs": 20, "lr": 1e-5, "warmup_steps": 500,
                "schedule": "cosine", "weight_decay": 1e-3, "batch_size": 16,
                "train_horizon_trs": 20, "window_stride_trs": 20, "train_on": "rollout",
                "cache_rollouts": True, "select_by": "parcel_pearson", "patience": None},
}


# The settings each design fixes. Design YAMLs may not set or override them.
DESIGNS = {
    "fmri_forecaster": {"objective": "fmri", "mode": "stimulus", "freeze_encoder": False},
    "joint_world_model": {"objective": "latent", "mode": "stimulus", "freeze_encoder": False},  # world model, own encoder
    "stage0_forecaster": {"objective": "fmri", "mode": "brain", "freeze_encoder": False},  # brain-only forecaster, for its encoder
    "jepa_stage0": {"objective": "latent", "mode": "brain", "freeze_encoder": False},  # a stage 0 with a JEPA objective
    "transformer_world_model": {"objective": "latent", "mode": "stimulus", "freeze_encoder": True},  # world model, frozen stage-0 encoder
    "transformer_world_model_finetuned": {"objective": "latent", "mode": "stimulus", "freeze_encoder": False},  # its stage-0 encoder trained further
}
# Older names, as saved in the configurations of existing runs.
OLD_DESIGNS = {"A": "fmri_forecaster", "B": "joint_world_model", "C0": "stage0_forecaster", "L0": "jepa_stage0", "C": "transformer_world_model", "Cft": "transformer_world_model_finetuned"}
# Settings of removed features (added fMRI data, the measurement token, reveals, Muon, block SIGReg, the masked stage 0). Saved run
# configurations hold them, so they stay accepted, at their defaults only.
RETIRED = {"data": ("extra_movie", "extra_rest", "rest_store", "extra_windows", "extra_sd", "rest_only", "epoch_steps"),
           "model": ("measurement_token",),
           "train": ("optimizer", "reveal_readout", "sigreg_scale_file", "masked_target", "mask_blocks", "mask_lengths",
                     "mask_suffix_probability", "masked_predictor_depth", "masked_target_norm")}
# The designs whose encoder starts from a stage-0 run.
PHASE0_DESIGNS = ("transformer_world_model", "transformer_world_model_finetuned")
LOCKED = ("objective", "mode", "encoder_checkpoint", "freeze_encoder")
# Settings the Transformer world model's frozen encoder shares with its stage-0 run; overriding one for the Transformer world model
# looks up the stage-0 run trained with the same override.
ENCODER_SHAPE = ("model.state_window_trs", "model.state_dim")


def _check_design(cfg):
    design = cfg["design"] = OLD_DESIGNS.get(cfg["design"], cfg["design"])
    if design is None:
        return
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {sorted(DESIGNS)}")
    for key, value in DESIGNS[design].items():
        if cfg["train"][key] != value:
            raise ValueError(f"train.{key} is fixed by design {design} (expected {value!r})")
    has_encoder = bool(cfg["train"]["encoder_checkpoint"])
    if has_encoder != (design in PHASE0_DESIGNS):
        raise ValueError(f"train.encoder_checkpoint is fixed by design {design}: "
                         + ("the stage-0 encoder is required" if design in PHASE0_DESIGNS else "it must be null"))


def _positive_int(cfg, section, key):
    value = cfg[section][key]
    if type(value) is not int or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer")


def _finite(cfg, section, key, *, positive):
    value = cfg[section][key]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{section}.{key} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{section}.{key} is out of range")


def validate_config(cfg):
    _check_design(cfg)
    for section, keys in RETIRED.items():
        for key in keys:
            if cfg[section][key] != DEFAULTS[section][key]:
                raise ValueError(f"{section}.{key} must be {json.dumps(DEFAULTS[section][key])}: the feature it set was removed")
    if cfg["wandb"]["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    for key in ("project", "entity", "name"):
        value = cfg["wandb"][key]
        if value is None and key != "project":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"wandb.{key} must be a non-empty string")
    for section, keys in {
        "data": ("sequence_states", "window_stride_trs", "batch_size"),
        "model": ("state_window_trs", "context_states", "state_dim", "predictor_depth",
                  "attention_heads", "predictor_mlp_dim", "stimulus_context_trs", "state_step"),
        "train": ("epochs", "sigreg_knots", "sigreg_projections", "log_every_steps", "accumulation_steps"),
        "eval": ("prefix_trs", "horizon_trs", "window_stride_trs", "probe_windows"),
        "readout": ("epochs", "batch_size", "train_horizon_trs", "window_stride_trs"),
    }.items():
        for key in keys:
            _positive_int(cfg, section, key)
    for section in ("train", "readout"):  # early stopping, in epochs without a better selection score
        if cfg[section]["patience"] is not None:
            _positive_int(cfg, section, "patience")
    for section, key in (("train", "rollout_steps"), ("readout", "warmup_steps"), ("readout", "hidden_dim")):
        value = cfg[section][key]
        if type(value) is not int or value < 0:
            raise ValueError(f"{section}.{key} must be a non-negative integer")
    dropout = cfg["train"]["modality_dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("train.modality_dropout must be in [0, 1)")
    fraction = cfg["train"]["warmup_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 <= fraction < 1:
        raise ValueError("train.warmup_fraction must be in [0, 1)")
    if cfg["train"]["sigreg_knots"] < 2:
        raise ValueError("train.sigreg_knots must be at least 2")
    if cfg["data"]["sequence_states"] < 2:
        raise ValueError("data.sequence_states must be at least 2")
    for section, key, positive in [("data", "tr_seconds", True), ("data", "hrf_delay_seconds", False),
                                   ("train", "lr", True), ("train", "weight_decay", False),
                                   ("train", "grad_clip", False), ("train", "sigreg_weight", False),
                                   ("train", "rollout_weight", False), ("train", "prediction_weight", False),
                                   ("model", "predictor_dropout", False),
                                   ("readout", "lr", True), ("readout", "weight_decay", False),
                                   ("eval", "probe_ridge", False)]:
        _finite(cfg, section, key, positive=positive)
    if cfg["model"]["predictor_dropout"] >= 1:
        raise ValueError("model.predictor_dropout must be below 1")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(cfg["data"]["preload"], bool):
        raise ValueError("data.preload must be true or false")
    if type(cfg["data"]["num_workers"]) is not int or cfg["data"]["num_workers"] < 0:
        raise ValueError("data.num_workers must be a non-negative integer")
    if cfg["model"]["state_dim"] % cfg["model"]["attention_heads"]:
        raise ValueError("state_dim must be divisible by attention_heads")
    if cfg["model"]["stimulus_context"] not in {"none", "concat", "linear", "concat-audio"}:
        raise ValueError("model.stimulus_context must be none, concat, linear or concat-audio")
    if cfg["model"]["stimulus_context"] == "none" and cfg["model"]["stimulus_context_trs"] != 1:
        raise ValueError("model.stimulus_context: none uses one row; choose concat or linear for more")
    if cfg["model"]["stimulus_scaling"] not in {"layernorm", "standardize"}:
        raise ValueError("model.stimulus_scaling must be layernorm or standardize")
    if cfg["model"]["stimulus_scaling"] == "standardize" and (cfg["model"]["stimulus_features"] != "tribev2_context"
                                                             or cfg["model"]["stimulus_context"] not in {"none", "concat"}):
        raise ValueError("model.stimulus_scaling standardize applies to tribev2_context rows, one per transition or several concatenated")
    if not isinstance(cfg["model"]["zero_brain"], bool):
        raise ValueError("model.zero_brain must be true or false")
    if cfg["model"]["conditioning"] not in {"adaln", "input", "output"}:
        raise ValueError("model.conditioning must be adaln, input or output")
    if cfg["train"]["prediction_weight"] == 0 and not cfg["train"]["rollout_steps"]:
        raise ValueError("train.prediction_weight 0 needs a rollout loss (train.rollout_steps)")
    t = cfg["train"]
    if t["target"] not in {"encoder", "initial", "ema"}:
        raise ValueError("train.target must be encoder, initial or ema")
    if t["target"] != "encoder" and (t["objective"] != "latent" or t["freeze_encoder"]):
        raise ValueError("train.target initial or ema scores a latent world model whose encoder trains "
                         "(train.objective latent, train.freeze_encoder false)")
    parcel = t["parcel_weight"]
    if isinstance(parcel, bool) or not isinstance(parcel, (float, int)) or not math.isfinite(parcel) or parcel < 0:
        raise ValueError("train.parcel_weight must be a non-negative number")
    if parcel and (t["objective"] != "latent" or cfg["model"]["state_step"] != 1):
        raise ValueError("train.parcel_weight adds a parcel head to a latent world model with state_step 1")
    if t["latent_error"] not in {"squared", "l1"}:
        raise ValueError("train.latent_error must be squared or l1")
    scale = t["l1_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or scale <= 0:
        raise ValueError("train.l1_scale must be positive")
    ramp = t["sigreg_ramp_epochs"]
    if type(ramp) is not int or ramp < 0:
        raise ValueError("train.sigreg_ramp_epochs must be a non-negative integer")
    momentum = t["target_momentum"]
    if isinstance(momentum, bool) or not isinstance(momentum, (float, int)) or not 0 <= momentum < 1:
        raise ValueError("train.target_momentum must be in [0, 1)")
    factor = t["encoder_lr_factor"]
    if isinstance(factor, bool) or not isinstance(factor, (float, int)) or not 0 < factor <= 1:
        raise ValueError("train.encoder_lr_factor must be in (0, 1]")
    if factor != 1 and t["freeze_encoder"]:
        raise ValueError("train.encoder_lr_factor scales a trainable encoder's rate (train.freeze_encoder false)")
    if t["rollout_step_weights"] not in {"uniform", "ramp_up", "ramp_down"}:
        raise ValueError("train.rollout_step_weights must be uniform, ramp_up or ramp_down")
    if type(t["rollout_detach_every"]) is not int or t["rollout_detach_every"] < 0:
        raise ValueError("train.rollout_detach_every must be a non-negative integer (0 backpropagates the whole rollout)")
    schedule = t["rollout_schedule"]
    if schedule is not None:
        # [[epoch, steps], ...]: from each listed epoch on, the rollout covers that many steps.
        pairs = schedule if isinstance(schedule, list) else []
        if (not pairs or any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p) for p in pairs)
                or pairs[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(pairs, pairs[1:]))
                or any(not 1 <= p[1] <= t["rollout_steps"] for p in pairs)):
            raise ValueError("train.rollout_schedule lists [epoch, steps] pairs from epoch 0 in increasing epochs, "
                             "with 1 <= steps <= train.rollout_steps")
    if (t["rollout_step_weights"] != "uniform" or t["rollout_detach_every"] or schedule is not None) and (
            t["objective"] != "latent" or not t["rollout_steps"]):
        raise ValueError("train.rollout_step_weights, rollout_detach_every and rollout_schedule shape the latent rollout "
                         "loss (train.objective latent with train.rollout_steps)")
    weight = t["slow_loss_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (float, int)) or weight <= 0:
        raise ValueError("train.slow_loss_weight must be positive")
    if t["freeze_fast_encoder"] and (t["freeze_encoder"] or not cfg["model"]["fast_window_trs"]):
        raise ValueError("train.freeze_fast_encoder trains only a slow latent's lag weights (model.fast_window_trs, train.freeze_encoder false)")
    if t["freeze_encoder_body"] and (t["freeze_encoder"] or t["freeze_fast_encoder"] or not cfg["model"]["fast_window_trs"]
                                     or cfg["model"]["encoder_kind"] != "transformer"):
        raise ValueError("train.freeze_encoder_body trains a Transformer fast encoder's output map and linear path and the lag weights "
                         "(model.encoder_kind transformer with model.fast_window_trs, train.freeze_encoder and freeze_fast_encoder false)")
    if weight != 1.0 and (t["objective"] != "latent" or not cfg["model"]["fast_window_trs"]):
        raise ValueError("train.slow_loss_weight scales a slow latent's loss (train.objective latent with model.fast_window_trs)")
    for key in ("teacher_weight", "teacher_pair_weight"):
        if isinstance(t[key], bool) or not isinstance(t[key], (float, int)) or t[key] < 0:
            raise ValueError(f"train.{key} must be a number of at least 0")
    if bool(t["teacher_weight"] or t["teacher_pair_weight"]) != bool(t["teacher_run"]):
        raise ValueError("train.teacher_run and a positive train.teacher_weight or teacher_pair_weight go together")
    if t["teacher_run"] and (t["objective"] != "latent" or t["mode"] != "stimulus" or not t["rollout_steps"] or t["rollout_schedule"]
                             or cfg["model"]["context_states"] != 1 or cfg["model"]["state_step"] != 1 or cfg["model"]["history_blocks"]
                             or cfg["model"]["chunk_frames"]):
        raise ValueError("train.teacher_run trains against a frozen world model's own predictions in a one-state rollout (train.objective "
                         "latent, mode stimulus, train.rollout_steps without a schedule; model.context_states 1, state_step 1, no history "
                         "blocks or chunks)")
    if cfg["model"]["stimulus_features"] not in {"projections", "tribev2_context"}:
        raise ValueError("model.stimulus_features must be projections or tribev2_context")
    if cfg["model"]["stimulus_features"] == "tribev2_context" and cfg["train"]["mode"] == "stimulus":
        path = cfg["data"]["context_features"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("model.stimulus_features: tribev2_context needs data.context_features "
                             "(prepared by state_of_mind.nbs_data.stimuli.tribev2_context)")
    if cfg["train"]["objective"] not in {"latent", "fmri"}:
        raise ValueError("train.objective must be latent or fmri")
    m = cfg["model"]
    if m["encoder_kind"] not in {"linear", "transformer"}:
        raise ValueError("model.encoder_kind must be linear or transformer")
    if m["encoder_kind"] == "transformer" and (m["encoder_hidden_dim"] or m["encoder_width"] % m["encoder_heads"]):
        raise ValueError("model.encoder_kind transformer takes no model.encoder_hidden_dim, and model.encoder_heads divides model.encoder_width")
    if m["encoder_kind"] != "transformer" and (not m["encoder_output_norm"] or m["encoder_linear_path"]):
        raise ValueError("model.encoder_output_norm and model.encoder_linear_path set a Transformer encoder: they need model.encoder_kind transformer")
    if cfg["train"]["mode"] not in {"brain", "stimulus"}:
        raise ValueError("train.mode must be brain or stimulus")
    for section in ("train", "readout"):
        if cfg[section]["schedule"] not in {"cosine", "constant"}:
            raise ValueError(f"{section}.schedule must be cosine or constant")
    if cfg["train"]["precision"] not in {"fp32", "bf16"}:
        raise ValueError("train.precision must be fp32 or bf16")
    if not isinstance(cfg["train"]["freeze_encoder"], bool):
        raise ValueError("train.freeze_encoder must be true or false")
    if cfg["train"]["freeze_encoder"] and not cfg["train"]["encoder_checkpoint"]:
        raise ValueError("freeze_encoder requires train.encoder_checkpoint")
    for key in ("history_blocks", "history_block_states"):
        value = cfg["model"][key]
        if type(value) is not int or value < 0:
            raise ValueError(f"model.{key} must be a non-negative integer")
    if not isinstance(cfg["model"]["history_measured_only"], bool):
        raise ValueError("model.history_measured_only must be true or false")
    if cfg["model"]["history_measured_only"] and not cfg["model"]["history_blocks"]:
        raise ValueError("model.history_measured_only applies to history blocks")
    if not isinstance(cfg["model"]["history_spaced"], bool):
        raise ValueError("model.history_spaced must be true or false")
    if cfg["model"]["history_spaced"] and not cfg["model"]["history_blocks"]:
        raise ValueError("model.history_spaced applies to history blocks")
    read = cfg["model"]["context_states"] + cfg["model"]["history_blocks"] * cfg["model"]["history_block_states"]
    step = cfg["model"]["state_step"]
    # With a state step s, a training window of S states holds s chains of S / s states s frames apart.
    chain = cfg["data"]["sequence_states"] // step
    if step > 1:
        if cfg["model"]["history_blocks"]:
            raise ValueError("model.state_step above 1 does not take history blocks")
        if cfg["data"]["sequence_states"] % step:
            raise ValueError("data.sequence_states must be a multiple of model.state_step")
        if cfg["eval"]["prefix_trs"] < step:
            raise ValueError("eval.prefix_trs must give each of the model.state_step chains a state")
    for key in ("state_tokens", "fast_window_trs", "slow_extra_branches", "predictor_width"):
        if type(cfg["model"][key]) is not int or cfg["model"][key] < 0:
            raise ValueError(f"model.{key} must be a non-negative integer")
    if cfg["model"]["state_tokens"] < 1:
        raise ValueError("model.state_tokens must be at least 1")
    if cfg["model"]["state_tokens"] > 1:
        if read != 1 or step > 1 or cfg["model"]["chunk_frames"]:
            raise ValueError("model.state_tokens reads one state, with no state step or chunks")
        if cfg["model"]["state_dim"] % cfg["model"]["state_tokens"]:
            raise ValueError("model.state_dim must split evenly into model.state_tokens")
    if cfg["model"]["fast_window_trs"]:
        if cfg["model"]["state_tokens"] != 2 + cfg["model"]["slow_extra_branches"] or cfg["model"]["fast_window_trs"] > cfg["model"]["state_window_trs"]:
            raise ValueError("model.fast_window_trs needs two state tokens (one more per model.slow_extra_branches) and fits in model.state_window_trs")
        if not cfg["train"]["freeze_encoder"] and not cfg["train"]["encoder_checkpoint"]:
            raise ValueError("model.fast_window_trs takes an encoder from a checkpoint, frozen or trained on")
    for key in ("movie_rank", "encoder_hidden_dim"):
        if type(cfg["model"][key]) is not int or cfg["model"][key] < 0:
            raise ValueError(f"model.{key} must be a non-negative integer")
    if cfg["model"]["movie_rank"] and cfg["model"]["conditioning"] == "adaln":
        raise ValueError("model.movie_rank applies to an additive movie (model.conditioning input or output)")
    if not isinstance(cfg["model"]["own_drive"], bool) or (cfg["model"]["own_drive"] and (
            cfg["model"]["state_tokens"] < 2 or cfg["model"]["conditioning"] == "adaln")):
        raise ValueError("model.own_drive needs state tokens and an additive movie")
    if not isinstance(cfg["model"]["slow_per_dimension"], bool) or (cfg["model"]["slow_per_dimension"] and not cfg["model"]["fast_window_trs"]):
        raise ValueError("model.slow_per_dimension needs a slow latent (model.fast_window_trs)")
    if cfg["model"]["predictor_kind"] not in {"transformer", "mlp", "corrected"}:
        raise ValueError("model.predictor_kind must be transformer, mlp or corrected")
    if cfg["model"]["predictor_kind"] == "corrected":
        if read != 1 or step > 1 or cfg["model"]["chunk_frames"] or cfg["train"]["mode"] != "stimulus":
            raise ValueError("model.predictor_kind corrected reads one state and the movie, with no state step or chunks")
        if not cfg["train"]["predictor_checkpoint"] or not cfg["train"]["predictor_checkpoint_movie"]:
            raise ValueError("model.predictor_kind corrected takes its matrix from train.predictor_checkpoint, with train.predictor_checkpoint_movie")
    if cfg["model"]["slow_extra_branches"] and not cfg["model"]["fast_window_trs"]:
        raise ValueError("model.slow_extra_branches adds slow latents to a fast one (model.fast_window_trs)")
    if cfg["model"]["predictor_width"] and (cfg["model"]["predictor_kind"] == "mlp" or cfg["model"]["conditioning"] != "input" or read != 1
                                             or cfg["model"]["chunk_frames"] or cfg["model"]["movie_tokens"]):
        raise ValueError("model.predictor_width narrows a Transformer with input conditioning reading one state")
    if not isinstance(cfg["model"]["predictor_output_norm"], bool):
        raise ValueError("model.predictor_output_norm must be true or false")
    if not isinstance(cfg["model"]["movie_tokens"], bool):
        raise ValueError("model.movie_tokens must be true or false")
    if cfg["model"]["movie_tokens"] and (cfg["model"]["stimulus_context"] != "concat" or cfg["model"]["stimulus_context_trs"] < 2 or cfg["model"]["conditioning"] != "input"
                                         or read != 1 or step > 1 or cfg["model"]["chunk_frames"] or cfg["model"]["own_drive"] or cfg["model"]["movie_rank"]
                                         or cfg["model"]["predictor_kind"] != "transformer" or cfg["train"]["mode"] != "stimulus"):
        raise ValueError("model.movie_tokens splits several concatenated rows, on a Transformer with input conditioning reading one state")
    if cfg["model"]["token_positions_file"] and (cfg["model"]["state_tokens"] < 2 or cfg["model"]["predictor_kind"] != "transformer"):
        raise ValueError("model.token_positions_file gives state tokens their positions, on a Transformer")
    if cfg["model"]["predictor_kind"] == "mlp":
        if read != 1 or step > 1 or cfg["model"]["chunk_frames"] or cfg["model"]["own_drive"] or cfg["model"]["movie_rank"]:
            raise ValueError("model.predictor_kind mlp reads one state, with no state step, chunks, own drives or movie rank")
        if cfg["train"]["mode"] != "stimulus":
            raise ValueError("model.predictor_kind mlp needs train.mode stimulus")
        for key in ("mlp_hidden_dim", "mlp_depth"):
            if type(cfg["model"][key]) is not int or cfg["model"][key] < 0:
                raise ValueError(f"model.{key} must be a non-negative integer")
    chunk = cfg["model"]["chunk_frames"]
    if type(chunk) is not int or chunk < 0:
        raise ValueError("model.chunk_frames must be a non-negative integer")
    if chunk:
        if step > 1:
            raise ValueError("model.chunk_frames does not take a state step")
        if cfg["model"]["history_measured_only"]:
            raise ValueError("model.chunk_frames feeds its predictions back, so it does not take history_measured_only")
        if cfg["model"]["history_blocks"] and not cfg["model"]["history_spaced"]:
            raise ValueError("model.chunk_frames reads single states, so history blocks need history_spaced")
        if cfg["train"]["objective"] != "latent" or cfg["train"]["rollout_detach_every"]:
            raise ValueError("model.chunk_frames needs train.objective latent and no train.rollout_detach_every")
        if cfg["train"]["rollout_steps"] % chunk or (cfg["data"]["sequence_states"] - read) % chunk:
            raise ValueError("train.rollout_steps and the states after those read must be whole chunks")
        if cfg["eval"]["prefix_trs"] < read:
            raise ValueError("eval.prefix_trs must cover the states the predictor reads")
    if cfg["model"]["history_blocks"] and chain <= read:
        raise ValueError("data.sequence_states must exceed the states the predictor reads")
    if cfg["train"]["rollout_steps"] and chain < read + cfg["train"]["rollout_steps"]:
        raise ValueError("data.sequence_states must cover the states the predictor reads + rollout_steps"
                         + (" in each chain" if step > 1 else ""))
    if cfg["readout"]["train_on"] not in {"rollout", "encoded"}:
        raise ValueError("readout.train_on must be rollout or encoded")
    if not isinstance(cfg["readout"]["cache_rollouts"], bool):
        raise ValueError("readout.cache_rollouts must be true or false")
    if cfg["readout"]["select_by"] not in {"fmri_mse", "parcel_pearson"}:
        raise ValueError("readout.select_by must be fmri_mse or parcel_pearson")
    output_dir = cfg["train"]["output_dir"]
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        raise ValueError("train.output_dir must be a path, or null for an automatic run directory")
    tag = cfg["train"]["run_tag"]
    if tag is not None and (not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", tag)):
        raise ValueError("train.run_tag may contain only letters, digits, '.', '_' and '-'")
    if not cfg["data"]["bundle_path"]:
        raise ValueError("data.bundle_path is required")
    for key in ("train_runs", "val_runs", "dev_runs", "test_runs"):
        patterns = cfg["data"][key]
        required = key in ("train_runs", "val_runs")
        if not isinstance(patterns, list) or (required and not patterns) or \
                not all(isinstance(p, str) and p for p in patterns):
            raise ValueError(f"data.{key} must " + ("contain" if required else "be a list of")
                             + " explicit run IDs or glob patterns")
    d = cfg["data"]
    if not isinstance(d["history_fill"], bool):
        raise ValueError("data.history_fill must be true or false")
    if d["history_fill"] and not cfg["model"].get("fast_window_trs"):
        raise ValueError("data.history_fill needs a long-history encoder (model.fast_window_trs)")
    subjects = cfg["data"]["heldout_subjects"]
    if not isinstance(subjects, list) or not all(isinstance(s, str) and s for s in subjects):
        raise ValueError("data.heldout_subjects must be a list of subject IDs")
    frames = cfg["data"]["split_frames"]
    if not isinstance(frames, dict) or set(frames) - {"train", "val", "dev", "test"}:
        raise ValueError("data.split_frames must map train, val, dev or test to {run pattern: [[start, stop], ...]}")
    for split, spec in frames.items():
        if not isinstance(spec, dict) or not all(
                isinstance(pattern, str) and pattern and isinstance(ranges, list) and ranges and all(
                    isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)
                    and 0 <= pair[0] < pair[1] for pair in ranges) for pattern, ranges in spec.items()):
            raise ValueError(f"data.split_frames.{split} must map run patterns to [[start, stop], ...] "
                             "frame ranges with 0 <= start < stop")
    return cfg


def run_name(cfg):
    """Name a stage-1 run by the settings that experiments vary."""
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    parts = [f"{t['objective']}-{t['mode']}",
             f"k{m['state_window_trs']}c{m['context_states']}d{m['state_dim']}l{m['predictor_depth']}",
             f"seq{d['sequence_states']}"]
    if m["state_step"] != 1:
        parts.append(f"step{m['state_step']}")
    if m["chunk_frames"]:
        parts.append(f"chunk{m['chunk_frames']}")
    if m["fast_window_trs"]:
        parts.append(f"fast{m['fast_window_trs']}")
    if m["state_tokens"] > 1:
        parts.append(f"tokens{m['state_tokens']}" + ("own" if m["own_drive"] else ""))
    if m["movie_rank"]:
        parts.append(f"movierank{m['movie_rank']}")
    if m["slow_per_dimension"]:
        parts.append("slowperdim")
    if m["predictor_kind"] == "mlp":
        parts.append(f"mlp{m['mlp_depth']}x{m['mlp_hidden_dim']}")
    if m["predictor_kind"] == "corrected":
        parts.append("corrected")
    if not m["predictor_output_norm"]:
        parts.append("nofinalnorm")
    if m["movie_tokens"]:
        parts.append("movietokens")
    if m["token_positions_file"]:
        parts.append("tokenpositions")
    if m["encoder_hidden_dim"]:
        parts.append(f"encmlp{m['encoder_hidden_dim']}")
    if m["history_blocks"]:
        parts.append(f"hist{m['history_blocks']}x{m['history_block_states']}"
                     + ("measured" if m["history_measured_only"] else "") + ("spaced" if m["history_spaced"] else ""))
    if cfg["eval"]["prefix_trs"] != 1:
        parts.append(f"pre{cfg['eval']['prefix_trs']}")
    if t["mode"] == "stimulus":
        parts.append(f"lag{math.floor(d['hrf_delay_seconds'] / d['tr_seconds'] + 0.5)}")
        if m["stimulus_features"] == "tribev2_context":
            parts.append("tribectx")
        if m["stimulus_context"] != "none":
            short = {"concat": "cat", "linear": "lin", "concat-audio": "cataudio"}[m["stimulus_context"]]
            parts.append(f"ctx-{short}{m['stimulus_context_trs']}")
        if t["modality_dropout"]:
            parts.append(f"mdrop{t['modality_dropout']:g}")
        if m["stimulus_scaling"] != "layernorm":
            parts.append("std")
        if m["conditioning"] != "adaln":
            parts.append(f"cond-{m['conditioning']}")
    if t["rollout_steps"]:
        parts.append(f"roll{t['rollout_steps']}w{t['rollout_weight']:g}")
    if t["prediction_weight"] != 1:
        parts.append(f"pw{t['prediction_weight']:g}")
    if t["rollout_step_weights"] != "uniform":
        parts.append(f"sw-{t['rollout_step_weights'].replace('_', '')}")
    if t["rollout_detach_every"]:
        parts.append(f"tbptt{t['rollout_detach_every']}")
    if t["rollout_schedule"]:
        parts.append("curr" + "-".join(f"{e}x{n}" for e, n in t["rollout_schedule"]))
    if t["objective"] == "latent" and not t["freeze_encoder"]:
        parts.append(f"sig{t['sigreg_weight']:g}")
    if t["target"] != "encoder":
        parts.append("tgt-initial" if t["target"] == "initial" else f"tgt-ema{t['target_momentum']:g}")
    if t["parcel_weight"]:
        parts.append(f"parcels{t['parcel_weight']:g}")
    if t["predictor_checkpoint"]:
        parts.append("predinit")
    if t["encoder_checkpoint"]:
        # Name the source run too: <objective>-<mode> from its directory name, plus a short
        # hash of its path, so runs built on different encoders never share a directory.
        source = Path(t["encoder_checkpoint"])
        digest = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:6]
        parts.append(f"{'encfrozen' if t['freeze_encoder'] else 'encinit'}-{source.parent.name.split('_')[0]}-{digest}")
    parts += [f"lr{t['lr']:g}", f"bs{d['batch_size']}", f"ep{t['epochs']}", f"seed{cfg['seed']}"]
    if t["run_tag"]:
        parts.append(t["run_tag"])
    return "_".join(parts)


def default_run_dir(cfg, root="checkpoints"):
    """checkpoints/<dataset>/<run name>, relative to the working directory (the repository root)."""
    dataset = json.loads((Path(cfg["data"]["bundle_path"]) / "bundle.json").read_text())["dataset_id"]
    return Path(root) / dataset / run_name(cfg)


def apply_overrides(cfg, assignments):
    """Apply ``section.key=value`` assignments; values are parsed as YAML (numbers, lists, null)."""
    _assign(cfg, assignments)
    return validate_config(cfg)


def _assign(cfg, assignments):
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        section, _, key = name.partition(".")
        if not separator or not key:
            raise ValueError(f"Override must look like section.key=value: {assignment!r}")
        if section not in cfg or not isinstance(cfg[section], dict) or key not in cfg[section]:
            raise ValueError(f"Unknown configuration field: {name}")
        if cfg["design"] and section == "train" and key in LOCKED:
            raise ValueError(f"train.{key} is fixed by design {cfg['design']}; choose another design YAML")
        cfg[section][key] = expand(yaml.safe_load(text))


def _read_yaml(path):
    """The YAML mapping at path, with ${STATE_OF_MIND_*} (and legacy ${NEUROWORLD_*}) placeholders replaced by the
    roots of state_of_mind/paths.py."""
    with Path(path).open() as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return expand(supplied)


def _merge(cfg, supplied):
    for section, values in supplied.items():
        if section not in cfg:
            raise ValueError(f"Unknown configuration field: {section}")
        if section in ("seed", "design"):
            cfg[section] = values
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(values) - set(cfg[section])
        if unknown:
            raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
        cfg[section].update(values)


def load_config(path, overrides=()):
    """Load a design YAML (base + design) or a saved, fully resolved run configuration.

    ``overrides`` are ``section.key=value`` assignments applied on top. For the Transformer world model, those in
    ENCODER_SHAPE also select the stage-0 run: the one trained with the same assignments.
    """
    path = Path(path)
    supplied = _read_yaml(path)
    base = supplied.pop("base", None)
    encoder_from = supplied.pop("encoder_from", None)
    design = supplied.get("design")
    cfg = deepcopy(DEFAULTS)
    if base is not None:
        base_values = _read_yaml(path.parent / base)
        if {"base", "design", "encoder_from"} & set(base_values):
            raise ValueError(f"{base} is a base configuration; it may not declare a design")
        if any(key in base_values.get("train", {}) for key in LOCKED):
            raise ValueError(f"{base} may not set {LOCKED}; they are fixed by each design")
        _merge(cfg, base_values)
        if design is None:
            raise ValueError(f"{path} names a base, so it must declare design: {sorted(DESIGNS)}")
        if any(key in supplied.get("train", {}) for key in LOCKED):
            raise ValueError(f"{path} may not set {LOCKED}; they are fixed by design {design}")
    _merge(cfg, supplied)
    if base is not None:
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {sorted(DESIGNS)}")
        cfg["train"].update(DESIGNS[design])
        cfg["train"]["encoder_checkpoint"] = None
    elif encoder_from is not None:
        raise ValueError("encoder_from is only valid in a Transformer world model YAML that names a base")
    _assign(cfg, overrides)
    if base is not None and design in PHASE0_DESIGNS:
        shared = [item for item in overrides if item.partition("=")[0] in ENCODER_SHAPE]
        cfg["train"]["encoder_checkpoint"] = str(_phase0_checkpoint(path, encoder_from, shared))
    return validate_config(cfg)


def _phase0_checkpoint(path, encoder_from, overrides=()):
    """best.pt of the run trained from the stage-0 YAML that the Transformer world model names in encoder_from."""
    if not encoder_from:
        raise ValueError(f"{path}: the Transformer world model needs encoder_from: <stage-0 YAML>")
    phase0 = load_config(path.parent / encoder_from, overrides)
    if phase0["design"] != "stage0_forecaster":
        raise ValueError(f"{encoder_from} must be a stage0_forecaster YAML")
    checkpoint = default_run_dir(phase0) / "best.pt"
    if not checkpoint.is_file():
        shared = f" with --set {' '.join(overrides)}" if overrides else ""
        raise FileNotFoundError(f"The Transformer world model uses the stage-0 encoder from {encoder_from}, expected at "
                                f"{checkpoint}; train {encoder_from}{shared} first")
    return checkpoint
