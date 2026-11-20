"""Dataset construction, epoch-seeded loaders, optional encoder initialization and early stopping."""
import torch
from torch.utils.data import DataLoader
from neuroworld.data.schaefer_v2_dataset import SchaeferWindowDataset
from neuroworld.data.splits import check_splits, frame_ranges, split_runs
from neuroworld.utils.seed import seed_worker


def window_settings(cfg, purpose):
    """States per window and window stride for one purpose: train, eval or readout."""
    if purpose == "train":
        return cfg["data"]["sequence_states"], cfg["data"]["window_stride_trs"]
    if purpose == "eval":
        return cfg["eval"]["prefix_trs"] + cfg["eval"]["horizon_trs"], cfg["eval"]["window_stride_trs"]
    if purpose == "readout":
        return cfg["eval"]["prefix_trs"] + cfg["readout"]["train_horizon_trs"], cfg["readout"]["window_stride_trs"]
    raise ValueError(f"Unknown window purpose: {purpose}")


def history_fill_frames(cfg):
    """data.history_fill: a long-history encoder's oldest frames (its window minus its fast frames) may be missing."""
    m = cfg["model"]
    if not cfg["data"]["history_fill"] or not m.get("fast_window_trs"):
        return 0
    return m["state_window_trs"] - m["fast_window_trs"]


def build_dataset(cfg, split, *, purpose=None, preload=None):
    """Windows of one split (train, val, dev or test; data/splits.py), after checking that no two
    splits share frames."""
    d, m = cfg["data"], cfg["model"]
    num_states, stride = window_settings(cfg, purpose or ("train" if split == "train" else "eval"))
    records = check_splits(cfg)
    runs = [record["run_id"] for record in split_runs(cfg, split, records)]
    if not runs:
        raise ValueError(f"The {split} split has no runs after removing data.heldout_subjects")
    ranges = {run_id: frame_ranges(cfg, split, run_id) for run_id in runs}
    return SchaeferWindowDataset(
        d["bundle_path"], runs, tr_seconds=d["tr_seconds"],
        state_window_trs=m["state_window_trs"], num_states=num_states,
        hrf_delay_seconds=d["hrf_delay_seconds"], window_stride_trs=stride,
        include_stimulus=cfg["train"]["mode"] == "stimulus",
        preload=d["preload"] if preload is None else preload,
        stimulus_context_trs=m["stimulus_context_trs"],
        context_features=d["context_features"] if m["stimulus_features"] == "tribev2_context" else None,
        context_width=m["context_width"],
        frame_ranges={run_id: value for run_id, value in ranges.items() if value},
        history_fill_frames=history_fill_frames(cfg),
    )


def local_batch_size(global_batch, num_processes):
    """Configured batch sizes are global; each process loads an equal share."""
    if global_batch % num_processes:
        raise ValueError(f"Global batch {global_batch} is not divisible by {num_processes} processes")
    return global_batch // num_processes


def build_loader(dataset, cfg, *, shuffle=False, batch_size=None, drop_last=False):
    # Accelerate's seedable sampler and set_epoch reshuffle each epoch from this seed.
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=batch_size or cfg["data"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], worker_init_fn=seed_worker,
                      generator=generator, drop_last=drop_last)


def load_encoder_init(model, cfg):
    """Initialize (and optionally freeze) the encoder from a compatible stage-1 checkpoint.

    The Transformer world model uses it to load and freeze the encoder of its stage-0 run.
    """
    from neuroworld.train.checkpoint import read_checkpoint
    path = cfg["train"]["encoder_checkpoint"]
    if not path:
        return
    checkpoint = read_checkpoint(path)
    source = checkpoint["config"]
    for key in ("state_window_trs", "state_dim"):
        if source["model"][key] != cfg["model"][key]:
            raise ValueError(f"Encoder checkpoint has incompatible model.{key}")
    if abs(source["data"]["tr_seconds"] - cfg["data"]["tr_seconds"]) > 1e-8:
        raise ValueError("Encoder initialization requires the same native TR")
    weights = {name[len("encoder."):]: tensor for name, tensor in checkpoint["model"].items()
               if name.startswith("encoder.")}
    model.encoder.load_state_dict(weights, strict=True)
    if cfg["train"]["freeze_encoder"]:
        model.freeze_encoder()


def load_predictor_init(model, cfg):
    """Start the predictor from a stage-0 run's predictor (train.predictor_checkpoint) and go on training all of it. The
    stage-0 run has no movie, so its movie projection is not loaded: the predictor starts as the brain-only dynamics,
    with the movie added from zero. With train.predictor_checkpoint_movie, the file holds the whole predictor, movie included."""
    from neuroworld.train.checkpoint import read_checkpoint
    path = cfg["train"]["predictor_checkpoint"]
    if not path:
        return
    checkpoint = read_checkpoint(path)
    movie = cfg["train"].get("predictor_checkpoint_movie", False)
    weights = {name[len("predictor."):]: tensor for name, tensor in checkpoint["model"].items()
               if name.startswith("predictor.") and (movie or not name.startswith(("predictor.condition.", "predictor.condition_norm.")))}
    missing, extra = model.predictor.load_state_dict(weights, strict=False)
    if extra or [name for name in missing if not name.startswith(("condition.", "condition_norm.", "correction."))]:
        raise ValueError(f"Predictor checkpoint does not match: missing {missing}, unexpected {extra}")


def patience_ran_out(scores, patience):
    """Early stopping: true once the last `patience` epochs have not beaten the best selection score (higher is
    better). Epochs keep their place in the learning-rate schedule, so a stopped run keeps the model a full-length
    run would have kept, unless that run found a better epoch later. A patience of None never stops."""
    if not patience or not scores:
        return False
    best = max(range(len(scores)), key=scores.__getitem__)
    return len(scores) - 1 - best >= patience
