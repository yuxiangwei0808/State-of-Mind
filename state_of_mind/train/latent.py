"""One visible PyTorch training loop for the latent world model or the fMRI forecaster."""
import json
import math
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
from accelerate import __version__ as accelerate_version
from accelerate.utils import broadcast_object_list, set_seed
from state_of_mind.eval.probe import fit_ridge_probe, probe_subset
from state_of_mind.eval.rollout import evaluate
from state_of_mind.models import build_model
from state_of_mind.models.collapse import SIGReg
from state_of_mind.train.checkpoint import (read_checkpoint, restore_model,
    restore_optimizer_and_rng, save_checkpoint)
from state_of_mind.train.losses import LatentObjective
from state_of_mind.train.runtime import training_accelerator, average_metrics
from state_of_mind.train.tracking import init_tracking, log_metrics
from state_of_mind.train.stage_utils import (build_dataset, build_loader, load_encoder_init, load_predictor_init,
                                         local_batch_size, patience_ran_out)
from state_of_mind.utils.config import default_run_dir, validate_config


def build_optimizer(module, lr, weight_decay, encoder_lr_factor=1.0):
    """AdamW on what trains. encoder_lr_factor scales the encoder's rate: a warm-started encoder's weights are small
    against the rate, and Adam's step does not shrink with them."""
    if encoder_lr_factor == 1.0:
        return torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    encoder = [p for p in module.encoder.parameters() if p.requires_grad]
    others = [p for p in module.parameters() if p.requires_grad and not any(p is q for q in encoder)]
    return torch.optim.AdamW([{"params": others}, {"params": encoder, "lr": lr * encoder_lr_factor}],
                             lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, *, warmup_steps, total_steps, schedule):
    """Per-step linear warmup, then cosine decay to zero or a constant rate."""
    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant" or total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(objective, loader, optimizer, scheduler, cfg, accelerator, *, epoch=0, global_step=0):
    objective.train()
    total = {"loss": 0.0, "prediction_loss": 0.0, "rollout_loss": 0.0, "sigreg": 0.0,
             "latent_spread": 0.0, "grad_norm": 0.0}
    count, steps, micro = 0, 0, 0
    started = time.perf_counter()
    trainable = [p for p in objective.parameters() if p.requires_grad]
    clip = cfg["train"]["grad_clip"] if cfg["train"]["grad_clip"] > 0 else float("inf")
    # With train.accumulation_steps > 1, each optimizer step averages the gradients of that many
    # consecutive batches (a trailing incomplete group is skipped), so the global batch is unchanged.
    accumulation = cfg["train"]["accumulation_steps"]
    total_steps = len(loader) // accumulation
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        if steps == total_steps:
            break
        micro += 1
        update = micro % accumulation == 0
        with nullcontext() if update else accelerator.no_sync(objective):
            with accelerator.autocast():
                loss, metrics = objective(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {micro - 1}")
            accelerator.backward(loss / accumulation)
        size = batch["brain"].shape[0]
        count += size
        total["loss"] += float(loss.detach()) * size
        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + float(value) * size  # optional terms (parcel_loss) appear here
        if not update:
            continue
        grad_norm = accelerator.clip_grad_norm_(trainable, clip)
        optimizer.step()
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()
            follow = getattr(accelerator.unwrap_model(objective).model, "follow_target", None)
            if follow is not None:
                follow()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        total["grad_norm"] += float(grad_norm) * size * accumulation
        if steps == 1 or steps % cfg["train"]["log_every_steps"] == 0 or steps == total_steps:
            summary, seen = average_metrics(total, count, accelerator)
            if accelerator.is_main_process:
                elapsed = time.perf_counter() - started
                print(json.dumps({"event": "train_progress", "epoch": epoch,
                                  "step": steps, "steps_in_epoch": total_steps,
                                  "global_step": global_step + steps, "lr": optimizer.param_groups[0]["lr"],
                                  "elapsed_seconds": elapsed, "windows_per_second": seen / elapsed,
                                  "peak_gpu_memory_mb": gpu_memory_mb(accelerator.device), **summary},
                                 allow_nan=False), flush=True)
                log_metrics(accelerator, {"epoch": epoch,
                    "train": {**summary, "lr": optimizer.param_groups[0]["lr"]},
                    "performance": {"windows_per_second": seen / elapsed,
                                    "peak_gpu_memory_mb": gpu_memory_mb(accelerator.device)}},
                    step=global_step + steps)
    if not count:
        raise ValueError("Training loader yielded no windows")
    summary, _ = average_metrics(total, count, accelerator)
    summary["lr"] = optimizer.param_groups[0]["lr"]
    return summary, steps


def gpu_memory_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None


def run_metadata(cfg, device, resume, **extra):
    root = Path(__file__).resolve().parents[2]
    command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
               "-c", "core.hooksPath=/dev/null", "-C", str(root)]
    commit, dirty = None, None
    try:
        head = subprocess.run(command + ["rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        status = subprocess.run(command + ["status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=5)
        commit = head.stdout.strip() if head.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"started_at": datetime.now(timezone.utc).isoformat(),
            "source_commit": commit, "source_dirty": dirty,
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "accelerate_version": accelerate_version,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "seed": cfg["seed"], "mode": cfg["train"]["mode"],
            "resume_from": str(Path(resume).resolve()) if resume else None, **extra}


def resolve_output_dir(cfg, resume):
    """Explicit train.output_dir, else the resumed run's directory, else a new named run directory.

    A new (not resumed) run never starts in a directory that already holds one.
    """
    if cfg["train"]["output_dir"]:
        if not resume and (Path(cfg["train"]["output_dir"]) / "latest.pt").exists():
            raise FileExistsError(f"{cfg['train']['output_dir']} already holds a run; resume it with --resume, "
                                  "or choose another RUN_DIR (--output-dir)")
        return
    if resume:
        cfg["train"]["output_dir"] = str(Path(resume).parent)
        return
    output = default_run_dir(cfg)
    if (output / "latest.pt").exists():
        raise FileExistsError(f"{output} already holds a run; resume it with --resume, "
                              "or set RUN_TAG (--tag) or RUN_DIR (--output-dir) for a new one")
    cfg["train"]["output_dir"] = str(output)


def fit(cfg, *, resume=None):
    validate_config(cfg)
    with training_accelerator(cfg) as accelerator:
        resolve_output_dir(cfg, resume)
        accelerator.print(json.dumps({"run_dir": cfg["train"]["output_dir"]}), flush=True)
        set_seed(cfg["seed"])
        device = accelerator.device
        train_data, val_data = build_dataset(cfg, "train"), build_dataset(cfg, "val")
        probe_loader = None
        if cfg["train"]["objective"] == "latent" and accelerator.is_main_process:
            # Fixed training windows for the selection probe; they reuse the training arrays.
            probe_data = build_dataset(cfg, "train", purpose="readout", preload=False)
            if hasattr(train_data, "_arrays"):
                probe_data._arrays = train_data._arrays
            probe_loader = build_loader(probe_subset(probe_data, cfg["eval"]["probe_windows"], cfg["seed"]), cfg)
        summary = {"dataset": train_data.dataset_id, "train_windows": len(train_data),
                   "val_windows": len(val_data), "train_runs": train_data.run_ids,
                   "val_runs": val_data.run_ids, "tr_seconds": train_data.tr_seconds,
                   "lag_trs": train_data.lag_trs,
                   "effective_hrf_delay_seconds": train_data.effective_hrf_delay_seconds,
                   "train_states_per_window": train_data.num_states,
                   "eval_prefix_trs": cfg["eval"]["prefix_trs"], "eval_horizon_trs": cfg["eval"]["horizon_trs"]}
        checkpoint = read_checkpoint(resume) if resume else None
        model = build_model(cfg, train_data.subject_ids, initialize_stimulus=checkpoint is None)
        if checkpoint is not None:
            restore_model(model, cfg, checkpoint, resume=True)
            if checkpoint["data_summary"] != summary:
                raise ValueError("Resume dataset/window selection differs from the saved checkpoint")
        else:
            load_encoder_init(model, cfg)
            load_predictor_init(model, cfg)
            if hasattr(model, "reset_target"):
                model.reset_target()
            if getattr(model, "stimulus_scale", None) is not None:  # statistics of every training run's rows
                model.stimulus_scale.fit([train_data._open(record["modalities"]["context"])
                                          for record in train_data.records])
        model.to(device)
        optimizer = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"], cfg["train"]["encoder_lr_factor"])
        # As in LeWM, drop the final partial batch so every SIGReg call sees a full batch.
        accumulation = cfg["train"]["accumulation_steps"]
        batch_per_rank = local_batch_size(cfg["data"]["batch_size"], accelerator.num_processes * accumulation)
        train_loader = accelerator.prepare_data_loader(
            build_loader(train_data, cfg, shuffle=True, drop_last=True, batch_size=batch_per_rank))
        total_steps = cfg["train"]["epochs"] * (len(train_loader) // accumulation)
        # As in LeWM, warm up over a fraction of all optimizer steps.
        warmup_steps = round(cfg["train"]["warmup_fraction"] * total_steps)
        lr_scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps,
                                       total_steps=total_steps, schedule=cfg["train"]["schedule"])
        regularizer = SIGReg(cfg["train"]["sigreg_knots"], cfg["train"]["sigreg_projections"]).to(device)
        objective = LatentObjective(model, regularizer, cfg["train"])
        objective, optimizer, scheduler = accelerator.prepare(objective, optimizer, lr_scheduler)
        if accelerator.num_processes > 1:
            set_seed(cfg["seed"], device_specific=True)
        start_epoch, global_step, best_score, history = 0, 0, float("-inf"), []
        schedule_change = None
        if checkpoint is not None:
            restore_optimizer_and_rng(optimizer, scheduler, checkpoint, device)
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            best_score = checkpoint.get("best_score", float("-inf"))
            history = checkpoint["history"]
            if checkpoint["total_steps"] != total_steps:
                # A longer run after resume recomputes the cosine schedule; this is recorded, not hidden.
                schedule_change = {"from_total_steps": checkpoint["total_steps"], "to_total_steps": total_steps}
                # Apply the new schedule at the saved step before the first optimizer update.
                rates = [base * factor(lr_scheduler.last_epoch)
                         for base, factor in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)]
                for group, rate in zip(optimizer.param_groups, rates):
                    group["lr"] = rate
                lr_scheduler.load_state_dict({**lr_scheduler.state_dict(), "_last_lr": rates})
        output = Path(cfg["train"]["output_dir"])
        wandb_run = None
        if accelerator.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
            metadata = run_metadata(cfg, device, resume, total_steps=total_steps,
                world_size=accelerator.num_processes, batch_size_per_rank=batch_per_rank,
                accumulation_steps=accumulation,
                effective_batch_size=cfg["data"]["batch_size"], warmup_steps=warmup_steps,
                schedule_change=schedule_change)
            with (output / "run.jsonl").open("a" if resume else "w") as handle:
                handle.write(json.dumps(metadata) + "\n")
            wandb_run = init_tracking(accelerator, cfg, output, stage="latent", metadata=metadata,
                data_summary=summary, resume=checkpoint.get("wandb_run") if checkpoint else None)
            if not resume:
                (output / "performance.jsonl").write_text("")
            (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (output / "data.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"objective": cfg["train"]["objective"], "mode": model.mode, "encoder_frozen": model.encoder_frozen,
                              "target": cfg["train"]["target"],
                              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                              "train_windows": len(train_data), "val_windows": len(val_data),
                              "total_steps": total_steps, "lag_trs": train_data.lag_trs,
                              "effective_hrf_delay_seconds": train_data.effective_hrf_delay_seconds}), flush=True)
            # Recreate the metrics file from checkpoint history when resuming.
            with (output / "metrics.jsonl").open("w") as handle:
                for entry in history:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
        for epoch in range(start_epoch, cfg["train"]["epochs"]):
            # Checked from the history, so a resumed run stops where the uninterrupted one would.
            if patience_ran_out([entry["val"]["selection_score"] for entry in history], cfg["train"]["patience"]):
                accelerator.print(json.dumps({"event": "early_stop", "epoch": epoch,
                                              "patience": cfg["train"]["patience"]}), flush=True)
                break
            epoch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_loader.set_epoch(epoch)
            accelerator.unwrap_model(objective).epoch = epoch
            train_metrics, steps = train_one_epoch(objective, train_loader, optimizer, scheduler, cfg,
                                                   accelerator, epoch=epoch, global_step=global_step)
            train_seconds = time.perf_counter() - epoch_started
            global_step += steps
            validation_started = time.perf_counter()
            validation = [None]
            if accelerator.is_main_process:
                val_loader = build_loader(val_data, cfg)
                with accelerator.autocast():
                    probe = None
                    if not getattr(model, "forecasts_fmri", False):
                        probe = fit_ridge_probe(model, probe_loader, device, prefix_trs=cfg["eval"]["prefix_trs"],
                                                ridge=cfg["eval"]["probe_ridge"])
                    validation[0] = evaluate(model, val_loader, device, prefix_trs=cfg["eval"]["prefix_trs"],
                                             horizon_trs=cfg["eval"]["horizon_trs"], readout=probe)
                # Select by the primary fMRI metric: decoded by the ridge probe for the latent world
                # model (latent errors cannot rank epochs whose latent spaces differ), directly for
                # the fMRI forecaster.
                validation[0]["selection_score"] = validation[0]["fmri"]["model"]["parcel_pearson"]
            validation = broadcast_object_list(validation)[0]
            validation_seconds = time.perf_counter() - validation_started
            score = validation["selection_score"]
            if score is None:
                raise ValueError("Validation fMRI parcel Pearson is undefined")
            improved = score > best_score
            best_score = max(best_score, score)
            entry = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": validation}
            history.append(entry)
            state = dict(accelerator=accelerator, epoch=epoch, global_step=global_step, total_steps=total_steps,
                         best_score=best_score, history=history, data_summary=summary, wandb_run=wandb_run)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, scheduler, cfg, **state)
            save_checkpoint(output / "latest.pt", model, optimizer, scheduler, cfg, **state)
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
                performance = {"epoch": epoch, "global_step": global_step,
                               "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                               "epoch_seconds": time.perf_counter() - epoch_started,
                               "train_windows_per_second": len(train_data) / train_seconds,
                               "peak_gpu_memory_mb": gpu_memory_mb(device)}
                # Wall-clock measurements stay outside checkpoint metric history so exact
                # resume comparisons remain about model behavior rather than elapsed time.
                with (output / "performance.jsonl").open("a") as handle:
                    handle.write(json.dumps(performance, allow_nan=False) + "\n")
                print(json.dumps(entry, allow_nan=False), flush=True)
                print(json.dumps({"event": "epoch_performance", **performance}), flush=True)
                log_metrics(accelerator, {**entry, "performance": performance,
                                          "best": {"selection_score": best_score}}, step=global_step)
        return history


def main():
    import argparse
    from state_of_mind.utils.config import load_config, validate_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", help="Restore an epoch checkpoint with the same process count")
    parser.add_argument("--output-dir", help="Run directory; default checkpoints/<dataset>/<run name>")
    parser.add_argument("--tag", dest="run_tag", help="Suffix for the automatic run name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Global training batch size, split across GPUs")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--set", nargs="+", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override YAML settings, e.g. --set train.rollout_steps=4 data.sequence_states=7")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["design"] is None and not args.resume:
        raise SystemExit(f"{args.config} declares no design; launch a design YAML such as "
                         "state_of_mind/configs/algonauts/transformer_world_model.yaml")
    for key in ("epochs", "lr", "output_dir", "precision", "run_tag"):
        value = getattr(args, key)
        if value is not None:
            cfg["train"][key] = value
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    fit(validate_config(cfg), resume=args.resume)


if __name__ == "__main__":
    main()
