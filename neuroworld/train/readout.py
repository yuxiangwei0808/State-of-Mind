"""Stage 2: train the frozen-latent readout on autoregressive rollouts.

The world model is loaded, frozen and never updated. Readout windows use the
evaluation prefix and readout.train_horizon_trs; training latents come from rollouts
(readout.train_on: rollout) or from encoded targets (encoded) for comparison.
Validation always decodes rollouts, as evaluation does, and reports fMRI MSE and the
evaluation's parcel Pearson; readout.select_by chooses which one picks readout_best.pt.
"""
import json
import time
from pathlib import Path
import torch
import yaml
from accelerate.utils import broadcast_object_list, set_seed
from neuroworld.eval.fmri import FmriAccumulator
from neuroworld.eval.rollout import move_batch
from neuroworld.models import Readout, build_model
from neuroworld.train.checkpoint import (READOUT_FORMAT, read_checkpoint, restore_model,
    save_readout_checkpoint, checkpoint_sha256)
from neuroworld.train.latent import build_optimizer, build_scheduler, run_metadata, gpu_memory_mb
from neuroworld.train.stage_utils import build_dataset, build_loader, local_batch_size, patience_ran_out
from neuroworld.utils.config import validate_config
from neuroworld.train.runtime import training_accelerator, average_metrics
from neuroworld.train.tracking import init_tracking, log_metrics


def load_world_model(cfg, checkpoint_path, device):
    checkpoint = read_checkpoint(checkpoint_path)
    model = build_model(cfg, checkpoint.get("subjects"), initialize_stimulus=False)
    restore_model(model, cfg, checkpoint)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


def load_readout(path, device, *, world_model_path):
    checkpoint = read_checkpoint(path, expected=READOUT_FORMAT)
    fingerprint = checkpoint.get("world_model_sha256")
    if not fingerprint:
        raise ValueError("Readout checkpoint has no world-model fingerprint; retrain the readout")
    if fingerprint != checkpoint_sha256(world_model_path):
        raise ValueError("Readout was trained with a different world-model checkpoint")
    source = checkpoint["config"]
    readout = Readout(source["model"]["state_dim"], source["readout"]["hidden_dim"], checkpoint["subjects"])
    readout.load_state_dict(checkpoint["readout"], strict=True)
    readout.to(device).eval()
    return readout, checkpoint


@torch.no_grad()
def rollout_targets(model, batch, cfg, train_on=None):
    """Latents to decode and the measured frames they must reproduce: [B,H,D], [B,H,1000]."""
    k, h0 = model.encoder.state_window_trs, cfg["eval"]["prefix_trs"]
    brain = batch["brain"]
    horizon = brain.shape[1] - (k + h0 - 1)
    if horizon < 1:
        raise ValueError("Readout windows must extend beyond the observed prefix")
    if (train_on or cfg["readout"]["train_on"]) == "rollout":
        latents = model.rollout(brain[:, :k + h0 - 1], batch["stimuli"], horizon_trs=horizon)
    else:
        latents = model.encode(brain)[:, h0:]
    return latents, brain[:, k + h0 - 1:]


def materialize(model, loader, cfg, accelerator, readout, train_on=None):
    for raw in loader:
        batch = move_batch(raw, accelerator.device)
        with accelerator.autocast():
            latents, observed = rollout_targets(model, batch, cfg, train_on)
        yield latents.cpu(), observed.cpu(), readout.subject_indices(raw["subject_id"]), list(raw["run_id"])


def cached_batches(model, loader, cfg, accelerator, readout, train_on=None):
    latents, observed, subjects, runs = [], [], [], []
    for l, o, s, r in materialize(model, loader, cfg, accelerator, readout, train_on):
        latents.append(l)
        observed.append(o)
        subjects.append(s)
        runs += r
    return torch.cat(latents), torch.cat(observed), torch.cat(subjects), runs


def iterate_cached(cache, batch_size, *, shuffle, generator):
    latents, observed, subjects, runs = cache
    order = torch.randperm(len(subjects), generator=generator) if shuffle else torch.arange(len(subjects))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        yield latents[index], observed[index], subjects[index], [runs[i] for i in index.tolist()]


def readout_epoch(readout, batches, accelerator, *, optimizer=None, scheduler=None, grad_clip=1.0,
                  epoch=0, global_step=0, log_every_steps=100, steps_in_epoch=None):
    total, count, grad_total = 0.0, 0, 0.0
    started = time.perf_counter()
    for step, (latents, observed, subjects, _) in enumerate(batches, start=1):
        latents = latents.to(accelerator.device)
        observed = observed.to(accelerator.device)
        subjects = subjects.to(accelerator.device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), accelerator.autocast():
            loss = (readout(latents, subjects) - observed).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite readout loss")
        if optimizer is not None:
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(readout.parameters(), grad_clip if grad_clip > 0 else float("inf"))
            grad_total += float(grad_norm) * len(subjects)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
        total += float(loss.detach()) * len(subjects)
        count += len(subjects)
        if optimizer is not None and (step == 1 or step % log_every_steps == 0 or step == steps_in_epoch):
            metrics, seen = average_metrics({"fmri_mse": total, "grad_norm": grad_total}, count, accelerator)
            if accelerator.is_main_process:
                progress = {"epoch": epoch, "global_step": global_step + step,
                    "train": {**metrics, "lr": optimizer.param_groups[0]["lr"]},
                    "performance": {"windows_per_second": seen / (time.perf_counter() - started),
                                    "peak_gpu_memory_mb": gpu_memory_mb(accelerator.device)}}
                print(json.dumps({"event": "readout_progress", **progress}, allow_nan=False), flush=True)
                log_metrics(accelerator, progress, step=global_step + step)
    if not count:
        raise ValueError("Readout loader yielded no windows")
    metrics, _ = average_metrics({"fmri_mse": total}, count, accelerator)
    return metrics["fmri_mse"]


@torch.no_grad()
def validate_readout(readout, batches, accelerator):
    """Validation fMRI MSE and the evaluation's parcel Pearson (per run and parcel, then averaged)."""
    total, count, accumulator = 0.0, 0, None
    for latents, observed, subjects, runs in batches:
        with accelerator.autocast():
            predicted = readout(latents.to(accelerator.device), subjects.to(accelerator.device))
        predicted = predicted.float().cpu()
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Non-finite readout prediction")
        if accumulator is None:
            accumulator = FmriAccumulator(observed.shape[1], parcels=observed.shape[2])
        total += float((predicted - observed).square().mean(dim=(1, 2)).sum())
        count += len(subjects)
        accumulator.add(runs, [readout.subjects[i] for i in subjects.tolist()], predicted, observed)
    if not count:
        raise ValueError("Readout validation yielded no windows")
    return {"fmri_mse": total / count, "parcel_pearson": accumulator.summary()["parcel_pearson"]}


def fit_readout(cfg, world_model_path, *, output_dir=None):
    validate_config(cfg)
    if cfg["train"]["objective"] == "fmri":
        raise ValueError("The fMRI forecaster trains its readout in Stage 1; there is no Stage 2")
    with training_accelerator(cfg) as accelerator:
        set_seed(cfg["seed"])
        device = accelerator.device
        model, world_checkpoint = load_world_model(cfg, world_model_path, device)
        world_model_sha256 = checkpoint_sha256(world_model_path)
        model = accelerator.prepare_model(model, evaluation_mode=True)
        reference = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        train_data = build_dataset(cfg, "train", purpose="readout")
        val_data = build_dataset(cfg, "val", purpose="readout")
        unknown = set(val_data.subject_ids) - set(train_data.subject_ids)
        if unknown:
            raise ValueError(f"Validation subjects without a trained head: {sorted(unknown)[:3]}")
        r = cfg["readout"]
        readout = Readout(cfg["model"]["state_dim"], r["hidden_dim"], train_data.subject_ids).to(device)
        optimizer = build_optimizer(readout, r["lr"], r["weight_decay"])
        batch_per_rank = local_batch_size(r["batch_size"], accelerator.num_processes)
        train_loader = accelerator.prepare_data_loader(
            build_loader(train_data, cfg, batch_size=batch_per_rank, shuffle=not r["cache_rollouts"]))
        total_steps = r["epochs"] * len(train_loader)
        scheduler = build_scheduler(optimizer, warmup_steps=r["warmup_steps"], total_steps=total_steps,
                                    schedule=r["schedule"])
        training_readout, optimizer, scheduler = accelerator.prepare(readout, optimizer, scheduler)
        if accelerator.num_processes > 1:
            set_seed(cfg["seed"], device_specific=True)
        output = Path(output_dir) if output_dir else Path(world_model_path).parent / "readout"
        summary = {"dataset": train_data.dataset_id, "train_windows": len(train_data),
                   "val_windows": len(val_data), "train_runs": train_data.run_ids, "val_runs": val_data.run_ids,
                   "subjects": train_data.subject_ids, "prefix_trs": cfg["eval"]["prefix_trs"],
                   "train_horizon_trs": r["train_horizon_trs"], "train_on": r["train_on"],
                   "world_model_path": str(Path(world_model_path).resolve()),
                   "world_model_epoch": world_checkpoint["epoch"],
                   "world_model_sha256": world_model_sha256}
        wandb_run = None
        if accelerator.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
            metadata = run_metadata(cfg, device, None, stage="readout", total_steps=total_steps,
                world_size=accelerator.num_processes, batch_size_per_rank=batch_per_rank,
                effective_batch_size=r["batch_size"])
            (output / "run.jsonl").write_text(json.dumps(metadata) + "\n")
            wandb_run = init_tracking(accelerator, cfg, output, stage="readout",
                                      metadata=metadata, data_summary=summary)
            (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (output / "data.json").write_text(json.dumps(summary, indent=2) + "\n")
            (output / "metrics.jsonl").write_text("")
            print(json.dumps({"stage": "readout", "subjects": train_data.subject_ids,
                              "trainable_parameters": sum(p.numel() for p in readout.parameters()),
                              "train_windows": len(train_data), "val_windows": len(val_data),
                              "total_steps": total_steps, "cache_rollouts": r["cache_rollouts"]}), flush=True)
        cache_started = time.perf_counter()
        train_cache = val_cache = None
        if r["cache_rollouts"]:
            with accelerator.autocast():
                train_cache = cached_batches(model, train_loader, cfg, accelerator, readout)
                if accelerator.is_main_process:
                    val_cache = cached_batches(model, build_loader(val_data, cfg, batch_size=r["batch_size"]),
                                               cfg, accelerator, readout, train_on="rollout")
            accelerator.wait_for_everyone()
            accelerator.print(json.dumps({"event": "rollouts_cached", "seconds": time.perf_counter() - cache_started,
                                          "train_windows_per_rank": len(train_cache[2]),
                                          "val_windows": len(val_data)}), flush=True)
            log_metrics(accelerator, {"performance": {"cache_seconds": time.perf_counter() - cache_started}}, step=0)
        # Higher is better for the selection score: parcel Pearson, or the negative MSE.
        best_score, best_loss, history, global_step, scores = float("-inf"), float("inf"), [], 0, []
        for epoch in range(r["epochs"]):
            if patience_ran_out(scores, r["patience"]):
                accelerator.print(json.dumps({"event": "early_stop", "epoch": epoch, "patience": r["patience"]}),
                                  flush=True)
                break
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training_readout.train()
            generator = torch.Generator().manual_seed(cfg["seed"] + epoch + accelerator.process_index)
            train_loader.set_epoch(epoch)
            if train_cache is not None:
                batches = iterate_cached(train_cache, batch_per_rank, shuffle=True, generator=generator)
            else:
                batches = materialize(model, train_loader, cfg, accelerator, readout)
            train_loss = readout_epoch(training_readout, batches, accelerator, optimizer=optimizer,
                scheduler=scheduler, grad_clip=cfg["train"]["grad_clip"], epoch=epoch,
                global_step=global_step, log_every_steps=cfg["train"]["log_every_steps"],
                steps_in_epoch=len(train_loader))
            global_step += len(train_loader)
            readout.eval()
            val = [None]
            if accelerator.is_main_process:
                if val_cache is not None:
                    batches = iterate_cached(val_cache, r["batch_size"], shuffle=False, generator=generator)
                else:
                    batches = materialize(model, build_loader(val_data, cfg, batch_size=r["batch_size"]),
                                          cfg, accelerator, readout, train_on="rollout")
                val[0] = validate_readout(readout, batches, accelerator)
            val = broadcast_object_list(val)[0]
            entry = {"epoch": epoch, "global_step": global_step, "lr": optimizer.param_groups[0]["lr"],
                     "train": {"fmri_mse": train_loss}, "val": val,
                     "epoch_seconds": time.perf_counter() - started}
            history.append(entry)
            score = -val["fmri_mse"] if r["select_by"] == "fmri_mse" else val["parcel_pearson"]
            scores.append(score)
            improved = score > best_score
            best_score = max(best_score, score)
            best_loss = min(best_loss, val["fmri_mse"])
            state = dict(accelerator=accelerator, world_model_path=summary["world_model_path"], epoch=epoch,
                         global_step=global_step, total_steps=total_steps, best_loss=best_loss,
                         history=history, data_summary=summary, world_model_sha256=world_model_sha256,
                         wandb_run=wandb_run)
            if improved:
                save_readout_checkpoint(output / "readout_best.pt", training_readout, optimizer, scheduler, cfg, **state)
            save_readout_checkpoint(output / "readout_latest.pt", training_readout, optimizer, scheduler, cfg, **state)
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
                print(json.dumps(entry, allow_nan=False), flush=True)
                best = -best_score if r["select_by"] == "fmri_mse" else best_score
                log_metrics(accelerator, {**entry, "best": {"fmri_mse": best_loss, r["select_by"]: best},
                    "performance": {"epoch_seconds": entry["epoch_seconds"],
                                    "peak_gpu_memory_mb": gpu_memory_mb(device)}}, step=global_step)
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor.detach().cpu(), reference[name]):
                raise RuntimeError(f"World model parameter changed during readout training: {name}")
        return history


def main():
    import argparse
    from neuroworld.utils.config import apply_overrides, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True, help="Frozen stage-1 world model, usually best.pt")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Global readout batch size, split across GPUs")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--set", nargs="+", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override settings, e.g. --set readout.hidden_dim=0 readout.select_by=parcel_pearson")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    for key in ("epochs", "lr", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            cfg["readout"][key] = value
    if args.precision is not None:
        cfg["train"]["precision"] = args.precision
    fit_readout(cfg, args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
