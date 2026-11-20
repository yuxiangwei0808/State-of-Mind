"""Accelerate setup and reduction shared by training and evaluation."""
from contextlib import contextmanager
from datetime import timedelta
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, is_wandb_available
from neuroworld.train.tracking import wandb_options


def create_accelerator(cfg):
    requested = cfg["train"]["device"]
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Use train.device auto/cpu/cuda and CUDA_VISIBLE_DEVICES to select GPUs")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("train.device cuda requires an available GPU")
    precision = cfg["train"]["precision"]
    if precision == "bf16" and (requested == "cpu" or not torch.cuda.is_available()):
        raise ValueError("train.precision bf16 requires a CUDA device")
    tracking = wandb_options(cfg)["mode"] != "disabled"
    if tracking and not is_wandb_available():
        raise ImportError("W&B logging requires wandb; install neuroworld/requirements.txt")
    return Accelerator(
        log_with="wandb" if tracking else None,
        cpu=requested == "cpu", mixed_precision="bf16" if precision == "bf16" else "no",
        dataloader_config=DataLoaderConfiguration(split_batches=False, even_batches=True,
                                                  use_seedable_sampler=True, data_seed=cfg["seed"]),
        # The loop schedules one step per actual optimizer update.
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
                         InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )


@contextmanager
def training_accelerator(cfg):
    accelerator = create_accelerator(cfg)
    try:
        yield accelerator
    except BaseException:
        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb" and hasattr(tracker, "run"):
                    tracker.run.finish(exit_code=1)
            accelerator.trackers.clear()
        raise
    finally:
        accelerator.free_memory()
        accelerator.end_training()


def average_metrics(totals, count, accelerator):
    values = torch.tensor([*totals.values(), count], dtype=torch.float64, device=accelerator.device)
    values = accelerator.reduce(values, reduction="sum").tolist()
    count = int(values[-1])
    return {key: value / count for key, value in zip(totals, values[:-1])}, count
