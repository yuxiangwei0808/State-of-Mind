"""W&B run setup and scalar logging through Accelerate."""
import json
import math
import os
from pathlib import Path
from neuroworld.train.checkpoint import rng_state, restore_rng


def wandb_options(cfg):
    options = dict(cfg["wandb"])
    for key, variable in (("project", "WANDB_PROJECT"), ("entity", "WANDB_ENTITY"),
                          ("name", "WANDB_NAME"), ("mode", "WANDB_MODE")):
        if os.environ.get(variable):
            options[key] = os.environ[variable]
    if options["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("W&B mode must be online, offline, or disabled")
    return options


def init_tracking(accelerator, cfg, output, *, stage, metadata, data_summary, resume=None):
    options = wandb_options(cfg)
    if options["mode"] == "disabled" or not accelerator.is_main_process:
        return None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    group = output.parent.name if stage == "readout" else output.name
    kwargs = {"name": options["name"] or f"{group}-{stage}", "entity": options["entity"],
              "mode": options["mode"], "dir": str(output), "job_type": stage, "group": group}
    if resume and options["mode"] == "online":
        kwargs.update(id=resume["id"], resume="allow")
    # Tracker initialization must not change the training RNG, including on resume.
    rng = rng_state()
    try:
        accelerator.init_trackers(options["project"],
            config={**cfg, "wandb": options, "stage": stage, "runtime": metadata,
                    "data_summary": data_summary, "previous_wandb_run": resume},
            init_kwargs={"wandb": kwargs})
        run = accelerator.get_tracker("wandb", unwrap=True)
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        info = {"id": run.id, "project": run.project, "entity": run.entity,
                "name": run.name, "mode": options["mode"], "url": None if run.offline else run.url}
        (output / "wandb-run.json").write_text(json.dumps(info, indent=2) + "\n")
        return info
    finally:
        restore_rng(rng)


def scalar_metrics(values, prefix=""):
    """Flatten nested metrics; horizon arrays get one scalar per future step."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (list, tuple)):
            result.update(scalar_metrics(dict(enumerate(value, start=1)), name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    return result


def log_metrics(accelerator, values, *, step):
    if accelerator.is_main_process and accelerator.trackers:
        # Use our training step as the chart axis, not W&B's append-only record
        # counter: an interrupted epoch can replay earlier steps after resume.
        accelerator.log({**scalar_metrics(values), "global_step": step})
