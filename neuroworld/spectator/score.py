"""Spectator's scores on every benchmark task, as the paper reports them.

The model has no decoder, so it is read through a linear probe per viewer (one pooled probe on HCP), fitted on the model's
own rollouts on the training recordings and frozen before any scored recording. The probe is fitted twice: the second
time on rollouts whose hidden frames are filled through the first probe, as Updating fills them. Queries are as long as the
model's window (`seconds`), so frames before a run start stay zero, as in training.

Tasks: Forecast on every scored split; State; Updating; the future-frames check; with gap, Forecast on the training
recordings (the training gap).
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import neuroworld
from neuroworld.benchmark import forecast as F, grid as G, nonanticipation as N, state as S, updating as U
from neuroworld.benchmark.adapters import RolloutAdapter
from neuroworld.benchmark.data import BenchmarkData
from neuroworld.benchmark.latent import LatentProbe, Probe
from neuroworld.train.readout import load_world_model
from neuroworld.utils.config import load_config

SPLITS = {  # Forecast splits, Updating splits, the split of the future-frames check
    "algonauts": (("val", "dev", "test"), ("val", "dev", "test"), "val"),
    "hcp_movies": (("val", "test", "heldout-adapt", "heldout-fit", "heldout-test"), ("val", "test", "heldout-adapt", "heldout-test"), "heldout-adapt"),
}


class Head:
    """The probe as the adapter's readout; before the probe exists it reads zeros."""
    subjects = []

    def __init__(self, probe=None):
        self.probe = probe

    def subject_indices(self, subject_ids, device=None):
        return list(subject_ids)

    def __call__(self, states, subjects):
        if self.probe is None:
            return torch.zeros(*states.shape[:-1], 1000, device=states.device)
        return torch.from_numpy(self.probe(states.float().cpu().numpy(), subjects)).to(states.device)


def _forget(adapter):
    adapter._memory, adapter._memory_bytes = OrderedDict(), 0


def latent_row(run_dir, name, seconds, dataset, store, device="cuda", log=print, data_config=None):
    """The model's benchmark row, `<name> (latent)`, with its probe fitted (or read from the store). data_config: the benchmark's
    dataset config (default the package's base.yaml); one with fewer training runs fits the probe and the gap on those only."""
    G.HISTORY_SECONDS = float(seconds)
    data = BenchmarkData(data_config or Path(neuroworld.__file__).parent / "configs" / dataset / "base.yaml")
    cfg = load_config(Path(run_dir) / "config.yaml")
    model, _ = load_world_model(cfg, Path(run_dir) / "best.pt", device)
    if dataset == "hcp_movies":
        # HCP's probe is pooled, so new viewers use it unchanged: every viewer is listed, and the benchmark's calibration
        # (called by State and Updating on held-out splits) finds none to give a head.
        Head.subjects = sorted({run.subject_id for split in ("train", "val", "heldout-fit", "heldout-adapt", "heldout-test")
                                for run in data.runs(split)})
    states = getattr(model, "read_states", cfg["model"]["context_states"]) * getattr(model, "state_step", 1)
    adapter = RolloutAdapter(f"{name} open-loop", model, Head(), cfg, device, states)
    origins = U.origins
    U.origins = lambda data, run: np.zeros(0, dtype=np.int64)
    first = Probe(data, adapter, store, log=log)
    U.origins = origins
    adapter.readout = Head(first)
    _forget(adapter)
    adapter.name = name
    probe = Probe(data, adapter, store, log=log)
    adapter.readout = Head(probe)
    _forget(adapter)
    return data, LatentProbe(adapter, probe, f"{name} (latent)")


def score(run_dir, seconds, dataset, store, *, name=None, gap=False, check=True, device="cuda", log=print, data_config=None):
    name = name or Path(run_dir).name
    data, row = latent_row(run_dir, name, seconds, dataset, store, device, log, data_config)
    forecast_splits, updating_splits, check_split = SPLITS[dataset]
    for split in forecast_splits:
        F.run(data, [row], split, output=store, log=log)
    _forget(row.model)
    S.run(data, [row], output=store, log=log)
    U.run(data, [row], splits=updating_splits, output=store, log=log)
    if check:
        N.run(data, row, output=store, split=check_split, log=log)
    if gap:
        F.run(data, [row], "train", output=store, log=log)
    log(f"{name} scored")
