"""Benchmark adapters for this project's trained models (submissions, not references).

A rollout model needs K + H0 - 1 observed frames (K frames per state, H0 initial states) and one stimulus row
per transition, built exactly as data/schaefer_v2_dataset.py builds training windows: the row predicting
frame j is stimulus frame j - lag. A query's prefix fills the newest frames; frames the query does not give
(a shorter prefix) or that are invalid enter as zero, the z-scored training mean. Longer prefixes are cut to
what the model reads, so these models carry no memory beyond their window.

    load_run(run_dir)            a world model or forecaster with its readout, or the fMRI forecaster
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from neuroworld.benchmark.interface import stimulus_source
from neuroworld.data.bundle import read_jsonl

MAX_OPEN_ARRAYS = 48
MEMORY_BYTES = 2 ** 30  # earlier rollouts a RolloutAdapter keeps for reuse (least recently used dropped first)


class StimulusWindows:
    """Per-timeline stimulus arrays of a model config: bundle modalities or precomputed context rows."""

    def __init__(self, cfg):
        bundle = Path(cfg["data"]["bundle_path"])
        manifest = json.loads((bundle / "bundle.json").read_text())
        tr = float(cfg["data"]["tr_seconds"])
        self.lag = int(np.floor(cfg["data"]["hrf_delay_seconds"] / tr + 0.5))
        self.context = int(cfg["model"].get("stimulus_context_trs", 1))
        self.paths = {}
        if cfg["model"].get("stimulus_features", "projections") == "tribev2_context":
            root = Path(cfg["data"]["context_features"])
            index = json.loads((root / "index.json").read_text())["timelines"]
            self.paths = {timeline: {"context": root / entry["path"]} for timeline, entry in index.items()}
        else:
            for feature in read_jsonl(bundle / manifest["artifacts"]["features"]["path"]):
                self.paths[feature["timeline_id"]] = {name: bundle / entry["path"]
                                                      for name, entry in feature["modalities"].items()}
        self._arrays = OrderedDict()

    def _array(self, path):
        # A small LRU of memory maps: each open map holds a file descriptor (three modalities x hundreds of runs
        # would pass the usual 1024 limit).
        if path in self._arrays:
            self._arrays.move_to_end(path)
        else:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            while len(self._arrays) > MAX_OPEN_ARRAYS:
                self._arrays.popitem(last=False)
        return self._arrays[path]

    def rows(self, timeline_id, first, length, source=None):
        """{name: [length, ...]} rows first .. first+length-1, zero outside the recording. ``source(frames,
        modality)`` maps the stimulus frames to (timelines, frames) for a remapped query, per modality (None for
        fused contextual rows)."""
        frames = np.arange(first, first + length)
        names = list(self.paths[timeline_id])
        out = {}
        for name in names:
            modality = None if name == "context" else name
            timelines, mapped = source(frames, modality) if source else (np.full(length, timeline_id, dtype=object), frames)
            for timeline in dict.fromkeys(timelines):
                take = np.flatnonzero(timelines == timeline)
                array = self._array(self.paths[timeline][name])
                block = out.setdefault(name, np.zeros((length, *array.shape[1:]), dtype=np.float32))
                inside = take[(mapped[take] >= 0) & (mapped[take] < len(array))]
                block[inside] = array[mapped[inside]]
        return out


class RolloutAdapter:
    def __init__(self, name, model, readout, cfg, device, prefix_states):
        self.name, self.model, self.readout, self.device = name, model, readout, device
        self.state_frames = model.encoder.state_window_trs
        self.prefix_states = prefix_states
        self.frames = self.state_frames + prefix_states - 1
        self.stimulus = StimulusWindows(cfg)
        self.forecaster = getattr(model, "forecasts_fmri", False)
        self._memory, self._memory_bytes = OrderedDict(), 0

    @torch.no_grad()
    def predict(self, batch):
        return self.predict_states(batch)[0]

    def _fill_gaps(self, batch):
        """Missing prefix frames (Updating reveals only some) are filled with the model's own decoded forecasts: each
        gap is rolled out from the frames before it, observed or already filled, and the filled frames are re-encoded
        (observation imputation, not native latent propagation: the decoder shapes the later states). Each query is
        filled from its own measurements, whatever else is in the batch. The returned prefix starts at the oldest
        frame that can still reach the model."""
        from dataclasses import replace
        valid, need = batch.prefix_valid, self.frames
        if valid[:, -need:].all() or batch.splices is not None or batch.histories is not None:
            return batch
        width, masks, groups = valid.shape[1], {}, {}
        for i, row in enumerate(valid):
            masks.setdefault(row.tobytes(), []).append(i)
        for rows in masks.values():
            missing = ~valid[rows[0]]
            if not missing[-need:].any():
                continue  # the frames the model reads are all observed
            # Start from the newest fully observed window: the model reads only its newest frames, so older gaps
            # (such as frames before the run start in a long history) do not matter, and queries that differ only
            # there are filled together. A prefix shorter than the frames the model reads starts at its first frame.
            j = max(0, width - need)
            while j > need and missing[j - need:j].any():
                j -= 1
            groups.setdefault(missing[max(0, j - need):].tobytes(), (j, []))[1].extend(rows)
        lo = max(0, min(j for j, _ in groups.values()) - need)
        prefix = np.where(valid[:, lo:, None], np.nan_to_num(batch.prefix[:, lo:]), 0.0).astype(np.float32)
        filled = valid[:, lo:].copy()  # observed, or filled below
        for j, rows in groups.values():
            rows, missing = sorted(rows), ~valid[rows[0]]
            part = replace(batch, prefix=prefix, prefix_valid=valid[:, lo:]).take(rows)
            while j < width:
                if not missing[j]:
                    j += 1
                    continue
                e = j
                while e + 1 < width and missing[e + 1]:
                    e += 1
                first = max(lo, j - need)  # the frames the gap's rollout reads
                if j > first:  # a gap at the prefix's first frame (a run start) has nothing to roll out from
                    prefix[rows, j - lo:e + 1 - lo] = self._remembered(replace(
                        part, origins=part.origins - width + j, prefix=prefix[rows, first - lo:j - lo],
                        prefix_valid=np.ones((len(rows), j - first), dtype=bool), horizon=e - j + 1))[0]
                    filled[np.ix_(rows, np.arange(j - lo, e + 1 - lo))] = True
                j = e + 1
        return replace(batch, prefix=prefix, prefix_valid=filled)

    @torch.no_grad()
    def predict_states(self, batch):
        """([B, horizon, 1000] fMRI, [B, horizon, D] rolled latent states, or None for an fMRI forecaster)."""
        return self._remembered(self._fill_gaps(batch))

    def _remembered(self, batch):
        """_rollout, reusing an earlier rollout of a query with the same run, participant, movie, origin and input
        frames that went at least as far ahead (the rollout is causal, so a shorter one is its first steps). At each
        reveal, Updating asks again about the whole hidden period so far; this keeps its cost linear in the reveals."""
        if batch.splices is not None or batch.histories is not None:
            return self._rollout(batch)
        window = self._window(batch)
        # Overlapping episodes of a run reach the same frame from different histories: the input frames are in the key.
        keys = [(*key, hash(w.tobytes())) for key, w in zip(zip(batch.run_ids, batch.subject_ids, batch.timeline_ids,
                                                                batch.origins.tolist()), window)]
        found = []
        for key, w in zip(keys, window):
            entry = self._memory.get(key)
            if entry is not None and len(entry[1]) >= batch.horizon and np.array_equal(entry[0], w):
                self._memory.move_to_end(key)
            else:
                entry = None
            found.append(entry)
        miss = [i for i, entry in enumerate(found) if entry is None]
        if miss:
            fmri, states = self._rollout(batch.take(miss))
            for n, i in enumerate(miss):
                found[i] = (window[i].copy(), fmri[n].copy(), None if states is None else states[n].copy())
                self._remember(keys[i], found[i])
        h = batch.horizon
        return (np.stack([entry[1][:h] for entry in found]),
                None if found[0][2] is None else np.stack([entry[2][:h] for entry in found]))

    def _remember(self, key, entry):
        size = lambda entry: sum(a.nbytes for a in entry if a is not None)
        if key in self._memory:
            self._memory_bytes -= size(self._memory.pop(key))
        self._memory[key] = entry
        self._memory_bytes += size(entry)
        while self._memory_bytes > MEMORY_BYTES:
            self._memory_bytes -= size(self._memory.popitem(last=False)[1])

    def _window(self, batch):
        """[B, frames, parcels] the newest prefix frames the model reads (invalid or absent frames as zero)."""
        need = self.frames
        window = np.zeros((len(batch), need, batch.prefix.shape[2]), dtype=np.float32)
        take = min(need, batch.prefix.shape[1])
        if take:
            window[:, need - take:] = np.where(batch.prefix_valid[:, -take:, None], batch.prefix[:, -take:], 0.0)
        return window

    def _rollout(self, batch):
        horizon, need = batch.horizon, self.frames
        prefix = self._window(batch)
        # Window start s = t0 - need; the first stimulus row is s + K - lag - (context - 1).
        first = batch.origins - need + self.state_frames - self.stimulus.lag - (self.stimulus.context - 1)
        length = self.prefix_states + horizon - 1 + self.stimulus.context - 1
        rows = [self.stimulus.rows(batch.timeline_ids[i], int(first[i]), length,
                                   (lambda f, m, i=i: stimulus_source(batch, i, f, m)) if batch.remapped(i) else None)
                for i in range(len(batch))]
        stimuli = {name: torch.from_numpy(np.stack([r[name] for r in rows])).to(self.device) for name in rows[0]}
        brain = torch.from_numpy(prefix).to(self.device)
        head = self.model.readout if self.forecaster else self.readout
        subjects = head.subject_indices(list(batch.subject_ids), self.device)
        if self.forecaster:
            fmri = self.model.rollout_frames(brain, stimuli, subjects, horizon_trs=horizon)
            return fmri.float().cpu().numpy(), None
        states = self.model.rollout(brain, stimuli, horizon_trs=horizon)
        fmri = self.readout(states, subjects)
        return fmri.float().cpu().numpy(), states.float().cpu().numpy()


class ProbeHead:
    """In a RolloutAdapter's decoder place, with no decoder: predicted states to parcels through the run's frozen latent probe, or
    zeros before there is a probe (open-loop rollouts do not use them)."""

    def __init__(self, probe=None):
        self.probe = probe

    def subject_indices(self, subject_ids, device=None):
        return list(subject_ids)

    def __call__(self, states, subjects):
        if self.probe is None:
            return torch.zeros(*states.shape[:-1], 1000, device=states.device)
        return torch.from_numpy(self.probe(states.float().cpu().numpy(), subjects)).to(states.device)


class ProbeFillAdapter(RolloutAdapter):
    """A latent model with no decoder: Updating's hidden frames are its predicted states read by its own frozen probe, then re-encoded as
    usual. So every task reads only the latent. benchmark.__main__ fits the probe in two steps: on open-loop rollouts, then on rollouts
    with measurements filled by the first."""
    decodes = False
    fills_with_probe = True

    def use_probe(self, probe):
        self.readout = ProbeHead(probe)
        self._memory, self._memory_bytes = OrderedDict(), 0  # remembered rollouts hold frames decoded by the last head



def load_run(run_dir, device="cpu", name=None):
    from neuroworld.train.readout import load_readout, load_world_model
    from neuroworld.utils.config import load_config
    run = Path(run_dir)
    cfg = load_config(run / "config.yaml")
    model, _ = load_world_model(cfg, run / "best.pt", device)
    readout = None
    if not getattr(model, "forecasts_fmri", False):
        if not (run / "readout" / "readout_best.pt").is_file():
            return ProbeFillAdapter(name or run.name, model, ProbeHead(), cfg, device, getattr(model, "read_states", cfg["model"]["context_states"])
                                    * getattr(model, "state_step", 1))
        readout, _ = load_readout(run / "readout" / "readout_best.pt", device, world_model_path=run / "best.pt")
    # Give every model the initial states it was trained to use (its context), as for the aligned references; with a
    # state step s, each of its s chains reads that many states s frames apart, so it gets s times as many.
    return RolloutAdapter(name or run.name, model, readout, cfg, device,
                          getattr(model, "read_states", cfg["model"]["context_states"])
                          * getattr(model, "state_step", 1))

