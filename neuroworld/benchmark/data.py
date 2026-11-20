"""Benchmark view of a prepared bundle: runs, masks, splits, exposure tags and repeated segments.

Brain frames are valid when they pass motion QC and are not non-steady (stimulus support is a
separate mask: HCP rests have no stimulus features, but their brain frames may serve as prefix).
Targets must lie inside one stimulus-supported segment and inside the split's frame ranges.
Prefix frames are inputs, so they may come from any frames of the run.
"""
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import re

import numpy as np

from neuroworld.benchmark.grid import GRIDS
from neuroworld.data.splits import SPLITS, check_splits, content_key, frame_ranges, split_runs
from neuroworld.utils.config import load_config

# Standard-split membership -> exposure tag for models trained with that split.
EXPOSURE = {"train": "seen in fitting", "val": "seen in selection", "dev": "seen in selection",
            "test": "unseen", "heldout": "unseen", "heldout-fit": "adaptation", "heldout-adapt": "unseen",
            "heldout-test": "unseen"}
# New-participant adaptation (HCP): held-out participants' run 1 adapts a readout head, runs 2-3 are scored.
ADAPTATION_RUN = "/run-01/"
HCP_RETEST_TRS = (83, 85)  # the 84-s test-retest clip ends every HCP run


def segments(mask):
    """[(start, stop), ...] of the True runs of a boolean vector."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def content_group(record):
    """Bootstrap content unit: a Friends episode (both halves) or a Movie10/HCP content part."""
    return re.sub(r"(friends:s\d{2}e\d{2})[a-z]$", r"\1", content_key(record["timeline_id"]))


@dataclass
class Run:
    run_id: str
    subject_id: str
    timeline_id: str
    split: str
    content: str
    group: str
    n: int
    brain_path: Path
    brain_ok: np.ndarray   # [n] valid brain frame
    support: np.ndarray    # [n] stimulus features present
    allowed: np.ndarray    # [n] frames the split may score
    stimulus: str = ""     # the stimulus handle queries carry, when it differs from the timeline (repeated viewings)

    @property
    def stimulus_id(self):
        """The stimulus handle queries carry: one per movie, shared by repeated viewings."""
        return self.stimulus or self.timeline_id

    @property
    def query_id(self):
        """An opaque handle for the run in queries, so a model cannot read the run or repetition from it."""
        import hashlib
        return "q" + hashlib.sha1(self.run_id.encode()).hexdigest()[:12]

    @property
    def exposure(self):
        return EXPOSURE[self.split]

    @cached_property
    def brain(self):
        return np.load(self.brain_path, mmap_mode="r", allow_pickle=False)

    @cached_property
    def target_ok(self):
        """Frames that may be targets: stimulus-supported and inside the split's ranges."""
        return self.support & self.allowed

    @cached_property
    def _bad_brain(self):
        return np.r_[0, np.cumsum(~self.brain_ok)]

    @cached_property
    def _bad_target(self):
        return np.r_[0, np.cumsum(~self.target_ok)]

    def prefix_complete(self, t0, frames):
        """Frames t0-frames..t0-1 exist and are all valid brain frames."""
        return t0 - frames >= 0 and self._bad_brain[t0] == self._bad_brain[t0 - frames]

    def target_span_ok(self, t0, steps):
        """Targets t0..t0+steps-1 lie in the run, inside one supported segment and the split's frames."""
        return t0 + steps <= self.n and self._bad_target[t0 + steps] == self._bad_target[t0]


class BenchmarkData:
    """Runs of one dataset (from its base config) by split, with the benchmark grid."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.bundle = Path(self.cfg["data"]["bundle_path"])
        manifest = json.loads((self.bundle / "bundle.json").read_text())
        self.dataset = manifest["dataset_id"]
        self.grid = GRIDS[self.dataset]
        self.records = {r["run_id"]: r for r in check_splits(self.cfg)}

    def bootstrap_unit(self, run):
        """Resampling unit: content parts on Algonauts (4 participants), participants on HCP."""
        return run.subject_id if self.dataset == "hcp_movies" else run.group

    def runs(self, split):
        if split in ("heldout", "heldout-fit", "heldout-adapt", "heldout-test"):
            # New participants: run 1 adapts a head (heldout-fit), runs 2-3 are scored (heldout-adapt), and the runs
            # of the test split (HCP run 4) are scored too (heldout-test). Every split but train is test data.
            heldout = set(self.cfg["data"]["heldout_subjects"])
            test = set(r["run_id"] for r in self._split_records("test", include_heldout=True))
            records = [r for r in self.records.values() if r["subject_id"] in heldout
                       and (r["run_id"] in test) == (split == "heldout-test")]
            if split == "heldout-fit":
                records = [r for r in records if ADAPTATION_RUN in r["run_id"]]
            elif split == "heldout-adapt":
                adapted = {r["subject_id"] for r in records if ADAPTATION_RUN in r["run_id"]}
                records = [r for r in records if ADAPTATION_RUN not in r["run_id"] and r["subject_id"] in adapted]
        elif split in SPLITS:
            records = split_runs(self.cfg, split, list(self.records.values()))
        else:
            raise ValueError(f"Unknown split {split!r}")
        return [self._run(record, split) for record in records]

    def _split_records(self, split, include_heldout=False):
        if not self.cfg["data"][f"{split}_runs"]:
            return []
        if not include_heldout:
            return split_runs(self.cfg, split, list(self.records.values()))
        cfg = {**self.cfg, "data": {**self.cfg["data"], "heldout_subjects": []}}
        return split_runs(cfg, split, list(self.records.values()))

    def _run(self, record, split):
        masks = {name: np.load(self.bundle / entry["path"], allow_pickle=False).astype(bool)
                 for name, entry in record["masks"].items()}
        brain_ok = masks["motion_qc"] & ~masks["nonsteady"] if "motion_qc" in masks else masks["valid_time"].copy()
        n = int(record["n_timepoints"])
        allowed = np.ones(n, dtype=bool)
        # Held-out participants use their non-test runs whole and their test runs as the test split does; other
        # splits keep their frame ranges.
        if split == "heldout-test":
            ranges = frame_ranges(self.cfg, "test", record["run_id"])
        else:
            ranges = None if split.startswith("heldout") else frame_ranges(self.cfg, split, record["run_id"])
        if ranges:
            allowed[:] = False
            for start, stop in ranges:
                allowed[start:stop] = True
        return Run(run_id=record["run_id"], subject_id=record["subject_id"], timeline_id=record["timeline_id"],
                   split=split, content=content_key(record["timeline_id"]), group=content_group(record), n=n,
                   brain_path=self.bundle / record["brain_views"]["projected_native"]["path"],
                   brain_ok=brain_ok, support=masks["stimulus_support"], allowed=allowed,
                   stimulus=self._stimulus_id(record["timeline_id"]))

    def _stimulus_id(self, timeline):
        """A repeated viewing (Movie10 "<part>_run-2") reads the same stimulus as the first viewing, under the first
        viewing's handle, so the handle does not reveal which viewing a query is from."""
        if timeline.endswith("_run-2"):
            first = timeline[: -len("_run-2")]
            if first in self.timelines:
                return first
        return timeline

    @cached_property
    def timelines(self):
        return {r["timeline_id"] for r in self.records.values()}

    def repeated_segments(self, run):
        """[(content_id, start, stop)] segments of this run that the same participant saw more than once.

        Algonauts: a Movie10 part watched twice is a whole run (its content key is shared by both viewings).
        HCP: the 84-s test-retest clip, the last supported segment of every run.
        """
        if self.dataset == "hcp_movies":
            parts = segments(run.support)
            if parts and HCP_RETEST_TRS[0] <= parts[-1][1] - parts[-1][0] <= HCP_RETEST_TRS[1]:
                return [("hcp7t:retest", *parts[-1])]
            return []
        return [(run.content, 0, run.n)]
