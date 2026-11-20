"""Native-TR training windows and the retained preparation smoke loader.

SchaeferV2SmokeDataset is an integrity check that filters whole-run rows.
SchaeferWindowDataset preserves gaps and explicitly aligns transition stimuli.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroworld.data.bundle import BundleValidationError, contained_path, read_jsonl
from neuroworld.nbs_data.bundle import (
    BRAIN_VIEW,
    BundlePaths,
    N_PARCELS,
    select_valid_time,
    validate_bundle,
)


class SchaeferV2SmokeDataset(Dataset[dict[str, Any]]):
    """Load whole valid-row fMRI tensors from a finalized v2 bundle."""

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        run_ids: Sequence[str] | None = None,
        max_runs: int | None = 1,
    ) -> None:
        self.paths = BundlePaths.from_root(bundle_root)
        self.manifest = validate_bundle(self.paths.root)
        records = read_jsonl(
            self.paths.runs_jsonl,
            required_keys={
                "run_id",
                "subject_id",
                "timeline_id",
                "tr_seconds",
                "time_origin_seconds",
                "brain_views",
                "masks",
            },
        )
        by_id = {str(record["run_id"]): record for record in records}
        if run_ids is not None:
            requested = list(run_ids)
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("run_ids must be a non-empty sequence of unique IDs")
            missing = [run_id for run_id in requested if run_id not in by_id]
            if missing:
                raise BundleValidationError(f"Unknown Schaefer-v2 run IDs: {missing}")
            records = [by_id[run_id] for run_id in requested]
        if max_runs is not None:
            if not isinstance(max_runs, int) or isinstance(max_runs, bool) or max_runs <= 0:
                raise ValueError("max_runs must be a positive integer or None")
            records = records[:max_runs]
        if not records:
            raise BundleValidationError("The Schaefer-v2 smoke dataset has no runs")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        brain_record = record["brain_views"][BRAIN_VIEW]
        valid_record = record["masks"]["valid_time"]
        brain_path = contained_path(
            self.paths.root,
            brain_record["path"],
            must_exist=True,
            kind="file",
            suffix=".npy",
        )
        valid_path = contained_path(
            self.paths.root,
            valid_record["path"],
            must_exist=True,
            kind="file",
            suffix=".npy",
        )
        brain = np.load(brain_path, allow_pickle=False, mmap_mode="r")
        valid_time = np.load(valid_path, allow_pickle=False, mmap_mode="r")
        y = select_valid_time(brain, valid_time)
        if y.ndim != 2 or y.shape[1] != N_PARCELS or y.shape[0] == 0:
            raise BundleValidationError(
                f"Run {record['run_id']!r} has no valid [T,{N_PARCELS}] samples"
            )
        tr_seconds = float(record["tr_seconds"])
        origin = float(record["time_origin_seconds"])
        physical_time = origin + np.arange(brain.shape[0], dtype=np.float64) * tr_seconds
        physical_time = select_valid_time(physical_time, valid_time)
        return {
            "y": torch.from_numpy(y),
            "physical_time_s": torch.from_numpy(physical_time),
            "run_id": str(record["run_id"]),
            "subject_id": str(record["subject_id"]),
            "timeline_id": str(record["timeline_id"]),
        }


def build_schaefer_v2_smoke_loader(
    bundle_root: str | Path,
    *,
    run_id: str | None = None,
) -> DataLoader[dict[str, Any]]:
    """Return the fixed one-run loader used by bundle acceptance tests."""

    dataset = SchaeferV2SmokeDataset(
        bundle_root,
        run_ids=None if run_id is None else [run_id],
        max_runs=1,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


__all__ = ["SchaeferV2SmokeDataset", "build_schaefer_v2_smoke_loader"]


class SchaeferWindowDataset(Dataset):
    """Continuous native-TR windows of S states for latent training and rollout.

    For K state frames and S states, brain has K+S-1 rows and encoding it gives
    S states. Stimulus row r conditions the transition from state r to r+1: its
    source frame is start+K+r-lag. Thus predicting brain frame j uses stimulus
    frame j-lag, exactly once. Training uses S=sequence_states; evaluation and
    readout training use S=prefix+horizon. Windows never cross invalid frames.

    ``context_features`` (a directory prepared by nbs_data/stimuli/tribev2_context.py) replaces
    the per-modality rows with one precomputed row per TR, ``stimuli["context"]``; windows are
    selected exactly as without it.

    ``frame_ranges`` ({run_id: [(start, stop), ...]}) keeps only windows whose brain frames and
    stimulus rows lie inside one listed range of their run (data/splits.py); unlisted runs are whole.

    ``history_fill_frames`` (F > 0) lets the oldest F frames of each window be missing, as a benchmark query's prefix may be:
    a long-history encoder's window then needs valid frames, inside the split, only where some state's newest K - F frames
    lie (frames start + F onward). In the oldest F frames, a frame enters as zero (the z-scored training mean, as the
    benchmark enters it) unless it passes motion QC, is not non-steady (masks ``motion_qc``/``nonsteady``; else
    ``valid_time``) and lies inside the split's frame ranges, so no window reads another split's brain frames.
    """

    def __init__(self, bundle_root, run_ids, *, tr_seconds, state_window_trs=1,
                 num_states=32, hrf_delay_seconds=5.0, window_stride_trs=1,
                 include_stimulus=True, preload=False, stimulus_context_trs=1, context_features=None, context_width=2048,
                 frame_ranges=None, history_fill_frames=0):
        from fnmatch import fnmatchcase
        from neuroworld.data.bundle import read_json_object
        from neuroworld.nbs_data.bundle import DATASET_TR_SECONDS, MODALITY_DIMS

        for name, value in [("state_window_trs", state_window_trs), ("num_states", num_states),
                            ("window_stride_trs", window_stride_trs)]:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(history_fill_frames) is not int or not 0 <= history_fill_frames < state_window_trs:
            raise ValueError("history_fill_frames must be an integer from 0 to state_window_trs - 1")
        self.history_fill_frames = history_fill_frames
        if num_states < 2:
            raise ValueError("num_states must be at least 2 (one observed state and one target)")
        if not np.isfinite(tr_seconds) or tr_seconds <= 0:
            raise ValueError("tr_seconds must be positive and finite")
        if not np.isfinite(hrf_delay_seconds) or hrf_delay_seconds < 0:
            raise ValueError("hrf_delay_seconds must be non-negative and finite")
        if not run_ids or isinstance(run_ids, str):
            raise ValueError("run_ids must be a non-empty list of IDs or glob patterns")
        self.paths = BundlePaths.from_root(bundle_root)
        manifest = read_json_object(self.paths.bundle_json)
        if manifest.get("layout_id") != "neuroworld-schaefer1000-v2" or manifest.get("n_parcels") != 1000:
            raise ValueError("Expected a prepared native-TR Schaefer-1000 v2 bundle")
        self.dataset_id = manifest["dataset_id"]
        expected_tr = DATASET_TR_SECONDS[self.dataset_id]
        if not np.isclose(tr_seconds, expected_tr, rtol=0, atol=1e-8):
            raise ValueError(f"Configured TR {tr_seconds} differs from {self.dataset_id} TR {expected_tr}")
        self.tr_seconds = float(tr_seconds)
        self.state_window_trs = state_window_trs
        self.num_states = num_states
        self.length = state_window_trs + num_states - 1
        if type(stimulus_context_trs) is not int or stimulus_context_trs < 1:
            raise ValueError("stimulus_context_trs must be a positive integer")
        self.stimulus_context_trs = stimulus_context_trs
        # One row per transition, plus N-1 older rows so each transition has an N-row window.
        self.stimulus_length = num_states - 1 + stimulus_context_trs - 1
        self.include_stimulus = include_stimulus
        self.lag_trs = int(np.floor(hrf_delay_seconds / tr_seconds + 0.5)) if include_stimulus else 0
        self.effective_hrf_delay_seconds = self.lag_trs * self.tr_seconds

        records = read_jsonl(self.paths.runs_jsonl)
        ids = {r["run_id"] for r in records}
        selected = set()
        for pattern in run_ids:
            if any(character in pattern for character in "*?["):
                matches = {i for i in ids if fnmatchcase(i, pattern)}
            else:
                matches = {pattern} & ids
            if not matches:
                raise ValueError(f"Run selection matches nothing: {pattern}")
            selected.update(matches)
        self.run_ids = sorted(selected)
        feature_records = {}
        if include_stimulus:
            feature_path = contained_path(self.paths.root, manifest["artifacts"]["features"]["path"], must_exist=True)
            for feature in read_jsonl(feature_path):
                timeline = feature["timeline_id"]
                if timeline in feature_records:
                    raise ValueError(f"Duplicate feature timeline: {timeline}")
                feature_records[timeline] = feature
        context_rows = None
        if include_stimulus and context_features is not None:
            context_root = Path(context_features)
            context_index = read_json_object(context_root / "index.json")
            if context_index.get("format") != "neuroworld-stimulus-context-v1" or context_index.get("dim") != context_width:
                raise ValueError(f"{context_root} is not a {context_width}-wide stimulus context directory")
            if context_index.get("dataset_id") != self.dataset_id:
                raise ValueError(f"{context_root} was prepared for {context_index.get('dataset_id')}, not {self.dataset_id}")
            context_rows = context_index["timelines"]
        self.records = []
        windows = []
        for record in records:
            if record["run_id"] not in selected:
                continue
            n = record["n_timepoints"]
            if not np.isclose(record["tr_seconds"], tr_seconds, rtol=0, atol=1e-8):
                raise ValueError(f"Run TR mismatch: {record['run_id']}")
            brain_path = self._array_path(record["brain_views"][BRAIN_VIEW], (n, 1000), np.float32)
            valid_path = self._array_path(record["masks"]["valid_time"], (n,), np.bool_)
            valid = np.load(valid_path, allow_pickle=False, mmap_mode="r")
            modalities = {}
            if include_stimulus:
                feature = feature_records.get(record["timeline_id"])
                if feature is None:
                    raise ValueError(f"Missing stimulus timeline: {record['timeline_id']}")
                for key in ("n_timepoints", "tr_seconds", "time_origin_seconds"):
                    if not np.isclose(feature[key], record[key], rtol=0, atol=1e-8):
                        raise ValueError(f"Stimulus {key} mismatch for {record['run_id']}")
                if feature.get("normalization") != "none" or feature.get("pooling", {}).get("hemodynamic_lag_seconds") not in (None, 0):
                    raise ValueError("Expected unnormalized stimulus features with no existing HRF shift")
                from neuroworld.nbs_data.bundle import DATASET_MODALITIES
                if set(feature["modalities"]) != DATASET_MODALITIES[self.dataset_id]:
                    raise ValueError(f"Unexpected modalities for {self.dataset_id}")
                for name, entry in feature["modalities"].items():
                    modalities[name] = self._array_path(entry, (n, 2, MODALITY_DIMS[name]), np.float32)
                if context_rows is not None:
                    entry = context_rows.get(record["timeline_id"])
                    if entry is None or entry["n_timepoints"] != n:
                        raise ValueError(f"No {n}-row stimulus context for {record['timeline_id']} in {context_root}")
                    path = (context_root / entry["path"]).resolve()
                    rows = np.load(path, mmap_mode="r")
                    if rows.shape != (n, context_width) or rows.dtype != np.float32:
                        raise ValueError(f"{path} must hold float32 rows of shape {(n, context_width)}")
                    modalities = {"context": path}
                support_path = self._array_path(record["masks"]["stimulus_support"], (n,), np.bool_)
                support = np.load(support_path, allow_pickle=False, mmap_mode="r")
            starts = np.arange(0, max(0, n - self.length + 1), window_stride_trs, dtype=np.int64)
            bad = np.r_[0, np.cumsum(~valid)]
            fill = history_fill_frames
            starts = starts[bad[starts + self.length] == bad[starts + fill]]
            if include_stimulus:
                first = starts + state_window_trs - self.lag_trs - (stimulus_context_trs - 1)
                inside = (first >= 0) & (first + self.stimulus_length <= n)
                starts, first = starts[inside], first[inside]
                bad_support = np.r_[0, np.cumsum(~support)]
                keep = bad_support[first + self.stimulus_length] == bad_support[first]
                starts, first = starts[keep], first[keep]
            ranges = (frame_ranges or {}).get(record["run_id"])
            if ranges:
                low, high = starts + fill, starts + self.length
                if include_stimulus:
                    low, high = np.minimum(low, first), np.maximum(high, first + self.stimulus_length)
                inside = np.zeros(len(starts), dtype=bool)
                for range_start, range_stop in ranges:
                    inside |= (low >= range_start) & (high <= range_stop)
                starts = starts[inside]
            index = len(self.records)
            history_ok = None
            if fill:
                masks = record["masks"]
                if "motion_qc" in masks and "nonsteady" in masks:
                    history_ok = (np.load(self._array_path(masks["motion_qc"], (n,), np.bool_), allow_pickle=False)
                                  & ~np.load(self._array_path(masks["nonsteady"], (n,), np.bool_), allow_pickle=False))
                else:
                    history_ok = np.asarray(valid, dtype=bool).copy()
                if ranges:
                    inside_split = np.zeros(n, dtype=bool)
                    for range_start, range_stop in ranges:
                        inside_split[range_start:range_stop] = True
                    history_ok &= inside_split
            self.records.append({"run_id": record["run_id"], "subject_id": record["subject_id"],
                                 "timeline_id": record["timeline_id"], "brain": brain_path,
                                 "valid": valid_path, "modalities": modalities,
                                 "origin": float(record["time_origin_seconds"]), "history_ok": history_ok})
            if len(starts):
                windows.append(np.column_stack((np.full_like(starts, index), starts)))
        if not windows:
            raise ValueError(f"No valid windows for {self.dataset_id}: K={state_window_trs}, "
                             f"S={num_states}, lag={self.lag_trs} TRs")
        self.windows = np.concatenate(windows)
        if preload:
            # Loaded before DataLoader workers fork, so they share these arrays.
            used = {self.records[i]["brain"] for i in np.unique(self.windows[:, 0])}
            used |= {path for i in np.unique(self.windows[:, 0])
                     for path in self.records[i]["modalities"].values()}
            self._arrays = {path: np.load(path, allow_pickle=False) for path in sorted(used)}

    @property
    def subject_ids(self):
        return sorted({record["subject_id"] for record in self.records})

    def _array_path(self, entry, shape, dtype):
        path = contained_path(self.paths.root, entry["path"], must_exist=True, kind="file", suffix=".npy")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Unexpected shape or dtype at {path}: {array.shape}, {array.dtype}")
        return path

    def __len__(self):
        return len(self.windows)

    def _open(self, path):
        """Memory-map each array once per process (or use preloaded arrays); reopening per window
        dominates load time on NFS."""
        cache = self.__dict__.setdefault("_arrays", {})
        if path not in cache:
            cache[path] = np.load(path, allow_pickle=False, mmap_mode="r")
        return cache[path]

    def __getstate__(self):
        # DataLoader workers reopen their own maps.
        return {key: value for key, value in self.__dict__.items() if key != "_arrays"}

    def __getitem__(self, index):
        run_index, start = map(int, self.windows[index])
        record = self.records[run_index]
        brain = self._open(record["brain"])
        frames = np.arange(start, start + self.length, dtype=np.int64)
        block = np.array(brain[start:start + self.length], copy=True)
        if self.history_fill_frames:
            fill = self.history_fill_frames
            block[:fill][~record["history_ok"][start:start + fill]] = 0.0
        result = {
            "brain": torch.from_numpy(block),
            "stimuli": {},
            "frame_indices": torch.from_numpy(frames),
            "state_frame_indices": torch.from_numpy(frames[self.state_window_trs - 1:].copy()),
            "physical_time_s": torch.from_numpy(record["origin"] + frames * self.tr_seconds),
            "tr_seconds": self.tr_seconds,
            "run_id": record["run_id"], "subject_id": record["subject_id"],
            "timeline_id": record["timeline_id"],
        }
        if self.include_stimulus:
            first = start + self.state_window_trs - self.lag_trs - (self.stimulus_context_trs - 1)
            stop = first + self.stimulus_length
            result["stimulus_frame_indices"] = torch.arange(first, stop)
            for name, path in record["modalities"].items():
                array = self._open(path)
                result["stimuli"][name] = torch.from_numpy(np.array(array[first:stop], copy=True))
        return result


__all__.append("SchaeferWindowDataset")
