"""neuroworld/paper/neuroworld_generated.py: NeuroWorld (k1 with history blocks, 180 s) trained with added windows whose past is
real fMRI and whose future is Spectator's forecast (the optional second student of
$NEUROWORLD_OUTPUT/paper-analysis/extra-nov13/synthetic-data-proposal.md).

    python -m neuroworld.paper.neuroworld_generated forecasts <k1 config> <out dir>     Spectator's forecasts of every training window's future

A k1 training window holds S = sequence_states frames, the first R = history_blocks * history_block_states + context_states of them
read as the past and the last F = S - R predicted (teacher-forced and rolled out). For every training window starting at frame s,
Spectator (ridge-jepa-180s-std-partial, its stored probe, as neuroworld/paper/generated.py's Teacher) forecasts frames s + R ...
s + S - 1 from the real fMRI of the 180 s before s + R and the movie. The forecasts are stored in the order of the training dataset's
windows, with that order, so training can check it reads the same windows.

Training (NeuroWorld's k1 training code, which is not part of this repository, with its build_dataset calling wrap()): with NEUROWORLD_GENERATED_FUTURES=<out dir>, the training split
holds every real window and, after them, a copy of each whose last F frames are the forecast (twice the windows per epoch). The
probe, the decoder and validation read real windows only.
"""
import sys
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ENV = "NEUROWORLD_GENERATED_FUTURES"


class GeneratedWindows(Dataset):
    """The real windows of a SchaeferWindowDataset, then the same windows with their last F frames replaced by the stored forecast."""

    def __init__(self, real, folder):
        self.real, self.folder = real, Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text())
        windows = np.load(self.folder / "windows.npy")  # [windows, 2]: index into run_ids, start frame
        if (meta["run_ids"] != [record["run_id"] for record in real.records] or windows.shape != real.windows.shape
                or not np.array_equal(windows, real.windows)):
            raise ValueError(f"{folder} holds forecasts of other windows than this training split")
        self.past, self.future = meta["past_frames"], meta["future_frames"]
        if self.past + self.future != real.length:
            raise ValueError(f"Forecast frames {self.past} + {self.future} do not fill windows of {real.length} frames")
        self.forecasts = np.load(self.folder / "forecasts.npy", mmap_mode="r")  # [windows, F, 1000] float32

    def __len__(self):
        return 2 * len(self.real)

    def __getattr__(self, name):  # the training loop reads the real dataset's settings (lag_trs, records, _arrays, ...)
        if name in ("real", "folder", "forecasts", "past", "future"):
            raise AttributeError(name)
        return getattr(self.real, name)

    def __getstate__(self):
        return {"real": self.real, "folder": self.folder, "past": self.past, "future": self.future}

    def __setstate__(self, state):  # DataLoader workers reopen the forecasts
        self.__dict__.update(state)
        self.forecasts = np.load(self.folder / "forecasts.npy", mmap_mode="r")

    def __getitem__(self, index):
        n = len(self.real)
        item = self.real[index % n]
        if index >= n:
            brain = item["brain"].clone()
            brain[self.past:] = torch.from_numpy(np.array(self.forecasts[index - n], dtype=np.float32))
            item["brain"] = brain
        return item


def wrap(dataset, split, purpose):
    """The training split with the generated windows added when NEUROWORLD_GENERATED_FUTURES names a forecast folder."""
    folder = os.environ.get(ENV)
    if not folder or split != "train" or purpose not in (None, "train"):
        return dataset
    return GeneratedWindows(dataset, folder)


def forecasts(config, out, device="cuda", teacher_arm="ridge-jepa-180s-std-partial", teacher_seconds=180.0):
    """Spectator's forecast of every training window's last F frames (module docstring), with its skill on them for scale."""
    from neuroworld.benchmark.data import BenchmarkData
    from neuroworld.paper.generated import CONFIGS, Teacher
    from neuroworld.train.stage_utils import build_dataset
    from neuroworld.utils.config import load_config
    log = lambda text: print(text, flush=True)
    cfg = load_config(config)
    m = cfg["model"]
    past = m["history_blocks"] * m["history_block_states"] + m["context_states"]
    real = build_dataset(cfg, "train", preload=False)
    future = real.length - past
    if m["state_window_trs"] != 1 or future <= 0:
        raise ValueError("Expected one frame per state and a future after the read states")
    data = BenchmarkData(CONFIGS["algonauts"])
    teacher = Teacher(teacher_arm, teacher_seconds, data, device)
    runs = {r.run_id: r for r in data.runs("train")}
    run_ids = [record["run_id"] for record in real.records]
    windows = np.asarray(real.windows, dtype=np.int64)  # [windows, 2]: index into run_ids, start frame
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(out / "forecasts.partial.npy", mode="w+", dtype=np.float32, shape=(len(windows), future, 1000))
    se = ref = 0.0
    for r, run_id in enumerate(run_ids):
        rows = np.flatnonzero(windows[:, 0] == r)
        if not len(rows):
            continue
        run = runs[run_id]
        origins = windows[rows, 1] + past
        pred = teacher.forecast(run, origins, future)
        store[rows] = pred
        frames = origins[:, None] + np.arange(future)
        real_future = np.asarray(run.brain[:], dtype=np.float32)[frames]
        se += float(((pred - real_future) ** 2).sum())
        ref += float(((real_future - real_future.reshape(-1, 1000).mean(0)) ** 2).sum())
        run.__dict__.pop("brain", None)
        if r % 100 == 0:
            log(f"{r}/{len(run_ids)} runs, {rows[-1] + 1} windows")
    store.flush()
    del store
    os.replace(out / "forecasts.partial.npy", out / "forecasts.npy")
    np.save(out / "windows.npy", windows)
    meta = {"config": str(config), "teacher": teacher_arm, "teacher_seconds": teacher_seconds, "past_frames": past, "future_frames": future,
            "windows": len(windows), "run_ids": run_ids,
            "teacher skill on the replaced frames (%, each run's mean as reference)": round(100 * (1 - se / ref), 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"forecasts of {len(windows)} windows ({future} frames each) in {out}; teacher skill on them {meta[list(meta)[-1]]}")


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):  # this script parses no options: show its usage and run nothing
        sys.exit(print(__doc__))
    import sys
    if sys.argv[1] != "forecasts":
        raise SystemExit(__doc__)
    forecasts(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or []))
