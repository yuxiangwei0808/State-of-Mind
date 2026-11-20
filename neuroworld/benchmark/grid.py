"""The benchmark's time grid (docs/benchmark.md#how-a-model-is-queried), fixed in seconds: the next minute is
forecast, every query supplies up to two minutes of history (at least the last 30 s complete), origins every 3 s.

Step k is the target k*TR after the last observed frame; it belongs to band (a, b] when a < k*TR <= b.
A prefix of P seconds holds the frames with k*TR <= P. TR decides which steps fall in a band.
"""
from dataclasses import dataclass
import math
from pathlib import Path

EPS = 1e-6  # seconds; 20 * 1.5 must land on the edge at 30 s

BANDS = {"B1": (0.0, 7.5), "B2": (7.5, 30.0), "B3": (30.0, 60.0)}  # details only; the headline is the minute
PREFIXES = {"P30": 30.0}  # the last 30 s before an origin must be complete (and is the linear forecaster's window)
LONGEST_PREFIX = "P30"
HISTORY_SECONDS = 120.0   # history supplied with every query, up to two minutes; scoring a model with a longer window (the
# paper's 180 s) sets it to that window (neuroworld/spectator/score.py, experiments/history_length.LONGEST)


PROTOCOL = "2026-10-23"  # the scoring protocol: change it whenever a change can alter saved results


def code_digest():
    """A short digest of the benchmark code (every module of this package): two uncommitted versions differ in it."""
    import hashlib
    h = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()[:12]


def stamp(model=None):
    """{"protocol", "source", "code"}: written into every result file, so the card can flag results from different
    protocols or code versions (the commit, marked dirty when the working tree has uncommitted changes, and the digest
    of the benchmark code). A model with a fitted evaluation readout (a latent row's probe) adds its provenance."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        source = (commit + ("-dirty" if dirty else "")) or "unknown"
    except (OSError, subprocess.SubprocessError):
        source = "unknown"
    out = {"protocol": PROTOCOL, "source": source, "code": code_digest()}
    if getattr(model, "provenance", None):
        out["readout"] = model.provenance
    return out


def frames_within(seconds, tr):
    """Number of frames k >= 1 with k * tr <= seconds."""
    return int(math.floor(seconds / tr + EPS / tr))


def steps_in(band, tr, max_steps=None):
    """1-based steps of a band at this TR, optionally cut at max_steps."""
    low, high = BANDS[band]
    first, last = frames_within(low, tr) + 1, frames_within(high, tr)
    if max_steps is not None:
        last = min(last, max_steps)
    return range(first, last + 1)


def window_steps(low, high, tr, max_steps=None):
    """1-based steps k with low < k*tr <= high (a declared, possibly truncated window)."""
    first, last = frames_within(low, tr) + 1, frames_within(high, tr)
    if max_steps is not None:
        last = min(last, max_steps)
    return range(first, last + 1)


def prefix_frames(name, tr):
    return frames_within(PREFIXES[name], tr)


@dataclass(frozen=True)
class DatasetGrid:
    """Per-dataset grid: TR, maximum horizon (seconds) and origin stride."""
    dataset: str
    tr: float
    max_horizon_s: float = 60.0  # the next minute
    stride_s: float = 3.0

    @property
    def horizon_steps(self):
        return frames_within(self.max_horizon_s, self.tr)

    @property
    def stride_trs(self):
        return max(1, int(round(self.stride_s / self.tr)))

    def prefixes(self):
        return {name: prefix_frames(name, self.tr) for name in PREFIXES}

    def history_frames(self):
        return frames_within(HISTORY_SECONDS, self.tr)

    def bands(self, max_steps=None):
        """{band: steps} for the bands with at least three steps within max_steps (default horizon_steps)."""
        limit = self.horizon_steps if max_steps is None else max_steps
        out = {}
        for band in BANDS:
            steps = steps_in(band, self.tr, limit)
            full = steps_in(band, self.tr)
            # A declared truncation of at most 15% of the band width is allowed; otherwise n/a.
            if len(steps) >= 3 and (len(full) - len(steps)) * self.tr <= 0.15 * (BANDS[band][1] - BANDS[band][0]) + EPS:
                out[band] = steps
        return out


GRIDS = {
    "algonauts": DatasetGrid("algonauts", 1.49),
    "hcp_movies": DatasetGrid("hcp_movies", 1.0),
}
