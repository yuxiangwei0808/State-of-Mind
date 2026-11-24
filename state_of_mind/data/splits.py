"""Data splits: which runs, and which frames of them, each split uses.

A split (``train``, ``val``, ``dev``, ``test``) selects runs by ID or glob pattern
(``data.<split>_runs``), minus the subjects reserved in ``data.heldout_subjects``. Only ``train``
and ``val`` are required. ``data.split_frames`` may restrict a split to frame ranges of its runs,
``{split: {run pattern: [[start, stop], ...]}}``; every window of that split then lies inside one
range (its brain frames and its stimulus rows). Two splits may share a run only with disjoint frame
ranges, which ``check_splits`` enforces.

``content_key`` names what a run showed, shared by every subject and every viewing of it. It groups
runs for the inter-subject references, the world-model controls, Cls@Top10 and the bootstrap.
"""
from fnmatch import fnmatchcase
from functools import lru_cache
import json
from pathlib import Path
import re

SPLITS = ("train", "val", "dev", "test")


def content_key(timeline_id):
    """Stimulus content of a timeline, the same for every subject, viewing and release version.

    Strips Algonauts' per-subject prefix (``sub-01:``), the suffix of a repeated Movie10 viewing
    (``figures01_run-2``) and the HCP stimulus release (``hcp7t:pre:``/``hcp7t:post:`` show the
    same clips on the same clock).
    """
    key = re.sub(r"^sub-[^:]+:", "", str(timeline_id))
    key = re.sub(r"_run-\d+$", "", key)
    return re.sub(r"^hcp7t:(?:pre|post):", "hcp7t:", key)


@lru_cache(maxsize=4)
def _records(runs_jsonl, mtime_ns):
    return tuple(json.loads(line) for line in Path(runs_jsonl).read_text().splitlines() if line.strip())


def bundle_runs(bundle_path):
    """The bundle's run records (runs/runs.jsonl), read once per file version."""
    path = Path(bundle_path) / "runs" / "runs.jsonl"
    return _records(str(path), path.stat().st_mtime_ns)


def _is_pattern(text):
    return any(character in text for character in "*?[")


def split_runs(cfg, split, records=None):
    """Run records of a split, in bundle order: matched by data.<split>_runs, minus held-out subjects."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    patterns = cfg["data"][f"{split}_runs"]
    if not patterns:
        raise ValueError(f"No {split} split: data.{split}_runs is empty")
    records = bundle_runs(cfg["data"]["bundle_path"]) if records is None else records
    ids = {record["run_id"] for record in records}
    selected = set()
    for pattern in patterns:
        matches = {i for i in ids if fnmatchcase(i, pattern)} if _is_pattern(pattern) else {pattern} & ids
        if not matches:
            raise ValueError(f"Run selection matches nothing: {pattern}")
        selected |= matches
    heldout = set(cfg["data"]["heldout_subjects"])
    return [r for r in records if r["run_id"] in selected and r["subject_id"] not in heldout]


def frame_ranges(cfg, split, run_id):
    """[(start, stop), ...] frames of the run that the split may use, or None for the whole run."""
    spec = cfg["data"]["split_frames"].get(split, {})
    matches = [ranges for pattern, ranges in spec.items() if fnmatchcase(run_id, pattern)]
    if len(matches) > 1:
        raise ValueError(f"{run_id} matches several data.split_frames.{split} patterns")
    return [tuple(pair) for pair in matches[0]] if matches else None


def check_splits(cfg, records=None):
    """Raise if two defined splits share frames of a run; return the bundle's run records."""
    records = bundle_runs(cfg["data"]["bundle_path"]) if records is None else records
    used = {}
    for split in SPLITS:
        if not cfg["data"][f"{split}_runs"]:
            continue
        for record in split_runs(cfg, split, records):
            run_id = record["run_id"]
            ranges = frame_ranges(cfg, split, run_id) or [(0, record["n_timepoints"])]
            for other, other_ranges in used.get(run_id, []):
                if any(a < d and c < b for a, b in ranges for c, d in other_ranges):
                    raise ValueError(f"Splits {other} and {split} share frames of {run_id}")
            used.setdefault(run_id, []).append((split, ranges))
    for split, spec in cfg["data"]["split_frames"].items():
        for pattern in spec:
            if not any(fnmatchcase(r["run_id"], pattern) for r in records):
                raise ValueError(f"data.split_frames.{split} pattern matches no run: {pattern}")
    return records


def with_split(cfg, split):
    """(cfg, source): cfg with the split's definition, taken from the dataset's base config when cfg
    lacks it (source is then that file's path, otherwise None).

    Runs trained before a development or test split existed keep their own training and validation
    splits; the new split comes from state_of_mind/configs/<dataset>/base.yaml and is checked against them.
    """
    from copy import deepcopy
    from state_of_mind.utils.config import load_config
    if cfg["data"][f"{split}_runs"]:
        return cfg, None
    dataset = json.loads((Path(cfg["data"]["bundle_path"]) / "bundle.json").read_text())["dataset_id"]
    base_path = Path(__file__).resolve().parents[1] / "configs" / dataset / "base.yaml"
    base = load_config(base_path)
    if not base["data"][f"{split}_runs"]:
        raise ValueError(f"No {split} split for {dataset}: data.{split}_runs is empty in {base_path}")
    cfg = deepcopy(cfg)
    cfg["data"][f"{split}_runs"] = base["data"][f"{split}_runs"]
    cfg["data"]["heldout_subjects"] = sorted(set(cfg["data"]["heldout_subjects"]) | set(base["data"]["heldout_subjects"]))
    if split in base["data"]["split_frames"]:
        cfg["data"]["split_frames"] = {**cfg["data"]["split_frames"], split: base["data"]["split_frames"][split]}
    check_splits(cfg)
    return cfg, base_path
