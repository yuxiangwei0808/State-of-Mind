"""Inter-subject correlation (ISC) reference rows for the fMRI metrics.

Several subjects watch the same stimulus, so their fMRI shares a stimulus-driven response.
On exactly the TRs the model predicted, per parcel and run:

``other_subjects``  Pearson between the run and the mean of the other subjects' fMRI for the same
                    stimulus at the same TRs, averaged like the model's parcel Pearson.
``isc_ceiling``     sqrt(mean pairwise ISC) per parcel, averaged over parcels: the expected
                    correlation between one subject and the response shared by all subjects,
                    assuming equal noise in every subject.

When subjects saw the stimulus more than once (Movie10's *Hidden Figures* and *Life*):

``repeat``          Pearson between the run and the same subject's other viewing(s) of the same
                    stimulus at the same TRs (test-retest), averaged like the model's parcel Pearson,
                    and ``repeat_ceiling``, sqrt of the mean test-retest r per parcel, averaged over
                    parcels. Familiarity effects lower it, so it is also only a reference.

Both are approximate, model-free references, not the actual ceiling of the model's score. ISC
only counts the response shared across subjects, while the model correlates each subject's fMRI
with a signal reconstructed from the stimulus, that subject's own brain history and a
per-subject readout, so the model can exceed them (notably over the first steps, where brain
history helps). Runs are grouped by stimulus content (data/splits.py::content_key); with many
subjects (HCP), each run is compared with at most MAX_OTHERS other subjects.
"""
import re
from collections import defaultdict
import numpy as np
from state_of_mind.data.splits import content_key

MAX_OTHERS = 24

def stimulus_key(timeline_id):
    """Stimulus timeline shared by all subjects (its feature rows); Algonauts timelines carry a
    per-subject ``sub-XX:`` prefix. Repeated viewings keep separate timelines; see content_key."""
    return re.sub(r"^sub-[^:]+:", "", str(timeline_id))


def _pearson(a, b):
    """Per-column Pearson of two [T, P] arrays; NaN where either column is constant."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    scale = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, (a * b).sum(axis=0) / scale, np.nan)


def _valid_trs(dataset, records, trs):
    for record in records:
        valid = dataset._open(record["valid"])
        trs = trs[trs < len(valid)]
        trs = trs[np.asarray(valid[trs], dtype=bool)]
    return trs


def _ceiling(pair_sum, pair_count):
    defined = np.asarray(pair_count) > 0
    r = np.asarray(pair_sum)[defined] / np.asarray(pair_count)[defined]
    return float(np.sqrt(np.clip(r, 0, None)).mean()), float(r.mean())


def intersubject_reference(dataset, frames_by_run, limits, *, max_others=MAX_OTHERS, seed=0):
    """ISC (and repeated-viewing) references over the first ``limit`` rollout steps, for each limit.

    ``frames_by_run`` maps a run id to the [windows, horizon] target frame indices that were
    evaluated. Returns {limit: summary or None}; None when no run shares its stimulus with another
    subject in the dataset.
    """
    records = {record["run_id"]: record for record in dataset.records}
    groups = defaultdict(list)
    for record in dataset.records:
        groups[content_key(record["timeline_id"])].append(record)
    rng = np.random.default_rng(seed)
    others, repeats = {}, {}
    for run_id in sorted(frames_by_run):
        record = records[run_id]
        group = groups[content_key(record["timeline_id"])]
        pool = [other for other in group if other["subject_id"] != record["subject_id"]]
        if len(pool) > max_others:
            pool = [pool[i] for i in sorted(rng.choice(len(pool), max_others, replace=False))]
        others[run_id] = pool
        repeats[run_id] = [other for other in group
                           if other["subject_id"] == record["subject_id"] and other["run_id"] != run_id]
    result = {}
    for limit in limits:
        run_values, pair_sum, pair_count = [], 0.0, 0
        repeat_values, repeat_sum, repeat_count = [], 0.0, 0
        for run_id, frames in sorted(frames_by_run.items()):
            record = records[run_id]
            requested = np.unique(np.asarray(frames)[:, :limit])
            if repeats[run_id]:
                trs = _valid_trs(dataset, [record] + repeats[run_id], requested)
                if len(trs) >= 3:
                    observed = np.asarray(dataset._open(record["brain"])[trs], dtype=np.float64)
                    values = []
                    for other in repeats[run_id]:
                        r = _pearson(observed, np.asarray(dataset._open(other["brain"])[trs], dtype=np.float64))
                        defined = np.isfinite(r)
                        repeat_sum = repeat_sum + np.where(defined, r, 0.0)
                        repeat_count = repeat_count + defined
                        values.append(r)
                    if np.isfinite(values).any():
                        repeat_values.append(float(np.nanmean(values)))
            if not others[run_id]:
                continue
            trs = _valid_trs(dataset, [record] + others[run_id], requested)
            if len(trs) < 3:
                continue
            observed = np.asarray(dataset._open(record["brain"])[trs], dtype=np.float64)
            rest = [np.asarray(dataset._open(other["brain"])[trs], dtype=np.float64) for other in others[run_id]]
            leave_one_out = _pearson(observed, np.mean(rest, axis=0))
            if np.isfinite(leave_one_out).any():
                run_values.append(float(np.nanmean(leave_one_out)))
            for other in rest:
                pairwise = _pearson(observed, other)
                defined = np.isfinite(pairwise)
                pair_sum = pair_sum + np.where(defined, pairwise, 0.0)
                pair_count = pair_count + defined
        if not run_values:
            result[limit] = None
            continue
        ceiling, isc = _ceiling(pair_sum, pair_count)
        result[limit] = {"other_subjects_parcel_pearson": float(np.mean(run_values)),
                         "isc_ceiling_parcel_pearson": ceiling, "pairwise_isc": isc, "runs": len(run_values),
                         "max_other_subjects": max(len(pool) for pool in others.values())}
        if repeat_values:
            repeat_ceiling = _ceiling(repeat_sum, repeat_count)[0]
            result[limit].update(repeat_parcel_pearson=float(np.mean(repeat_values)),
                                 repeat_ceiling_parcel_pearson=repeat_ceiling, repeat_runs=len(repeat_values))
    return result
