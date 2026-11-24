"""Streaming fMRI-space metrics: per-run correlations, horizon curves, subject summaries.

Per run: global Pearson r over the flattened parcel-by-time trajectory, mean
per-parcel temporal Pearson (undefined parcels excluded), and MSE. Run values
are averaged over held-out runs pooled across subjects, then also by subject.
Horizon curves pool all windows and parcels at each rollout step.

Also, following BrainVista (arXiv 2602.04512): spatial pattern correlation
(Pearson across parcels at each predicted step, averaged over steps), and
Cls@Top10: for each predicted 20-TR segment, rank its true segment among the
same subject's true segments from other runs by Pearson correlation; the score
is the fraction ranked in the top 10. With ``content_of`` (run ID -> stimulus
content), segments of the same content (a repeated viewing) are not candidates
either. Segments with fewer than 10 candidates are not scored (a single-content
split has no Cls@Top10).
"""
import math
import torch


class CorrelationSums:
    def __init__(self, shape):
        self.n = torch.zeros(shape, dtype=torch.float64)
        self.sx, self.sy = torch.zeros_like(self.n), torch.zeros_like(self.n)
        self.sxx, self.syy, self.sxy = torch.zeros_like(self.n), torch.zeros_like(self.n), torch.zeros_like(self.n)

    def add(self, x, y, dims):
        x, y = x.detach().double().cpu(), y.detach().double().cpu()
        if x.shape != y.shape:
            raise ValueError("Predicted and observed arrays must share a shape")
        count = 1
        for dim in dims:
            count *= x.shape[dim]
        self.n += count
        self.sx += x.sum(dim=dims)
        self.sy += y.sum(dim=dims)
        self.sxx += x.square().sum(dim=dims)
        self.syy += y.square().sum(dim=dims)
        self.sxy += (x * y).sum(dim=dims)

    def pearson(self):
        n = self.n.clamp_min(1)
        mean_x, mean_y = self.sx / n, self.sy / n
        cov = self.sxy / n - mean_x * mean_y
        var_x, var_y = self.sxx / n - mean_x.square(), self.syy / n - mean_y.square()
        defined = (self.n > 1) & (var_x > 1e-12) & (var_y > 1e-12)
        r = cov / torch.sqrt((var_x * var_y).clamp_min(1e-24))
        return torch.where(defined, r, torch.full_like(r, float("nan")))

    def mse(self):
        return (self.sxx - 2 * self.sxy + self.syy) / self.n.clamp_min(1)


def _clean(value):
    value = float(value)
    return None if math.isnan(value) else value


def _mean(values):
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _aggregate(items):
    return {"global_r": _mean([i["global_r"] for i in items]),
            "parcel_pearson": _mean([i["parcel_pearson"] for i in items]),
            "mse": _mean([i["mse"] for i in items]), "runs": len(items),
            "runs_with_undefined_r": sum(i["global_r"] is None for i in items)}


def spatial_correlation(predicted, observed):
    """Pearson across parcels for every [batch, step]; NaN where either pattern is flat."""
    x = predicted.detach().double().cpu()
    y = observed.detach().double().cpu()
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    scale = (x.square().sum(dim=-1) * y.square().sum(dim=-1)).sqrt()
    r = (x * y).sum(dim=-1) / scale.clamp_min(1e-24)
    return torch.where(scale > 1e-12, r, torch.full_like(r, float("nan")))


def _normalized_segments(values, length):
    """[B, H, P] -> [B * floor(H / length), length * P] rows with zero mean and unit variance."""
    count = values.shape[1] // length
    rows = values[:, :count * length].detach().double().cpu().reshape(values.shape[0] * count, -1)
    rows = rows - rows.mean(dim=1, keepdim=True)
    rows = rows / rows.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    return rows.to(torch.float16), count


def top_k_identification(subjects, runs, predicted, observed, k=10):
    """Fraction of queries whose true segment ranks in the top k among same-subject segments of
    other runs (``runs`` may hold content keys instead); queries with fewer than k candidates are skipped."""
    hits, queries, candidates = 0, 0, 0
    for subject in sorted(set(subjects)):
        rows = [i for i, value in enumerate(subjects) if value == subject]
        p = predicted[rows].float()
        t = observed[rows].float()
        similarity = p @ t.T / p.shape[1]
        run = [runs[i] for i in rows]
        same_run = torch.tensor([[a == b for b in run] for a in run])
        own = similarity.diagonal()
        allowed = ~same_run | torch.eye(len(rows), dtype=torch.bool)
        better = (similarity > own[:, None]) & allowed
        scored = allowed.sum(dim=1) - 1 >= k
        hits += int(((better.sum(dim=1) < k) & scored).sum())
        queries += int(scored.sum())
        candidates += int(allowed[scored].sum())
    return {"cls_top10": hits / queries if queries else None, "cls_queries": queries,
            "cls_mean_candidates": candidates / queries if queries else None}


class FmriAccumulator:
    def __init__(self, horizon, parcels=1000, identify=False, segment_trs=20, content_of=None):
        self.runs = {}
        self.content_of = content_of or (lambda run_id: run_id)
        self.parcels = parcels
        self.horizon = CorrelationSums((horizon,))
        self.pattern = torch.zeros(horizon, dtype=torch.float64)
        self.pattern_count = torch.zeros(horizon, dtype=torch.float64)
        self.segment_trs = min(segment_trs, horizon)
        self.segments = {"subjects": [], "runs": [], "predicted": [], "observed": []} if identify else None

    def add(self, run_ids, subject_ids, predicted, observed):
        """predicted/observed: [B, H, parcels] for windows of the listed runs."""
        if predicted.shape != observed.shape or predicted.ndim != 3 or predicted.shape[-1] != self.parcels:
            raise ValueError("Expected [batch, horizon, parcels] predicted and observed fMRI")
        self.horizon.add(predicted, observed, dims=(0, 2))
        pattern = spatial_correlation(predicted, observed)
        defined = torch.isfinite(pattern)
        self.pattern += torch.where(defined, pattern, torch.zeros_like(pattern)).sum(dim=0)
        self.pattern_count += defined.sum(dim=0)
        if self.segments is not None:
            rows, count = _normalized_segments(predicted, self.segment_trs)
            self.segments["predicted"].append(rows)
            self.segments["observed"].append(_normalized_segments(observed, self.segment_trs)[0])
            for row, run_id in enumerate(run_ids):
                self.segments["subjects"] += [str(subject_ids[row])] * count
                self.segments["runs"] += [str(self.content_of(str(run_id)))] * count
        for row, run_id in enumerate(run_ids):
            entry = self.runs.get(run_id)
            if entry is None:
                entry = self.runs[run_id] = {"subject_id": str(subject_ids[row]),
                                            "global": CorrelationSums(()),
                                            "parcel": CorrelationSums((self.parcels,))}
            entry["global"].add(predicted[row], observed[row], dims=(0, 1))
            entry["parcel"].add(predicted[row], observed[row], dims=(0,))

    def summary(self):
        runs = []
        for run_id, entry in sorted(self.runs.items()):
            parcel = entry["parcel"].pearson()
            finite = torch.isfinite(parcel)
            runs.append({"run_id": run_id, "subject_id": entry["subject_id"],
                         "global_r": _clean(entry["global"].pearson()),
                         "parcel_pearson": _clean(parcel[finite].mean()) if finite.any() else None,
                         "defined_parcels": int(finite.sum()), "mse": _clean(entry["global"].mse())})
        result = _aggregate(runs)
        result["pattern_corr"] = _clean(self.pattern.sum() / self.pattern_count.sum()) \
            if self.pattern_count.sum() else None
        result["pattern_corr_by_horizon"] = [_clean(v / n) if n else None
                                             for v, n in zip(self.pattern.tolist(), self.pattern_count.tolist())]
        if self.segments is not None and self.segments["subjects"]:
            result.update(top_k_identification(self.segments["subjects"], self.segments["runs"],
                                               torch.cat(self.segments["predicted"]),
                                               torch.cat(self.segments["observed"])))
            result["cls_segment_trs"] = self.segment_trs
        result["mse_by_horizon"] = [float(v) for v in self.horizon.mse().tolist()]
        result["pearson_by_horizon"] = [_clean(v) for v in self.horizon.pearson().tolist()]
        subjects = sorted({run["subject_id"] for run in runs})
        result["by_subject"] = {s: _aggregate([run for run in runs if run["subject_id"] == s]) for s in subjects}
        result["by_run"] = runs
        return result
