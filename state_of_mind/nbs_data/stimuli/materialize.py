"""CLI for exporting raw 2-Hz TRIBEv2 features used by Schaefer builds.

The command discovers the exact dataset timelines, reads the existing
pre-projector cache, and atomically writes raw arrays plus ``chunks.jsonl``.
It does not pool to native TR, normalize features, run model inference, or
compute file hashes. Interrupted exports resume from ``.<output>.staging``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from state_of_mind.nbs_data.stimuli.raw_cache import (
    RAW_CACHE_INDEX_NAME,
    ChunkIndexRecord,
    RawCacheError,
    RawCacheTransactionError,
    _prepare_raw_export,
    _publish_raw_export,
    _stage_raw_chunks,
    chunks_from_explicit_extractor,
    export_raw_chunks,
    iter_tribev2_raw_chunks,
)


_CLI_DATASETS = ("algonauts", "hcp_movies")
_TRIBE_DATASET = {
    "algonauts": "algonauts",
    "hcp_movies": "hcp",
}
_TRIBE_EXTRACTOR = (
    "state_of_mind.nbs_data.stimuli.raw_cache:iter_tribev2_raw_chunks"
)
_RAW_ONLY_KWARGS = frozenset({"events_path", "cache_dir", "config_path"})
_SOURCE_INVENTORY = {
    "algonauts": (1_392, 1_392),
    "hcp_movies": (725, 8),
}


def _canonical_cli_dataset(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RawCacheError("dataset is required")
    dataset = value.strip().lower().replace("-", "_").replace(" ", "_")
    if dataset not in _TRIBE_DATASET:
        raise RawCacheError(
            f"Unsupported dataset {value!r}; expected algonauts or hcp_movies"
        )
    return dataset


def discover_source_timeline_ids(
    source_root: str | Path,
    *,
    dataset: str,
) -> tuple[str, ...]:
    """Return the exact audited timeline inventory for one source dataset."""

    canonical = _canonical_cli_dataset(dataset)
    root = Path(source_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise RawCacheError(f"source_root is not a directory: {root}")
    try:
        from state_of_mind.nbs_data.fmri._sources import audit_run_sources, discover_runs

        runs = tuple(discover_runs(canonical, root, strict=True))
        audit = audit_run_sources(runs, require_full_inventory=True)
    except Exception as exc:
        raise RawCacheError(
            f"Cannot discover {canonical} source timelines under {root}: {exc}"
        ) from exc

    timeline_ids: set[str] = set()
    for run in runs:
        timeline_id = getattr(run, "timeline_id", None)
        if not isinstance(timeline_id, str) or not timeline_id.strip():
            raise RawCacheError(
                f"{canonical} source adapter returned a run without a timeline_id"
            )
        timeline_ids.add(timeline_id.strip())

    expected_runs, expected_timelines = _SOURCE_INVENTORY[canonical]
    audit_timelines = audit.get("timelines")
    try:
        audited = {
            value.strip()
            for value in audit_timelines
            if isinstance(value, str) and value.strip()
        }
    except TypeError:
        audited = set()
    if (
        audit.get("dataset_id") != canonical
        or audit.get("ready") is not True
        or audit.get("full_inventory") is not True
        or audit.get("run_count") != expected_runs
        or len(runs) != expected_runs
        or len(timeline_ids) != expected_timelines
        or audited != timeline_ids
    ):
        raise RawCacheError(
            f"{canonical} source audit did not prove the exact full inventory: "
            f"runs={len(runs)}/{expected_runs}, "
            f"timelines={len(timeline_ids)}/{expected_timelines}, "
            f"blockers={audit.get('blockers')!r}"
        )
    return tuple(sorted(timeline_ids))


def _validate_extractor_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(kwargs)
    if fields != _RAW_ONLY_KWARGS:
        raise RawCacheError(
            "Extractor kwargs must contain exactly events_path, cache_dir, and "
            f"config_path (missing={sorted(_RAW_ONLY_KWARGS - fields)}, "
            f"unexpected={sorted(fields - _RAW_ONLY_KWARGS)})"
        )
    events_path = kwargs["events_path"]
    if not isinstance(events_path, list) or len(events_path) != 4:
        raise RawCacheError("events_path must contain exactly four TSV paths")
    resolved_events: list[str] = []
    for value in events_path:
        if not isinstance(value, str) or not value.strip():
            raise RawCacheError("events_path entries must be non-empty strings")
        path = Path(value).expanduser().resolve(strict=False)
        if path.suffix.lower() != ".tsv":
            raise RawCacheError(f"events_path entries must be TSV paths: {value!r}")
        resolved_events.append(str(path))
    if len(set(resolved_events)) != 4:
        raise RawCacheError("events_path must contain four distinct TSV paths")
    for name in ("cache_dir", "config_path"):
        if not isinstance(kwargs[name], str) or not kwargs[name].strip():
            raise RawCacheError(f"{name} must be a non-empty JSON string")
    return dict(kwargs)


def _scope_chunks(
    chunks: Any,
    selected_timeline_ids: Sequence[str],
) -> Iterator[Any]:
    """Reject chunks outside the source inventory or an incomplete export."""

    selected = set(selected_timeline_ids)
    seen: set[str] = set()
    for chunk in chunks:
        timeline_id = getattr(chunk, "timeline_id", None)
        if not isinstance(timeline_id, str) or not timeline_id.strip():
            raise RawCacheError(
                "Raw extractor yielded a chunk without a non-empty string timeline_id"
            )
        timeline_id = timeline_id.strip()
        if timeline_id not in selected:
            raise RawCacheError(
                f"Raw extractor yielded a timeline outside source-root discovery: {timeline_id!r}"
            )
        seen.add(timeline_id)
        yield chunk
    missing = sorted(selected - seen)
    if missing:
        raise RawCacheError(
            f"Raw extractor did not yield every discovered source timeline; missing={missing!r}"
        )


def _partition_timeline_ids(
    timeline_ids: Sequence[str],
    workers: int,
) -> tuple[tuple[str, ...], ...]:
    """Round-robin timeline IDs into deterministic, non-overlapping workers."""

    if isinstance(workers, bool) or int(workers) != workers or int(workers) <= 0:
        raise RawCacheError("workers must be a positive integer")
    worker_count = min(int(workers), len(timeline_ids))
    if worker_count <= 0:
        raise RawCacheError("parallel export requires at least one timeline")
    return tuple(
        tuple(timeline_ids[offset::worker_count])
        for offset in range(worker_count)
    )


def _stage_timeline_partition(
    task: tuple[
        int,
        str,
        dict[str, Any],
        tuple[str, ...],
        str,
        str,
    ],
) -> tuple[int, list[ChunkIndexRecord]]:
    """Process-worker entry point for one disjoint timeline partition."""

    partition_index, extractor, kwargs, timeline_ids, root_text, staging_text = task
    worker_kwargs = dict(kwargs)
    worker_kwargs["selected_timeline_ids"] = list(timeline_ids)
    print(
        f"[parallel] partition {partition_index + 1} starting "
        f"with {len(timeline_ids)} timelines",
        flush=True,
    )
    chunks = _scope_chunks(
        chunks_from_explicit_extractor(
            extractor,
            source_kwargs=worker_kwargs,
            strict_timeline_ids=True,
        ),
        timeline_ids,
    )
    records = _stage_raw_chunks(
        chunks,
        root=Path(root_text),
        staging=Path(staging_text),
    )
    return partition_index, records


def _export_raw_chunks_parallel(
    *,
    extractor_kwargs: Mapping[str, Any],
    timeline_ids: Sequence[str],
    generated_index_dir: str | Path,
    workers: int,
    overwrite: bool,
) -> Path:
    """Extract disjoint timelines in spawned processes, then publish once."""

    partitions = _partition_timeline_ids(timeline_ids, workers)
    root, staging = _prepare_raw_export(
        generated_index_dir,
        index_name=RAW_CACHE_INDEX_NAME,
        overwrite=overwrite,
    )
    tasks = [
        (
            partition_index,
            _TRIBE_EXTRACTOR,
            dict(extractor_kwargs),
            partition,
            str(root),
            str(staging),
        )
        for partition_index, partition in enumerate(partitions)
    ]
    records: list[ChunkIndexRecord] = []
    context = mp.get_context("spawn")
    pool = context.Pool(processes=len(partitions))
    try:
        for partition_index, partition_records in pool.imap_unordered(
            _stage_timeline_partition,
            tasks,
            chunksize=1,
        ):
            records.extend(partition_records)
            print(
                f"[parallel] partition {partition_index + 1}/{len(partitions)} "
                f"finished with {len(partition_records)} chunks",
                flush=True,
            )
    except BaseException as exc:
        pool.terminate()
        pool.join()
        if isinstance(exc, (RawCacheError, RawCacheTransactionError)):
            raise
        raise RawCacheTransactionError(
            f"Parallel raw-cache staging failed: {exc}"
        ) from exc
    else:
        pool.close()
        pool.join()
    try:
        return _publish_raw_export(
            records,
            root=root,
            staging=staging,
            index_name=RAW_CACHE_INDEX_NAME,
        )
    except BaseException as exc:
        if isinstance(exc, (RawCacheError, RawCacheTransactionError)):
            raise
        raise RawCacheTransactionError(
            f"Atomic parallel raw-cache publication failed: {exc}"
        ) from exc


def _read_json_value(path_or_json: str | Path, *, label: str) -> Any:
    candidate = Path(path_or_json).expanduser()
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RawCacheError(f"Cannot parse {label} JSON file {candidate}: {exc}") from exc
    try:
        return json.loads(str(path_or_json))
    except json.JSONDecodeError as exc:
        raise RawCacheError(
            f"{label} must be a JSON file or JSON object string; got {path_or_json!r}"
        ) from exc


def materialize_native_features(
    *,
    extractor: str | Any | None = None,
    extractor_kwargs: Mapping[str, Any] | None = None,
    generated_index_dir: str | Path | None = None,
    dataset: str | None = None,
    source_root: str | Path | None = None,
    export_raw_only: bool = False,
    overwrite: bool = False,
    workers: int = 1,
) -> Path:
    """Export only source-scoped raw chunks for the active Schaefer pipeline."""

    if not export_raw_only:
        raise RawCacheError("This command supports only --export-raw-only")
    if extractor is None:
        raise RawCacheError("--export-raw-only requires --extractor")
    if source_root is None or dataset is None:
        raise RawCacheError("--export-raw-only requires --source-root and --dataset")
    if generated_index_dir is None:
        raise RawCacheError("--export-raw-only requires --generated-index-dir")
    if not (
        (isinstance(extractor, str) and extractor == _TRIBE_EXTRACTOR)
        or extractor is iter_tribev2_raw_chunks
    ):
        raise RawCacheError(
            "--export-raw-only requires the canonical TRIBEv2 raw extractor"
        )
    if extractor_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(extractor_kwargs, Mapping):
        kwargs = dict(extractor_kwargs)
    else:
        raise RawCacheError("extractor_kwargs must be a mapping")

    canonical = _canonical_cli_dataset(dataset)
    kwargs = _validate_extractor_kwargs(kwargs)
    timeline_ids = discover_source_timeline_ids(source_root, dataset=canonical)
    kwargs["dataset"] = _TRIBE_DATASET[canonical]
    kwargs["selected_timeline_ids"] = list(timeline_ids)
    if isinstance(workers, bool) or int(workers) != workers or int(workers) <= 0:
        raise RawCacheError("workers must be a positive integer")
    if int(workers) > 1:
        return _export_raw_chunks_parallel(
            extractor_kwargs=kwargs,
            timeline_ids=timeline_ids,
            generated_index_dir=generated_index_dir,
            workers=int(workers),
            overwrite=overwrite,
        )
    chunks = _scope_chunks(
        chunks_from_explicit_extractor(
            extractor,
            source_kwargs=kwargs,
            strict_timeline_ids=True,
        ),
        timeline_ids,
    )
    return export_raw_chunks(chunks, generated_index_dir, overwrite=overwrite)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-raw-only", action="store_true", required=True)
    parser.add_argument("--extractor", required=True, help="Canonical module:function extractor.")
    parser.add_argument("--extractor-kwargs", required=True, help="JSON object or JSON file.")
    parser.add_argument("--dataset", choices=_CLI_DATASETS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--generated-index-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn this many disjoint timeline extraction processes (default: 1).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    extractor_kwargs = _read_json_value(args.extractor_kwargs, label="extractor kwargs")
    if not isinstance(extractor_kwargs, Mapping):
        raise SystemExit("--extractor-kwargs must decode to a JSON object")
    try:
        index = materialize_native_features(
            extractor=args.extractor,
            extractor_kwargs=extractor_kwargs,
            generated_index_dir=args.generated_index_dir,
            dataset=args.dataset,
            source_root=args.source_root,
            export_raw_only=args.export_raw_only,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    except RawCacheError as exc:
        raise SystemExit(f"raw stimulus export failed: {exc}") from exc
    print(index)
    return 0


if __name__ == "__main__":
    main()
