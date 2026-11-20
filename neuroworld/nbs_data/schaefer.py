"""Two-command entry point for Schaefer-1000 fMRI preparation.

``prepare`` is a read-only dry run. ``run`` writes directly to the requested
output and resumes committed runs after interruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from neuroworld.nbs_data.pipeline import (
    PipelineError,
    prepare as prepare_dataset,
    run_preparation,
)


DATASETS = ("algonauts", "hcp_movies")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--atlas-manifest",
        type=Path,
        default=None,
        help="Required for HCP; omitted for Algonauts identity data.",
    )
    parser.add_argument("--raw-index", type=Path, required=True)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dry_run = commands.add_parser(
        "prepare",
        help="Dry-run the complete source, atlas, and raw-feature inventory.",
    )
    _add_inputs(dry_run)

    run = commands.add_parser(
        "run",
        help="Write or resume prepared data directly at --output.",
    )
    _add_inputs(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Process the first N canonical runs for a smoke test.",
    )
    run.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Spawn independent per-run fMRI workers; the parent remains the sole writer.",
    )
    run.add_argument(
        "--eager-volumes",
        action="store_true",
        help="Materialize each HCP float32 volume once to avoid repeated gzip reads.",
    )

    return parser


def _pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_id": args.dataset,
        "source_root": args.source_root,
        "atlas_manifest": args.atlas_manifest,
        "raw_index": args.raw_index,
    }


def run(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        report = prepare_dataset(**_pipeline_kwargs(args))
        _print_json(report)
        return 0 if report.get("ready") else 2
    if args.command == "run":
        kwargs = _pipeline_kwargs(args)
        kwargs.update(
            output=args.output,
            limit=args.limit,
            workers=args.workers,
            eager_volumes=args.eager_volumes,
        )
        _print_json(run_preparation(**kwargs))
        return 0
    raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        status = run(args)
    except (PipelineError, FileNotFoundError, ValueError) as exc:
        print(f"neuroworld-schaefer: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
