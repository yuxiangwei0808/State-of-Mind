"""Shared stimulus contracts and transformed-event timing helpers.

Native stimulus arithmetic lives in :mod:`.timing`. This module exposes that
shared API together with the event-table helpers used by every dataset's raw
TRIBEv2 adapter. Dataset-specific policy lives in the three sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from state_of_mind.data.algonauts_contract import canonical_algonauts_stimulus_id
from .timing import *  # noqa: F401,F403 - shared stimulus surface
from .timing import __all__ as _STIMULUS_EXPORTS


# These names describe the transformed-event/NeuralSet clock and are retained
# here so callers do not need to import the old command module.
EXPECTED_FREQUENCY_HZ = RAW_FREQUENCY_HZ
NEURALSET_SENTENCE_BOUNDARY_EPSILON_SECONDS = 1e-6
_BOUNDARY_COMPARISON_TOLERANCE_SECONDS = 1e-12


@dataclass(frozen=True)
class CanonicalEventGroup:
    """One deterministic representative timeline for one stimulus."""

    stimulus_id: str
    events: Any
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class TimelineEventGroup:
    """One concrete timeline whose temporal context remains distinct."""

    timeline_id: str
    stimulus_id: str | None
    events: Any
    start_seconds: float
    duration_seconds: float


def _text_cell(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _canonical_id_from_row(row: Mapping[str, object]) -> str:
    task = _text_cell(row.get("task"))
    if task is None:
        raise ValueError("Every transformed event must retain its Algonauts task field")
    movie = _text_cell(row.get("movie"))
    chunk = _text_cell(row.get("chunk"))
    if movie is not None and chunk is not None:
        return canonical_algonauts_stimulus_id(task, movie=movie, chunk=chunk)
    for field in ("stimulus", "timeline"):
        value = _text_cell(row.get(field))
        if value is not None:
            if field == "timeline" and ":" in value:
                value = value.rsplit(":", 1)[-1]
            try:
                return canonical_algonauts_stimulus_id(task, value)
            except ValueError:
                continue
    raise ValueError(
        "Cannot derive a canonical stimulus ID; transformed rows need task plus "
        "movie/chunk or a canonical stimulus field"
    )


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_transformed_event_start(
    value: object,
    event_type: object,
    *,
    label: str,
) -> float:
    """Normalize the known one-microsecond Sentence enclosure."""

    start = _finite_float(value)
    if start is None:
        raise ValueError(f"{label}: transformed event start is not finite")
    if start >= 0:
        return start
    kind = _text_cell(event_type)
    lower_bound = -(
        NEURALSET_SENTENCE_BOUNDARY_EPSILON_SECONDS
        + _BOUNDARY_COMPARISON_TOLERANCE_SECONDS
    )
    if kind == "Sentence" and start >= lower_bound:
        return 0.0
    raise ValueError(
        f"{label}: transformed {kind or 'unknown'} event has negative start {start}"
    )


def neuralset_segment_boundary_shift_seconds(
    events: Any,
    *,
    physical_support_start_seconds: object,
    label: str,
) -> float:
    """Return the audited raw-to-physical NeuralSet origin shift."""

    physical_start = _finite_float(physical_support_start_seconds)
    if physical_start is None or physical_start < 0:
        raise ValueError(f"{label}: physical support start is not finite/non-negative")
    raw_starts: list[float] = []
    physical_starts: list[float] = []
    for _, row in events.iterrows():
        raw_start = _finite_float(row.get("start"))
        physical_event_start = normalize_transformed_event_start(
            row.get("start"), row.get("type"), label=label
        )
        assert raw_start is not None
        raw_starts.append(raw_start)
        physical_starts.append(physical_event_start)
    if not raw_starts:
        raise ValueError(f"{label}: transformed event table is empty")
    expected_physical_start = min(physical_starts)
    if not math.isclose(
        physical_start,
        expected_physical_start,
        rel_tol=0.0,
        abs_tol=_BOUNDARY_COMPARISON_TOLERANCE_SECONDS,
    ):
        raise ValueError(
            f"{label}: physical support start {physical_start} disagrees with "
            f"normalized events {expected_physical_start}"
        )
    raw_start = min(raw_starts)
    shift = physical_start - raw_start
    if abs(shift) <= _BOUNDARY_COMPARISON_TOLERANCE_SECONDS:
        return 0.0
    if not (
        raw_start < 0
        and math.isclose(
            physical_start,
            0.0,
            rel_tol=0.0,
            abs_tol=_BOUNDARY_COMPARISON_TOLERANCE_SECONDS,
        )
        and 0 < shift <= (
            NEURALSET_SENTENCE_BOUNDARY_EPSILON_SECONDS
            + _BOUNDARY_COMPARISON_TOLERANCE_SECONDS
        )
    ):
        raise ValueError(f"{label}: unsupported NeuralSet boundary shift {shift}")
    return shift


def normalize_neuralset_segment_start(
    value: object,
    *,
    boundary_shift_seconds: object,
    label: str,
) -> float:
    """Map a returned NeuralSet segment start onto the physical timeline."""

    raw_start = _finite_float(value)
    shift = _finite_float(boundary_shift_seconds)
    if raw_start is None or shift is None or shift < 0:
        raise ValueError(f"{label}: segment start/boundary shift is invalid")
    if shift > (
        NEURALSET_SENTENCE_BOUNDARY_EPSILON_SECONDS
        + _BOUNDARY_COMPARISON_TOLERANCE_SECONDS
    ):
        raise ValueError(f"{label}: segment boundary shift exceeds the audited 1us")
    normalized = raw_start + shift
    if normalized < -_BOUNDARY_COMPARISON_TOLERANCE_SECONDS:
        raise ValueError(f"{label}: normalized segment start is negative")
    return 0.0 if normalized < 0 else normalized


def _event_bounds(events: Any, stimulus_id: str) -> tuple[float, float]:
    starts: list[float] = []
    stops: list[float] = []
    for _, row in events.iterrows():
        raw_start = _finite_float(row.get("start"))
        start = normalize_transformed_event_start(
            row.get("start"), row.get("type"), label=stimulus_id
        )
        assert raw_start is not None
        stop = _finite_float(row.get("stop"))
        if stop is None:
            duration = _finite_float(row.get("duration"))
            if duration is not None:
                stop = raw_start + duration
        starts.append(start)
        if stop is not None and stop >= raw_start:
            stops.append(stop)
    if not starts or not stops:
        raise ValueError(f"{stimulus_id}: cannot determine transformed event duration")
    start_seconds = min(starts)
    stop_seconds = max(stops)
    if stop_seconds <= start_seconds:
        raise ValueError(
            f"{stimulus_id}: invalid event bounds [{start_seconds}, {stop_seconds}]"
        )
    return start_seconds, stop_seconds - start_seconds


def iter_canonical_event_groups(
    events: Any,
    *,
    selected_stimulus_ids: set[str] | None = None,
) -> Iterator[CanonicalEventGroup]:
    """Yield one deterministic subject/run representative per stimulus."""

    required = {"task", "timeline", "subject", "type", "start"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Transformed events TSV is missing columns: {sorted(missing)}")
    if len(events) == 0:
        raise ValueError("Transformed events TSV is empty")

    frame = events.copy()
    frame["_neuroworld_stimulus_id"] = [
        _canonical_id_from_row(row) for row in frame.to_dict(orient="records")
    ]
    available = set(frame["_neuroworld_stimulus_id"])
    if selected_stimulus_ids is not None:
        unknown = selected_stimulus_ids - available
        if unknown:
            raise ValueError(f"Requested stimuli are absent from events TSV: {sorted(unknown)}")

    for stimulus_id in sorted(available):
        if selected_stimulus_ids is not None and stimulus_id not in selected_stimulus_ids:
            continue
        stimulus_rows = frame.loc[frame["_neuroworld_stimulus_id"] == stimulus_id]
        timelines = sorted(
            value
            for value in {_text_cell(item) for item in stimulus_rows["timeline"]}
            if value is not None
        )
        if not timelines:
            raise ValueError(f"{stimulus_id}: no valid timeline label")
        selected_timeline = timelines[0]
        representative = stimulus_rows.loc[
            stimulus_rows["timeline"] == selected_timeline
        ].copy()
        subjects = {
            value
            for value in (_text_cell(item) for item in representative["subject"])
            if value is not None
        }
        if len(subjects) != 1:
            raise ValueError(
                f"{stimulus_id}: representative timeline has subjects {sorted(subjects)}"
            )
        event_types = set(str(item) for item in representative["type"])
        for required_type in ("Video", "Audio"):
            if required_type not in event_types:
                raise ValueError(
                    f"{stimulus_id}: transformed timeline lacks {required_type} events; "
                    "run the full TRIBE audio/video/text transform first"
                )
        representative = representative.drop(columns=["_neuroworld_stimulus_id"])
        representative.loc[:, "split"] = "all"
        start_seconds, duration_seconds = _event_bounds(representative, stimulus_id)
        yield CanonicalEventGroup(
            stimulus_id=stimulus_id,
            events=representative,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )


def _optional_group_stimulus_id(events: Any) -> str | None:
    """Return a shared content ID when a timeline has exactly one stimulus."""

    for field in ("stimulus_id", "stimulus"):
        if field in events.columns:
            values = {
                value for value in (_text_cell(item) for item in events[field]) if value is not None
            }
            if len(values) == 1:
                return next(iter(values))
    canonical: set[str] = set()
    for row in events.to_dict(orient="records"):
        try:
            canonical.add(_canonical_id_from_row(row))
        except ValueError:
            continue
    return next(iter(canonical)) if len(canonical) == 1 else None


def iter_timeline_event_groups(
    events: Any,
    *,
    selected_stimulus_ids: set[str] | None = None,
    selected_timeline_ids: set[str] | None = None,
) -> Iterator[TimelineEventGroup]:
    """Yield every concrete TRIBEv2 timeline without context deduplication."""

    if selected_stimulus_ids is not None and selected_timeline_ids is not None:
        raise ValueError("Select transformed events by timeline or stimulus, not both")
    required = {"timeline", "subject", "type", "start"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Transformed events TSV is missing columns: {sorted(missing)}")
    if len(events) == 0:
        raise ValueError("Transformed events TSV is empty")

    observed_stimulus_ids: set[str] = set()
    timeline_values = {
        value for value in (_text_cell(item) for item in events["timeline"]) if value is not None
    }
    if selected_timeline_ids is not None:
        missing_timelines = selected_timeline_ids - timeline_values
        if missing_timelines:
            raise ValueError(f"Requested timelines are absent: {sorted(missing_timelines)}")
        timeline_values &= selected_timeline_ids

    if selected_stimulus_ids is not None:
        for timeline_id in sorted(timeline_values):
            representative = events.loc[events["timeline"].astype(str) == timeline_id]
            stimulus_id = _optional_group_stimulus_id(representative)
            if stimulus_id is not None:
                observed_stimulus_ids.add(stimulus_id)
        missing_selected = selected_stimulus_ids - observed_stimulus_ids
        if missing_selected:
            raise ValueError(
                "Requested stimuli are absent or not unique within a timeline: "
                f"{sorted(missing_selected)}"
            )

    for timeline_id in sorted(timeline_values):
        representative = events.loc[events["timeline"].astype(str) == timeline_id].copy()
        subjects = {
            value
            for value in (_text_cell(item) for item in representative["subject"])
            if value is not None
        }
        if len(subjects) != 1:
            raise ValueError(f"{timeline_id}: timeline has subjects {sorted(subjects)}")
        event_types = set(str(item) for item in representative["type"])
        for required_type in ("Video", "Audio"):
            if required_type not in event_types:
                raise ValueError(
                    f"{timeline_id}: transformed timeline lacks {required_type} events; "
                    "run the full TRIBE transform first"
                )
        stimulus_id = _optional_group_stimulus_id(representative)
        if selected_stimulus_ids is not None and stimulus_id not in selected_stimulus_ids:
            continue
        representative.loc[:, "split"] = "all"
        start_seconds, duration_seconds = _event_bounds(representative, timeline_id)
        yield TimelineEventGroup(
            timeline_id=timeline_id,
            stimulus_id=stimulus_id,
            events=representative,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )


def _resolve_events_paths(
    events_path: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    values: Sequence[str | Path]
    if isinstance(events_path, (str, Path)):
        values = (events_path,)
    else:
        values = events_path
    if not values:
        raise ValueError("At least one transformed events TSV is required")
    paths = tuple(
        sorted(
            (Path(value).expanduser().resolve(strict=True) for value in values),
            key=str,
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate transformed events TSV paths are not allowed")
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".tsv":
            raise ValueError(f"Transformed events input must be a .tsv file: {path}")
    return paths


def _read_transformed_events(paths: Sequence[Path], pandas_module: Any) -> Any:
    """Read transformed TSV shards while preserving literal text tokens."""

    required = {"timeline", "subject", "type", "start"}
    frames = []
    reference_columns: tuple[str, ...] | None = None
    for path in paths:
        frame = pandas_module.read_csv(
            path,
            sep="\t",
            low_memory=False,
            keep_default_na=False,
            na_values=[""],
        )
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing transformed event columns: {sorted(missing)}")
        if len(frame) == 0:
            raise ValueError(f"Transformed events shard is empty: {path}")
        columns = tuple(str(column) for column in frame.columns)
        if reference_columns is None:
            reference_columns = columns
        elif set(columns) != set(reference_columns):
            missing_from_shard = set(reference_columns) - set(columns)
            extra_in_shard = set(columns) - set(reference_columns)
            raise ValueError(
                f"Transformed shard schema differs for {path}: "
                f"missing={sorted(missing_from_shard)}, extra={sorted(extra_in_shard)}"
            )
        elif columns != reference_columns:
            frame = frame.loc[:, list(reference_columns)]
        frames.append(frame)
    return pandas_module.concat(frames, ignore_index=True, sort=False)


__all__ = [
    *_STIMULUS_EXPORTS,
    "EXPECTED_FREQUENCY_HZ",
    "NEURALSET_SENTENCE_BOUNDARY_EPSILON_SECONDS",
    "CanonicalEventGroup",
    "TimelineEventGroup",
    "iter_canonical_event_groups",
    "iter_timeline_event_groups",
    "neuralset_segment_boundary_shift_seconds",
    "normalize_neuralset_segment_start",
    "normalize_transformed_event_start",
]
