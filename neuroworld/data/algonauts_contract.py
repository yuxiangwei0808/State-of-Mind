"""Canonical identities and source contracts for the Algonauts S0 adapter.

The challenge HDF5 keys, TRIBEv2 event tables, and NeuralWorld bundle use
slightly different labels for the same acquisition.  This module is the one
place where those labels are translated.  It intentionally contains no HDF5,
Torch, or TRIBEv2 imports so runtime loaders and preparation tools can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ALGONAUTS_TASKS = ("friends", "movie10")
TRIBEV2_FRIENDS_SKIP_CONTENTS = frozenset(
    {
        "s04e01a",
        "s04e01b",
        "s04e13b",
        "s05e20a",
        "s06e03a",
    }
)


class AlgonautsContractError(ValueError):
    """Raised when an Algonauts label cannot be mapped without ambiguity."""


@dataclass(frozen=True)
class AlgonautsIdentity:
    """Canonical stimulus content plus its acquisition repeat."""

    task: str
    content: str
    run: int

    @property
    def stimulus_id(self) -> str:
        return f"{self.task}:{self.content}"

    @property
    def is_tribev2_skip(self) -> bool:
        return (
            self.task == "friends"
            and self.content in TRIBEV2_FRIENDS_SKIP_CONTENTS
        )


def _normalize_task(task: object) -> str:
    value = str(task).strip().lower()
    if value not in ALGONAUTS_TASKS:
        raise AlgonautsContractError(
            f"Unsupported Algonauts task {task!r}; expected one of {ALGONAUTS_TASKS}"
        )
    return value


def _acquisition_token(value: object) -> str:
    """Return the token after the last ``task-`` acquisition marker."""

    text = str(value).strip().lower()
    if not text:
        raise AlgonautsContractError("Algonauts acquisition label is empty")
    markers = list(re.finditer(r"(?:^|_)task-", text))
    if markers:
        text = text[markers[-1].end() :]
    return text


def canonical_algonauts_stimulus_id(
    task: str,
    value: str | None = None,
    *,
    movie: str | None = None,
    chunk: str | int | None = None,
) -> str:
    """Return a subject/session/repeat-independent stimulus identity."""

    task_name = _normalize_task(task)
    if movie is not None or chunk is not None:
        if movie is None or chunk is None:
            raise AlgonautsContractError("movie and chunk must be supplied together")
        movie_text = str(movie).strip().lower()
        chunk_text = str(chunk).strip().lower()
        if task_name == "movie10":
            try:
                chunk_text = f"{int(chunk_text):02d}"
            except ValueError as exc:
                if not re.fullmatch(r"\d{2}", chunk_text):
                    raise AlgonautsContractError(
                        f"Invalid movie10 chunk {chunk!r}; expected an integer"
                    ) from exc
        stimulus = movie_text + chunk_text
    else:
        if value is None:
            raise AlgonautsContractError(
                "A stimulus value or movie/chunk pair is required"
            )
        stimulus = _acquisition_token(value)
        prefix = f"{task_name}:"
        if stimulus.startswith(prefix):
            stimulus = stimulus[len(prefix) :]
        stimulus = re.sub(r"_run-\d+$", "", stimulus)

    pattern = (
        r"s\d{2}e\d{2}[a-d]"
        if task_name == "friends"
        else r"[a-z][a-z0-9_-]*\d{2}"
    )
    if re.fullmatch(pattern, stimulus) is None:
        raise AlgonautsContractError(
            f"Invalid canonical {task_name} stimulus {stimulus!r}; "
            f"expected pattern {pattern!r}"
        )
    return f"{task_name}:{stimulus}"


def algonauts_identity_from_hdf5_key(task: str, key: str) -> AlgonautsIdentity:
    """Parse one official HDF5 member identity.

    Friends has no acquisition-repeat suffix and uses run 0 in TRIBEv2.
    Movie10 uses run 1 when the suffix is absent; ``_run-N`` is authoritative
    when present.
    """

    task_name = _normalize_task(task)
    token = _acquisition_token(key)
    suffix = re.search(r"_run-(\d+)$", token)
    run = int(suffix.group(1)) if suffix is not None else (0 if task_name == "friends" else 1)
    if run < 0:
        raise AlgonautsContractError(f"Invalid Algonauts acquisition run {run}")
    stimulus_id = canonical_algonauts_stimulus_id(task_name, token)
    return AlgonautsIdentity(task_name, stimulus_id.split(":", 1)[1], run)


def algonauts_identity_from_event(
    task: str,
    stimulus: str,
    run: int | str,
) -> AlgonautsIdentity:
    """Parse an upstream event while validating its authoritative run column."""

    task_name = _normalize_task(task)
    token = _acquisition_token(stimulus)
    try:
        event_run = int(run)
    except (TypeError, ValueError) as exc:
        raise AlgonautsContractError(f"Invalid Algonauts event run {run!r}") from exc
    suffix = re.search(r"_run-(\d+)$", token)
    if suffix is not None and int(suffix.group(1)) != event_run:
        raise AlgonautsContractError(
            f"Stimulus suffix {suffix.group(0)!r} disagrees with event run {event_run}"
        )
    default_run = 0 if task_name == "friends" else 1
    if suffix is None and event_run != default_run:
        raise AlgonautsContractError(
            f"Unsuffixed {task_name} stimulus requires run {default_run}, got {event_run}"
        )
    stimulus_id = canonical_algonauts_stimulus_id(task_name, token)
    return AlgonautsIdentity(task_name, stimulus_id.split(":", 1)[1], event_run)


def normalize_algonauts_subject(subject: object, *, prefixed: bool = False) -> str:
    text = str(subject).strip()
    text = text.removeprefix("Algonauts2025/").removeprefix("sub-")
    if not text.isdigit():
        raise AlgonautsContractError(f"Invalid Algonauts subject {subject!r}")
    number = text.zfill(2)
    return f"sub-{number}" if prefixed else number


def canonical_algonauts_run_id(
    subject: object,
    identity: AlgonautsIdentity,
) -> str:
    subject_number = normalize_algonauts_subject(subject)
    return (
        f"Algonauts2025/{subject_number}/task-{identity.task}/"
        f"{identity.content}/run-{identity.run:02d}"
    )


def canonical_algonauts_raw_timeline(
    subject: object,
    identity: AlgonautsIdentity,
) -> str:
    """Return the timeline label used by the completed TRIBEv2 shards."""

    subject_name = normalize_algonauts_subject(subject, prefixed=True)
    stimulus = identity.content
    if identity.run > 1:
        stimulus = f"{stimulus}_run-{identity.run}"
    return f"{subject_name}:{identity.task}:{stimulus}"


def algonauts_stimulus_paths(
    root: str | Path,
    identity: AlgonautsIdentity,
) -> tuple[Path, Path]:
    """Return the official movie and transcript paths for an identity."""

    base = Path(root)
    if identity.task == "friends":
        movie = identity.content[:3]
        season_folder = f"s{int(movie[1:]):d}"
        video = (
            base
            / "stimuli"
            / "movies"
            / "friends"
            / season_folder
            / f"friends_{identity.content}.mkv"
        )
        transcript = (
            base
            / "stimuli"
            / "transcripts"
            / "friends"
            / season_folder
            / f"friends_{identity.content}.tsv"
        )
        return video, transcript

    match = re.fullmatch(r"(?P<movie>[a-z][a-z0-9_-]*?)(?P<chunk>\d{2})", identity.content)
    if match is None:
        raise AlgonautsContractError(
            f"Cannot resolve movie10 paths for {identity.content!r}"
        )
    movie = match.group("movie")
    chunk = match.group("chunk")
    video = base / "stimuli" / "movies" / "movie10" / movie / f"{movie}{chunk}.mkv"
    transcript = (
        base
        / "stimuli"
        / "transcripts"
        / "movie10"
        / movie
        / f"movie10_{movie}{chunk}.tsv"
    )
    return video, transcript


__all__ = [
    "ALGONAUTS_TASKS",
    "TRIBEV2_FRIENDS_SKIP_CONTENTS",
    "AlgonautsContractError",
    "AlgonautsIdentity",
    "algonauts_identity_from_event",
    "algonauts_identity_from_hdf5_key",
    "algonauts_stimulus_paths",
    "canonical_algonauts_raw_timeline",
    "canonical_algonauts_run_id",
    "canonical_algonauts_stimulus_id",
    "normalize_algonauts_subject",
]
