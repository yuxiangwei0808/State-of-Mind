import pytest
from neuroworld.data.algonauts_contract import AlgonautsContractError, canonical_algonauts_stimulus_id

def test_canonical_ids_drop_subject_session_and_repeat_run() -> None:
    assert canonical_algonauts_stimulus_id(
        "movie10", "ses-099_task-life01_run-2"
    ) == "movie10:life01"
    assert canonical_algonauts_stimulus_id(
        "friends", movie="s03", chunk="e12b"
    ) == "friends:s03e12b"
    with pytest.raises(AlgonautsContractError):
        canonical_algonauts_stimulus_id("friends", "../../s03e12b")


def test_event_grouping_deduplicates_subject_and_repeat_run() -> None:
    pd = pytest.importorskip("pandas")
    from neuroworld.nbs_data.stimuli.common import iter_canonical_event_groups

    rows = []
    for subject in ("sub-01", "sub-02"):
        for run in (1, 2):
            timeline = f"{subject}:movie10:life01" + ("" if run == 1 else "_run-2")
            common = {
                "task": "movie10",
                "movie": "life",
                "chunk": "1",
                "run": run,
                "subject": subject,
                "timeline": timeline,
                "start": 0.0,
                "duration": 3.0,
                "stop": 3.0,
            }
            rows.extend([common | {"type": "Video"}, common | {"type": "Audio"}])
    groups = list(iter_canonical_event_groups(pd.DataFrame(rows)))
    assert len(groups) == 1
    assert groups[0].stimulus_id == "movie10:life01"
    assert groups[0].events["subject"].nunique() == 1
    assert groups[0].events["run"].nunique() == 1


def test_timeline_grouping_preserves_repeated_contexts_and_general_ids() -> None:
    pd = pytest.importorskip("pandas")
    from neuroworld.nbs_data.stimuli.common import iter_timeline_event_groups

    rows = []
    for subject in ("sub-01", "sub-02"):
        common = {
            "subject": subject,
            "timeline": f"HCP7T/{subject}/task-movie/run-01",
            "stimulus_id": "hcp/movie-0001",
            "start": 0.0,
            "duration": 3.0,
            "stop": 3.0,
        }
        rows.extend([common | {"type": "Video"}, common | {"type": "Audio"}])
    groups = list(iter_timeline_event_groups(pd.DataFrame(rows)))
    assert [group.timeline_id for group in groups] == [
        "HCP7T/sub-01/task-movie/run-01",
        "HCP7T/sub-02/task-movie/run-01",
    ]
    assert {group.stimulus_id for group in groups} == {"hcp/movie-0001"}

    selected = list(
        iter_timeline_event_groups(
            pd.DataFrame(rows),
            selected_timeline_ids={
                "HCP7T/sub-02/task-movie/run-01"
            },
        )
    )
    assert [group.timeline_id for group in selected] == [
        "HCP7T/sub-02/task-movie/run-01"
    ]

    with pytest.raises(ValueError, match="Requested timelines are absent"):
        list(
            iter_timeline_event_groups(
                pd.DataFrame(rows),
                selected_timeline_ids={"missing/run"},
            )
        )
    with pytest.raises(ValueError, match="timeline or stimulus"):
        list(
            iter_timeline_event_groups(
                pd.DataFrame(rows),
                selected_timeline_ids={
                    "HCP7T/sub-01/task-movie/run-01"
                },
                selected_stimulus_ids={"hcp/movie-0001"},
            )
        )
