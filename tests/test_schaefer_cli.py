from __future__ import annotations

import json

import pytest

import neuroworld.nbs_data.schaefer as cli


def _inputs(command: str) -> list[str]:
    return [
        command,
        "--dataset",
        "algonauts",
        "--source-root",
        "/source",
        "--raw-index",
        "/cache/chunks.jsonl",
    ]


def test_prepare_is_read_only_full_inventory_check(monkeypatch, capsys) -> None:
    observed = {}

    def fake_prepare(**kwargs):
        observed.update(kwargs)
        return {"ready": False, "blockers": ["missing raw feature"]}

    monkeypatch.setattr(cli, "prepare_dataset", fake_prepare)
    args = cli.build_parser().parse_args(_inputs("prepare"))
    assert cli.run(args) == 2
    assert observed == {
        "dataset_id": "algonauts",
        "source_root": args.source_root,
        "atlas_manifest": None,
        "raw_index": args.raw_index,
    }
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_run_writes_directly_and_forwards_limit(monkeypatch, capsys) -> None:
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"layout_id": "neuroworld-schaefer1000-v2"}

    monkeypatch.setattr(cli, "run_preparation", fake_run)
    argv = _inputs("run") + [
        "--output",
        "/bundle",
        "--limit",
        "3",
        "--workers",
        "3",
        "--eager-volumes",
    ]
    args = cli.build_parser().parse_args(argv)
    assert cli.run(args) == 0
    assert observed["output"] == args.output
    assert observed["limit"] == 3
    assert observed["workers"] == 3
    assert observed["eager_volumes"] is True
    assert "stage_root" not in observed
    assert "destination" not in observed
    assert json.loads(capsys.readouterr().out)["layout_id"].endswith("v2")


@pytest.mark.parametrize("command", ("doctor", "build", "validate", "publish", "atlas"))
def test_cli_has_only_prepare_and_run(command) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command])


@pytest.mark.parametrize(
    "command,forbidden",
    [
        ("prepare", "--limit"),
        ("prepare", "--run-id"),
        ("prepare", "--asset-root"),
        ("run", "--stage-root"),
        ("run", "--destination"),
        ("run", "--run-id"),
        ("run", "--smoothing-fwhm-mm"),
    ],
)
def test_cli_rejects_removed_options(command, forbidden) -> None:
    argv = _inputs(command)
    if command == "run":
        argv.extend(["--output", "/bundle"])
    argv.append(forbidden)
    if forbidden in {
        "--limit",
        "--run-id",
        "--asset-root",
        "--stage-root",
        "--destination",
        "--smoothing-fwhm-mm",
    }:
        argv.append("value")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "false"))
def test_run_requires_positive_integer_limit(value) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            _inputs("run") + ["--output", "/bundle", "--limit", value]
        )


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "false"))
def test_run_requires_positive_integer_workers(value) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            _inputs("run") + ["--output", "/bundle", "--workers", value]
        )
