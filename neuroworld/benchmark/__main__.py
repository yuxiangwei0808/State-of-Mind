"""Benchmark command line (docs/benchmark.md).

    python -m neuroworld.benchmark forecast --dataset algonauts --split val dev --model run:<dir> --device cuda
    python -m neuroworld.benchmark report --dataset algonauts --split dev
    python -m neuroworld.benchmark state --model run:<dir> --device cuda              (Movie10 repeats; HCP repeated clip)
    python -m neuroworld.benchmark updating --model run:<dir> --device cuda
    python -m neuroworld.benchmark nonanticipation --model run:<dir> --device cuda
    python -m neuroworld.benchmark suite --dataset hcp_movies --model run:<dir> --device cuda   (every task, one fit)
    python -m neuroworld.benchmark profile [--reference <row>]                         (the card: every scored model)

Models: ``py:<file.py>:<Class>`` (an external model: any class with ``name`` and ``predict(QueryBatch)``, built as
``Class(dataset=..., device=...)``; examples/persistence.py), ``run:<dir>`` (a trained run),
``direct-linear`` (the Forecast and State baseline), ``ipsid`` (the Updating baseline), ``tribev2``, ``persistence``,
``latent:<model>`` (a latent model's two readouts: its own decoder, and its predicted states through a frozen linear
probe, named "<name>" and "<name> (latent)"; benchmark/latent.py).
``--features`` chooses the stimulus rows of the fitted references (raw, projections or context).
Results go to <output>/<dataset>/<split>/<model>/ (Forecast) or <output>/<dataset>/<task>/ (the others).
"""
import argparse
import json
from pathlib import Path

CONFIGS = {"algonauts": "neuroworld/configs/algonauts/base.yaml", "hcp_movies": "neuroworld/configs/hcp_movies/base.yaml"}
OUTPUT = "artifacts/benchmark"
# Evaluation sets: every split, each tagged with its exposure; everything but the training data is test data. Algonauts:
# Friends S1-5 (training), S6 (validation), Movie10 Hidden Figures and Life (development), Bourne and Wolf (test). HCP:
# training participants' training frames, their held-out clips (validation) and run 4 (test), new participants' run 1
# (calibration), runs 2-3 and run 4.
EVALUATION = {"algonauts": ["train", "val", "dev", "test"],
              "hcp_movies": ["train", "val", "test", "heldout-fit", "heldout-adapt", "heldout-test"]}
NONANTICIPATION = {"algonauts": "val", "hcp_movies": "heldout-adapt"}  # one held-out set suffices for the check


def _rows_cache(data, kind, output=OUTPUT):
    """Stimulus-row cache of a dataset, under the run's output directory."""
    return Path(output) / data.dataset / f"stimulus-rows-{kind}.npz"


def _log(text):
    print(text, flush=True)


def _models(name, data, device="cpu", output=OUTPUT, features=None):
    from neuroworld.benchmark.interface import Persistence
    if name.startswith("latent:"):  # both readouts of a latent model: its own decoder, and its frozen probe
        from neuroworld.benchmark.latent import LatentProbe, Probe
        model = _models(name.split(":", 1)[1], data, device, output, features)[0]
        if getattr(model, "fills_with_probe", False):  # no decoder: the first probe fills the rollouts the final probe is fitted on
            model.use_probe(Probe(data, model, output, log=_log, open_loop_only=True))
            probe = Probe(data, model, output, log=_log)
            model.use_probe(probe)
            return [LatentProbe(model, probe, f"{model.name} (latent)")]
        own = [model] if getattr(model, "decodes", True) else []  # a model without its own decoder has only the probe row
        return own + [LatentProbe(model, Probe(data, model, output, log=_log), f"{model.name} (latent)")]
    if name == "persistence":
        return [Persistence()]
    if name == "tribev2":
        from neuroworld.benchmark.references import SavedEncoder
        return [SavedEncoder(data)]
    if name.startswith("py:"):  # an external model: py:<file.py>:<Class>, built as Class(dataset=..., device=...)
        import importlib.util
        path, cls = name[3:].rsplit(":", 1)
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return [getattr(module, cls)(dataset=data.dataset, device=device)]
    if name == "ipsid":
        from neuroworld.benchmark.baselines import IPSIDBaseline
        return [IPSIDBaseline(data, features, log=_log, output=output)]
    if name.startswith("run:"):
        from neuroworld.benchmark.adapters import load_run
        return [load_run(name.split(":", 1)[1], device)]
    if name == "direct-linear":
        from neuroworld.benchmark.references import FEATURES, direct_linear
        kind = features or FEATURES[data.dataset]
        return [direct_linear(data, kind, cache=_rows_cache(data, kind, output), log=_log)]
    raise ValueError(f"Unknown model {name!r}")


def _each_model(args, data):
    from neuroworld.benchmark.latent import LatentProbe
    for model in _models(args.model, data, args.device, args.output, getattr(args, "features", None)):
        if args.name:
            model.name = f"{args.name} (latent)" if isinstance(model, LatentProbe) else args.name
        yield model


def _splits(args, data):
    return args.split or EVALUATION.get(data.dataset, ["val", "dev"])


def forecast(args):
    from neuroworld.benchmark import forecast as F
    from neuroworld.benchmark.data import BenchmarkData
    data = BenchmarkData(args.config or CONFIGS[args.dataset])
    for model in _each_model(args, data):
        for split in _splits(args, data):
            from neuroworld.benchmark.adapt import calibrate
            calibrate(model, data, split, log=_log)
            F.run(data, model, split, output=args.output, workers=args.workers, log=_log)


def report(args):
    """Forecast scorecard of every scored model on one split, with paired differences versus a reference; with
    --task state or updating, that task's table instead."""
    from neuroworld.benchmark import forecast as F, report as R
    if args.task in ("state", "updating"):
        from importlib import import_module
        module = import_module(f"neuroworld.benchmark.{args.task}")
        reference = args.reference or ("direct linear" if args.dataset == "hcp_movies" else "direct linear [projections]")
        folder = Path(args.output) / args.dataset / args.task
        text = module.table(folder, reference)
        (folder / "table.md").write_text(text + "\n")
        print(text)
        return
    folder = Path(args.output) / args.dataset / args.split
    results = {path.name: F.load(path) for path in sorted(folder.iterdir()) if (path / "results.json").is_file()}
    if not results:
        raise FileNotFoundError(f"No scored models under {folder}")
    summaries = {name: R.summarize(r) for name, r in results.items()}
    reference = args.reference or ("direct linear" if args.dataset == "hcp_movies" else "direct linear [projections]")
    unit = "participants" if args.dataset == "hcp_movies" else "content parts"
    lines = [f"# Forecast: {args.dataset} {args.split}", "",
             f"Drift-robust r and nMSE; 95% bootstrap intervals over {unit}.", "", R.scorecard(summaries)]
    if reference in results:
        lines += ["", f"Paired differences versus {reference}:", ""]
        diffs = {name: R.paired(r, results[reference]) for name, r in results.items() if name != reference}
        lines.append(R.scorecard(diffs))
    (folder / "scorecard.md").write_text("\n".join(lines) + "\n")
    encode = lambda s: {" / ".join(map(str, k)): v for k, v in s.items()}
    (folder / "summary.json").write_text(json.dumps({n: encode(s) for n, s in summaries.items()}, indent=1) + "\n")
    print("\n".join(lines))


def state(args):
    from neuroworld.benchmark import state as S
    from neuroworld.benchmark.data import BenchmarkData
    data = BenchmarkData(args.config or CONFIGS[args.dataset])
    for model in _each_model(args, data):
        S.run(data, model, splits=args.split, output=args.output, log=_log)


def updating(args):
    from neuroworld.benchmark import updating as U
    from neuroworld.benchmark.data import BenchmarkData
    data = BenchmarkData(args.config or CONFIGS[args.dataset])
    for model in _each_model(args, data):
        U.run(data, model, splits=_splits(args, data), output=args.output, log=_log, own_fill=args.own_fill)


def nonanticipation(args):
    from neuroworld.benchmark import nonanticipation as N
    from neuroworld.benchmark.data import BenchmarkData
    data = BenchmarkData(args.config or CONFIGS[args.dataset])
    for model in _each_model(args, data):
        N.run(data, model, output=args.output, split=NONANTICIPATION[data.dataset], log=_log)


TASKS = ("forecast", "state", "updating", "nonanticipation")


def profile(args):
    from neuroworld.benchmark import profile as P
    print(P.run(args.output, args.reference))


def suite(args):
    """Every real-data task, in one process (a fitted reference is fitted once). The models of one spec (a latent
    model's two readouts) are scored together, so they share its rollouts."""
    from neuroworld.benchmark import forecast as F, nonanticipation as N, state as S, updating as U
    from neuroworld.benchmark.adapt import calibrate
    from neuroworld.benchmark.data import BenchmarkData
    data = BenchmarkData(args.config or CONFIGS[args.dataset])
    splits = args.split or EVALUATION[data.dataset]
    models = list(_each_model(args, data))
    for task in args.tasks:
        _log(f"suite {', '.join(m.name for m in models)}: {task}")
        if task == "forecast":
            for split in splits:
                for model in models:
                    calibrate(model, data, split, log=_log)
                F.run(data, models, split, output=args.output, workers=args.workers, log=_log)
        elif task == "state":
            S.run(data, models, output=args.output, log=_log)
        elif task == "updating":
            U.run(data, models, splits=splits, output=args.output, log=_log)
        elif task == "nonanticipation":
            for model in models:
                N.run(data, model, output=args.output, split=NONANTICIPATION[data.dataset], log=_log)


def _raise_open_file_limit():
    """Scoring many runs keeps many fMRI arrays memory-mapped; some hosts default to 1024 open files."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or hard > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536, hard))


def main():
    _raise_open_file_limit()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help, dataset=True, splits=("val", "dev"), output=OUTPUT):
        p = sub.add_parser(name, help=help)
        p.add_argument("--model", required=True, help="See the module docstring for the model names")
        p.add_argument("--name", help="Result name for a single model (default: its own name)")
        p.add_argument("--device", default="cpu", help="cuda for trained models on a GPU")
        p.add_argument("--output", default=output, required=output is None)
        if dataset:
            p.add_argument("--dataset", choices=sorted(CONFIGS), default="algonauts")
            p.add_argument("--features", choices=("raw", "projections", "context"),
                           help="Stimulus rows of the fitted references (default: raw on Algonauts, projections on HCP)")
        p.add_argument("--config", required=not dataset, help="Dataset config overriding the default")
        if splits:
            p.add_argument("--split", nargs="+", default=None if splits == "evaluation" else list(splits),
                           help="Default: the dataset's evaluation sets (Algonauts val dev; HCP heldout-adapt)"
                           if splits == "evaluation" else None)
        return p

    command("forecast", "Forecast task on one or more splits", splits="evaluation").add_argument(
        "--workers", type=int, default=1, help="Forked CPU workers (CPU models only)")
    command("state", "State task on repeated viewings", splits=None).add_argument(
        "--split", nargs="+", help="Default: dev (Algonauts: only Movie10 has repeats); heldout-adapt (HCP)")
    command("updating", "Updating task: error against the number of revealed frames", splits="evaluation").add_argument(
        "--own-fill", action="store_true", help="Fill the frames not shown with the model's own forecast")
    command("nonanticipation", "Nonanticipation check", splits="evaluation")
    s = command("suite", "Every real-data task for one model", splits=None)
    s.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    s.add_argument("--workers", type=int, default=1, help="Forked CPU workers for Forecast (CPU models only)")
    s.add_argument("--split", nargs="+", help="Forecast and Updating splits (default: every non-locked split); State "
                                             "uses its own groups")
    pr = sub.add_parser("profile", help="The card: every scored model's Forecast, State and Updating per panel")
    pr.add_argument("--output", default=OUTPUT)
    pr.add_argument("--reference", help="Row for the paired differences (default: none)")
    r = sub.add_parser("report", help="Forecast scorecard of every scored model on one split")
    r.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    r.add_argument("--split", choices=("train", "val", "dev", "test", "heldout-fit", "heldout-adapt", "heldout-test"),
                   help="Forecast split")
    r.add_argument("--task", choices=("forecast", "state", "updating"), default="forecast")
    r.add_argument("--reference", help="Model for paired differences (default: the direct linear forecaster on "
                                       "TRIBEv2 projections)")
    r.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    {"forecast": forecast, "report": report, "state": state, "updating": updating,
     "nonanticipation": nonanticipation, "suite": suite, "profile": profile}[args.command](args)


if __name__ == "__main__":
    main()
