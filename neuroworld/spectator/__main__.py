"""python -m neuroworld.spectator <step> ...: Spectator's closed-form steps, its files and its scores (see __init__.py)."""
import argparse
import json

from neuroworld.utils.config import load_config, validate_config


def _config(path, encoder=None, predictor=None, overrides=()):
    """A resolved config with its encoder and start-matrix files replaced (train.encoder_checkpoint and
    train.predictor_checkpoint are fixed by the design, so --set may not change them)."""
    cfg = load_config(path, overrides)
    if encoder:
        cfg["train"]["encoder_checkpoint"] = str(encoder)
    if predictor:
        cfg["train"]["predictor_checkpoint"] = str(predictor)
    return validate_config(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neuroworld.spectator", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)

    s = steps.add_parser("config", help="A config with its encoder and start-matrix files replaced, for launch_stage1.sh")
    s.add_argument("config")
    s.add_argument("out")
    s.add_argument("--encoder")
    s.add_argument("--predictor")
    s.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")

    s = steps.add_parser("lags", help="Step 2: the slow latent's lag weights")
    s.add_argument("source", help="Stage-0 best.pt (Algonauts) or an encoder file whose fast map to keep (HCP)")
    s.add_argument("out")
    s.add_argument("--frames", type=int, required=True, help="Window frames: 120 on Algonauts, 180 on HCP (180 s)")
    s.add_argument("--ahead", type=int, required=True, help="Target frames: 16 on Algonauts, 24 on HCP (24 s)")
    s.add_argument("--dataset-config", default="neuroworld/configs/algonauts/base.yaml")

    s = steps.add_parser("matrix", help="Step 4 (and the JEPA phase's start): the closed-form matrix")
    s.add_argument("config")
    s.add_argument("out")
    s.add_argument("--encoder", help="Encoder file (sets train.encoder_checkpoint)")
    s.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    s.add_argument("--penalty", type=float, default=1e-3)
    s.add_argument("--movie-factor", type=float, nargs="+", default=[100.0])
    s.add_argument("--state-factor", type=float, default=10.0)
    s.add_argument("--full-windows-only", action="store_true", help="No partial-history windows (the JEPA phase's start)")
    s.add_argument("--rollmix", help="Also fit on this matrix's own rollout states (an ablation)")

    s = steps.add_parser("encoder", help="The JEPA phase's trained encoder, in the format later steps load")
    s.add_argument("run_best")
    s.add_argument("start_encoder")
    s.add_argument("out")

    s = steps.add_parser("assemble", help="The model's run directory from its encoder and matrix")
    s.add_argument("config")
    s.add_argument("matrix")
    s.add_argument("run_dir")
    s.add_argument("--encoder")
    s.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")

    s = steps.add_parser("score", help="Every benchmark task, with the model's own probe")
    s.add_argument("run_dir")
    s.add_argument("--seconds", type=float, default=180.0)
    s.add_argument("--dataset", choices=("algonauts", "hcp_movies"), default="algonauts")
    s.add_argument("--store", required=True, help="Benchmark result store")
    s.add_argument("--name", help="Row name (default: the run directory's name)")
    s.add_argument("--gap", action="store_true", help="Also Forecast on the training recordings")
    s.add_argument("--no-check", action="store_true", help="Skip the future-frames check")
    s.add_argument("--data-config", help="The benchmark's dataset config (default the package's base.yaml), e.g. one with fewer training runs")

    args = parser.parse_args(argv)
    if args.step == "config":
        import yaml
        from pathlib import Path
        Path(args.out).write_text(yaml.safe_dump(_config(args.config, args.encoder, args.predictor, args.set)))
    elif args.step == "lags":
        from neuroworld.spectator.lags import fit_lags
        print(json.dumps(fit_lags(args.source, args.dataset_config, args.frames, args.ahead, args.out), indent=1))
    elif args.step == "matrix":
        from neuroworld.spectator.matrix import fit_matrix
        fit_matrix(_config(args.config, args.encoder, None, args.set), args.out, penalty=args.penalty, movie_factors=args.movie_factor,
                   state_factor=args.state_factor, partial=not args.full_windows_only, rollmix=args.rollmix)
    elif args.step == "encoder":
        from neuroworld.spectator.build import encoder_file
        print("changed:", encoder_file(args.run_best, args.start_encoder, args.out))
    elif args.step == "assemble":
        from neuroworld.spectator.build import assemble
        print("wrote", assemble(_config(args.config, args.encoder, None, args.set), args.matrix, args.run_dir))
    elif args.step == "score":
        from neuroworld.spectator.score import score
        score(args.run_dir, args.seconds, args.dataset, args.store, name=args.name, gap=args.gap, check=not args.no_check,
              data_config=args.data_config)


if __name__ == "__main__":
    main()
