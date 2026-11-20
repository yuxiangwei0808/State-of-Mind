# Reproducing the paper

This page maps each table and figure of the paper to the code that computes it and the runs and result stores it reads. Paths are relative to the output root `$NEUROWORLD_OUTPUT` unless they start with another root (see [`neuroworld/paths.py`](../neuroworld/paths.py)); `<paper>` is the paper's repository, whose `viz/` folder draws most figures from the results named here.

## Result stores

Scores are stored per row (a model, or a model with its probe: `<name> (latent)`) in benchmark result stores ([benchmark](benchmark.md#commands)). The paper's scripts search them in this order (`ROOTS` in [`paper/analysis.py`](../neuroworld/paper/analysis.py)):

| Store | Holds |
|---|---|
| `benchmark-final` | the references rescored on the final protocol, on every split: TRIBE v2, and IPSID and NeuroWorld k1 at each history length |
| `benchmark-test` | *Bourne*, *Wolf* and HCP run 4 for Spectator, its variants, the Transformer rows and the linear forecaster, and the probes of the Spectator rows |
| `benchmark-iter-overflow` | Friends S6 and the first two Movie10 films for Spectator, its variants and the linear forecaster |
| `benchmark-history-context` | the linear forecaster at 120 s |
| `benchmark-ablations` | the ablation rows and their control |
| `$NEUROWORLD_EXTERNAL/baselines/...` | BrainVista's and MIRAGE's rows and saved forecasts (trained and scored by a collaborator with the published code) |

## Runs

Run folders are under `checkpoints/algonauts/wave2_20261102/` unless noted; each holds its resolved `config.yaml`, `best.pt` and logs.

| Model | Run | Benchmark row |
|---|---|---|
| Spectator, 180 s | `ridge-jepa-180s-std-partial` (frozen copy: `spectator-v1/`) | `ridge-jepa-180s-std-partial (latent)` |
| Spectator, other history lengths | `ridge-jepa-{7.5,15,30,60,90,120}s-std-partial` | `… (latent)` |
| Spectator on HCP | `ridge-jepa-180s-hcp-refit-m1e4` | `ridge-jepa-180s-hcp-refit-m1e4 (latent)` |
| Stage 0 | `checkpoints/algonauts/state_20261030/phase0-k5` | |
| JEPA phase | `D180` (encoder `history-D180.pt`) | |
| Previous final fit (twin of the encoder/predictor ablations) | `D180-win5`; frozen-encoder row `D180-start-win5` | `step2 D180-win5 q180 (latent)` |
| Input ablations (no movie, no fMRI) | `ridge-jepa-180s-std-partial-nomovie`, `-nofmri` | `… (latent)` |
| Other Spectator ablations | `ridge-jepa-180s-{noslow,m1,m10,roll80}`; `checkpoints/algonauts/ablations_20261113/` (control, random init, online target, mean slow weights, 3-frame window, state size 512/2,048, no video/text/audio) | see `paper/ablations.py` |
| Transformer encoder / predictor / both | `ridge-tfforecast-jepa-180s`, `transformer-jepa-180s-2x1024`, `transformer-tfforecast-linpath-jepa-180s-2x1024` (with `stage0-tfforecast-linpath`, `matrix-tfforecast-linpath-jepa-180s`) | `… (latent)` |
| Linear forecaster | fitted inside the benchmark ([`experiments/history_length.py`](../neuroworld/experiments/history_length.py), [`benchmark/references.py`](../neuroworld/benchmark/references.py)) | `linear [context] <L> s` |
| IPSID | fitted inside the benchmark ([`benchmark/baselines.py`](../neuroworld/benchmark/baselines.py)) | `IPSID [context] <L> s` (`IPSID [context]` at 120 s) |
| NeuroWorld (k1, with history blocks) | `checkpoints/algonauts/neuroworld_history_20261112/k1-<L>s` (7.5 s: `$NEUROWORLD_EXTERNAL/NeuralWorld/checkpoints/...`); HCP: `checkpoints/hcp_movies/neuroworld_k1_20261115` | `NeuroWorld k1 <L>s (latent)`, `NeuroWorld-style k1 (latent)` at 7.5 s |
| TRIBE v2 | the released model, zero-shot ([`eval/tribev2_reference.py`](../neuroworld/eval/tribev2_reference.py)); predictions in `$NEUROWORLD_DATA/<dataset>/derivatives/neuroworld-schaefer1000-v2-tribev2-predictions` | `TRIBEv2 zero-shot` |

Every deleted run and file is listed in `archive/runs-20261112/removed-files.tsv` and `archive/cleanup-20261119/removed.tsv`; their scores stay in the stores.

## Main paper

| Item | Code | Reads, writes |
|---|---|---|
| Main table (Forecast, State, Updating at 7.5, 120, 180 s; r) | `python -m neuroworld.paper.history_table` | the stores; writes `paper-analysis/results-history-table.json`. The linear forecaster's Updating is scored with `updating --own-fill` (stores `paper-analysis/linear-updating-ownfill/{budgets180,valdev,test}`); BrainVista's rows from `python -m neuroworld.paper.brainvista` (`results-brainvista.json`) |
| Inputs table (observations only, stimulus only, another individual) | `<paper>/viz/inputs/compute.py`, with [`paper/ablations.py`](../neuroworld/paper/ablations.py)'s helpers | the input-ablation rows |
| Ablation table | `python -m neuroworld.paper.ablations` | `benchmark-ablations` and the stores; writes `paper-analysis/results-ablations.json` |
| Time and cortex figures | `python -m neuroworld.paper.collect spectator ...` (forecasts and the state/movie split), `python -m neuroworld.paper.analysis collected|isc|maps`; drawn by `<paper>/viz/contributions/` | `paper-analysis/spectator`, `inputs-aligned/`, `parcel-maps.npz`, `results-isc.json` |

## Appendix

| Item | Code | Reads, writes |
|---|---|---|
| Settings table | the frozen configs, [`configs/spectator/`](../neuroworld/configs/spectator) | |
| Baseline settings | `benchmark/references.py`, `benchmark/baselines.py`, `experiments/history_length.py`, `paper/brainvista.py`; NeuroWorld's run READMEs | |
| Efficiency table | scripts in `paper-analysis/extra-nov14/efficiency/` (outside this repository) | `ours/efficiency-ours.json`, `efficiency-gpu.json` |
| History figure, properties figure | `<paper>/viz/history/draw_figure5.py`, `viz/properties/draw_figure6.py`, using [`paper/figures.py`](../neuroworld/paper/figures.py)'s helpers (`TABLE2`, `AT_180`, `with_ownfill`, `with_external`) | the stores |
| Significance table | `python -m neuroworld.paper.history_table` (`spectator_minus`) | `results-history-table.json` |
| Per-viewer table | `paper-analysis/extra-nov14/viewers/per_viewer.py`, `with_viewer.py`; `linear-updating-ownfill/viewers180.py` (outside this repository) | `per_viewer*.json` |
| Prior work's metrics (r, global r, Cls@Top10) | `python -m neuroworld.paper.standard_metrics save|compute` and `python -m neuroworld.paper.brainvista_crossfit` | writes `results-standard-metrics.json`, `results-brainvista-crossfit.json` (the saved forecasts were deleted; `save` rebuilds them) |
| HCP tables | `paper-analysis/extra-nov14/hcp/run_models.py`, `extra-nov14/hcp-mirage/`, `extra-nov14/hcp-brainvista/`, `extra-nov15/hcp-intervals/`, `extra-nov15/hcp-prior-metrics/`, `linear-updating-ownfill/hcp180.py` (outside this repository) | `table.json`, `intervals.json`, `absolute.json` there |
| New viewer table | `python -m neuroworld.paper.new_viewer select|fit|score|report` | `paper-analysis/extra-nov13/new-viewer/results.json` |
| Ablations by seconds ahead | `python -m neuroworld.paper.ablations` | |
| Long-horizon figure | `python -m neuroworld.paper.long_horizon run|report`; drawn by `<paper>/viz/long_horizon/` | `paper-analysis/extra-nov13/long-horizon/results.json` |
| State/movie swap table | `python -m neuroworld.paper.collect spectator` at every starting point, then `<paper>/viz/contributions/state_movie_table.py`, `state_swap.py` | `paper-analysis/spectator-all/` (decomposition matrices kept; forecasts deleted), `inputs-aligned/*.json` |
| "Relies on" figure | `<paper>/viz/contributions/gradshap_forecast.py`, `fig_relies_on.py` | |
| Inside-the-state figure | `python -m neuroworld.paper.analysis collected`, `python -m neuroworld.paper.figures inside` | `results-collected.json` |
| State-dimension figure | `<paper>/viz/state_dim/` | |
| Data-size figure and table | Spectator retrained on 10/25/50% of the episodes with the release configs (`paper-analysis/extra-nov13/data-size/run.sh`), students with `python -m neuroworld.paper.generated fit --data-config ...`; drawn by `<paper>/viz/data_size/` | `paper-analysis/extra-nov13/data-size/table.json` |
| Models trained on Spectator's forecasts | linear forecaster: `python -m neuroworld.paper.generated fit|report`; Transformer predictor: `neuroworld.train` with `train.teacher_run`; NeuroWorld: `python -m neuroworld.paper.neuroworld_generated` | `paper-analysis/extra-nov13/{generated,neuroworld-student}/table.json`, `generated/stage2/table.json` |

The experiments of the last three rows and the new-viewer and long-horizon rows are described with every result in `paper-analysis/extra-nov13/synthetic-data-proposal.md`.
