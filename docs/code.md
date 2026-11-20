# Code and training runtime

This page maps the code in [`neuroworld/`](../neuroworld) and describes the training runtime: the design configs, the two training stages, their launchers and the files a run writes. The model is described in [Spectator](spectator.md), the benchmark in [BrainStateBench](benchmark.md), data preparation in [data preparation](data-preparation.md), and the paper's tables in [reproduce](reproduce.md). Commands run from the repository root. In tables and text, file paths are relative to `neuroworld/`, except `tests/`, `examples/`, `pytest.ini` and the output folders `checkpoints/` and `artifacts/`, which are in the repository root.

## Code map

| Part | Files | Role |
|---|---|---|
| Locations | `paths.py` | The data, output and external roots (`NEUROWORLD_DATA`, `NEUROWORLD_OUTPUT`, `NEUROWORLD_EXTERNAL`) and `expand`, which substitutes them in configs |
| Configuration | `utils/config.py`, `utils/seed.py`, `configs/algonauts/`, `configs/hcp_movies/base.yaml`, `configs/spectator/`, `configs/accelerate.yaml` | Defaults, designs, loading, overrides, validation and run names; per-dataset settings; Spectator's frozen configs; Accelerate launch defaults |
| Data loading | `data/bundle.py`, `data/splits.py`, `data/schaefer_v2_dataset.py`, `data/algonauts_contract.py` | Read prepared bundles; select each split's runs and frame ranges; cut training and evaluation windows and align stimulus rows to brain frames; Algonauts identifiers |
| Data preparation | `nbs_data/` | Offline bundle building ([data preparation](data-preparation.md)) |
| Models | `models/world_model.py`, `models/forecaster.py`, `models/encoder.py`, `models/predictor.py`, `models/lewm_blocks.py`, `models/collapse.py`, `models/stimulus.py`, `models/stimulus_context.py`, `models/tribev2.py`, `models/readout.py` | The latent world model and the fMRI forecaster (`build_model`), fMRI encoders, predictors, LeWM blocks, SIGReg, stimulus rows, the released TRIBE v2 network, and the per-subject readout |
| Training | `train/latent.py`, `train/losses.py`, `train/readout.py`, `train/stage_utils.py`, `train/checkpoint.py`, `train/runtime.py`, `train/tracking.py` | Stage-1 loop and objectives; stage-2 readout; datasets, loaders and initialization; checkpoints and resume; Accelerate setup; W&B logging |
| Evaluation helpers | `eval/rollout.py`, `eval/probe.py`, `eval/fmri.py`, `eval/intersubject.py`, `eval/linear_baselines.py`, `eval/tribev2_reference.py` | Validation rollouts and metrics, the checkpoint-selection ridge probe, fMRI correlation statistics, inter-subject references, shared pieces of the linear references, TRIBE v2's zero-shot predictions |
| Spectator | `spectator/` (`lags.py`, `matrix.py`, `build.py`, `score.py`), `scripts/spectator.sh` | Spectator's closed-form steps, its run directory and its scores ([Spectator](spectator.md)) |
| Benchmark | `benchmark/` | BrainStateBench: queries, tasks, scoring, references, adapters and latent probes ([BrainStateBench](benchmark.md)) |
| Experiments | `experiments/history_length.py` | The linear forecaster, IPSID and a running mean scored at each history length |
| Paper analyses | `paper/` | The paper's tables and figures from the stored results ([reproduce](reproduce.md)) |
| Launchers | `scripts/launch_stage1.sh`, `scripts/launch_stage2.sh`, `scripts/spectator.sh` | Shell scripts for the two training stages and for Spectator end to end |
| Tests and example | `tests/`, `pytest.ini`, `examples/persistence.py` | Unit and contract tests on synthetic inputs; a model scored by the benchmark |
| Licenses | `licenses/` | Upstream licenses of adapted code ([below](#sources-and-adaptations)) |

### Command modules

Each command is the `main()` of the module named; `scripts/` holds only shell scripts.

| `python -m ...` | Calls |
|---|---|
| `neuroworld.train.latent <config>` | `train/latent.py::fit`: stage 1 |
| `neuroworld.train.readout <config> --checkpoint <best.pt>` | `train/readout.py::fit_readout`: stage 2 |
| `neuroworld.spectator config` / `lags` / `matrix` / `encoder` / `assemble` / `score` | `spectator/__main__.py::_config`, `lags.py::fit_lags`, `matrix.py::fit_matrix`, `build.py::encoder_file`, `build.py::assemble`, `score.py::score` |
| `neuroworld.benchmark forecast` / `state` / `updating` / `nonanticipation` | `benchmark/forecast.py::run`, `state.py::run`, `updating.py::run`, `nonanticipation.py::run` |
| `neuroworld.benchmark suite` / `profile` / `report` | every task for one model; `benchmark/profile.py::run` (the card); `benchmark/report.py::summarize`, `paired`, `scorecard` |
| `neuroworld.experiments.history_length run` / `report` | the history-length study, written in the benchmark's formats under `artifacts/history-length/` |
| `neuroworld.eval.tribev2_reference <run dir or config>` | `eval/tribev2_reference.py::run`: caches TRIBE v2's zero-shot predictions per stimulus and scores them per subject and against the group mean |
| `neuroworld.paper.<module>` | the paper's analyses, listed in [reproduce](reproduce.md) |
| `neuroworld.nbs_data.schaefer`, `neuroworld.nbs_data.stimuli.materialize`, `neuroworld.nbs_data.stimuli.tribev2_context` | bundle preparation ([data preparation](data-preparation.md)) |

The training, Spectator, benchmark, experiment and evaluation commands take `--help`.

## Training runtime

### Designs and config files

A training config is a **design YAML** that names a per-dataset **base YAML** and declares a `design`. The design fixes four settings, `train.objective`, `train.mode`, `train.freeze_encoder` and `train.encoder_checkpoint` (`DESIGNS`, `LOCKED` in `utils/config.py`); everything else comes from the base or an override.

| Design | Objective | Input | Encoder | Config |
|---|---|---|---|---|
| `fmri_forecaster` | `fmri`: next-frame fMRI error; encoder, predictor and readout train together in one stage | fMRI and stimulus | trained | `configs/algonauts/fmri_forecaster.yaml` |
| `joint_world_model` | `latent`: next-latent error plus SIGReg | fMRI and stimulus | trained | `configs/algonauts/joint_world_model.yaml` |
| `stage0_forecaster` | `fmri` | fMRI only | trained | `configs/algonauts/stage0_forecaster.yaml`, `configs/spectator/algonauts-stage0.yaml` |
| `jepa_stage0` | `latent` | fMRI only | trained | none |
| `transformer_world_model` | `latent` | fMRI and stimulus | frozen, loaded from `train.encoder_checkpoint` | `configs/algonauts/transformer_world_model.yaml`, `configs/spectator/algonauts.yaml`, `configs/spectator/hcp.yaml` |
| `transformer_world_model_finetuned` | `latent` | fMRI and stimulus | loaded from `train.encoder_checkpoint`, trained further | `configs/spectator/algonauts-jepa.yaml` |

- `fmri` objective (`train/losses.py::fmri_loss`): the readout decodes each predicted state to parcels, and rollouts feed each decoded frame back through the encoder. It has no stage 2.
- `latent` objective (`train/losses.py::latent_loss`): the predictor is scored against encoded states; SIGReg is added while the encoder trains. Parcels come later from the stage-2 readout or a probe.
- Without the stimulus (`train.mode: brain`), the Transformer predictor conditions every block on one learned constant.
- A design YAML of either of the last two designs names a `stage0_forecaster` YAML in `encoder_from`; `transformer_world_model.yaml` names `stage0_forecaster.yaml`. Loading it sets `train.encoder_checkpoint` to `best.pt` of the run the stage-0 YAML gets by default, `checkpoints/<dataset>/<stage-0 run name>/best.pt`, and fails if that run has not been trained. Overrides of `model.state_window_trs` or `model.state_dim` select the stage-0 run trained with the same overrides.
- `configs/algonauts/base.yaml` and `configs/hcp_movies/base.yaml` hold each dataset's bundle, splits, TR, haemodynamic delay and model, training, validation and readout settings. To train on HCP, write a design YAML with `base:` pointing at `configs/hcp_movies/base.yaml` (`base` is resolved relative to the design YAML).
- `configs/spectator/` holds fully resolved configs (no `base`); they load the same way as a saved run's `config.yaml`.

### Loading, overrides and paths

`utils/config.py::load_config(path, overrides)` builds a config in this order; later steps win:

1. `DEFAULTS`;
2. the base YAML;
3. the design YAML, then the design's fixed settings;
4. `--set section.key=value` assignments, values parsed as YAML (numbers, lists, `null`);
5. the command's own flags: `--epochs`, `--lr`, `--precision`, `--tag` (`train.run_tag`), `--output-dir` (`train.output_dir`) and `--batch-size` (`data.batch_size`) in stage 1; `--epochs`, `--lr`, `--batch-size` (the `readout.*` fields) and `--precision` in stage 2. The launchers set these flags from environment variables.

```bash
./neuroworld/scripts/launch_stage1.sh neuroworld/configs/algonauts/joint_world_model.yaml \
  --set train.rollout_steps=4 data.sequence_states=7 model.stimulus_features=tribev2_context
```

- Give all assignments after one `--set`; a second `--set` flag replaces the first list.
- Unknown sections or keys are errors. Setting a design's fixed key in a base, a design YAML or `--set` is an error; choose another design YAML instead.
- `validate_config` checks types, ranges and option combinations, and its error messages state each constraint. `RETIRED` lists keys that saved configs still carry; they are accepted only at their defaults. `OLD_DESIGNS` maps the design letters saved in older run configs (`A`, `B`, `C0`, `L0`, `C`, `Cft`) to the names above.
- String values may contain `${NEUROWORLD_DATA}`, `${NEUROWORLD_OUTPUT}` and `${NEUROWORLD_EXTERNAL}`, in YAML files and in `--set` values. `paths.py::expand` replaces them with the roots: the environment variable, or `data/`, `outputs/` and `external/` in the repository root when it is unset. A saved `config.yaml` holds the expanded paths.

Settings by group (defaults in `DEFAULTS`):

| Group | Main keys | Code |
|---|---|---|
| Data | `data.bundle_path`, `train_runs`, `val_runs`, `dev_runs`, `test_runs`, `heldout_subjects`, `split_frames`, `sequence_states`, `window_stride_trs`, `batch_size` (global), `preload`, `num_workers`, `context_features`, `history_fill` | `data/splits.py`, `data/schaefer_v2_dataset.py`, `train/stage_utils.py` |
| Encoder | `model.state_window_trs` (K), `state_dim` (D), `encoder_kind` (`linear`, `transformer`), `encoder_hidden_dim`; `fast_window_trs` with `state_tokens: 2` and `slow_per_dimension` for a fast and a slow latent | `models/encoder.py` |
| Predictor | `model.predictor_kind` (`transformer`, `mlp`, `corrected`), `context_states` (C), `predictor_depth`, `attention_heads`, `predictor_mlp_dim`, `conditioning` (`adaln`, `input`, `output`), `history_blocks`, `chunk_frames`, `state_step`, `mlp_depth` (0: one matrix) | `models/predictor.py`, `models/world_model.py` |
| Stimulus | `model.stimulus_features` (`projections`, `tribev2_context`), `stimulus_context` (`none`, `concat`, `linear`, `concat-audio`), `stimulus_context_trs` (N), `stimulus_scaling` (`layernorm`, `standardize`); `data.hrf_delay_seconds`; `train.modality_dropout` | `models/stimulus.py`, `models/stimulus_context.py` |
| Objective | `train.rollout_steps` (R), `rollout_weight`, `prediction_weight`, `rollout_step_weights`, `rollout_schedule`, `rollout_detach_every`, `latent_error`, `sigreg_weight`, `target` (`encoder`, `initial`, `ema`), `target_momentum`, `encoder_lr_factor`, `parcel_weight`, `teacher_run` | `train/losses.py` |
| Optimization | `train.epochs`, `lr`, `warmup_fraction`, `schedule`, `weight_decay`, `grad_clip`, `precision`, `accumulation_steps`, `patience`, `device` | `train/latent.py`, `train/runtime.py` |
| Validation | `eval.prefix_trs`, `horizon_trs`, `window_stride_trs`, `probe_windows`, `probe_ridge` | `eval/rollout.py`, `eval/probe.py` |
| Readout | `readout.hidden_dim`, `epochs`, `lr`, `batch_size`, `train_horizon_trs`, `train_on`, `cache_rollouts`, `select_by`, `patience` | `train/readout.py` |

### Stage 1: latent dynamics (or the fMRI forecaster)

`train/latent.py::fit` builds the training and validation windows, the model and an AdamW optimizer with linear warmup over `train.warmup_fraction` of all optimizer steps, then cosine decay (`train.schedule: cosine`) or a constant rate. Each epoch:

1. **Train.** `LatentObjective` wraps the loss so the whole objective is one distributed forward call. `latent_loss` encodes the window into S states, predicts states 1..S-1 from the states before them (at most C each) and adds, when `train.rollout_steps` is R > 0, the error of an R-step rollout from the first C states with predictions fed back. SIGReg is applied at each state position across the batch while the encoder trains. With `train.target: ema` or `initial`, a separate target encoder gives the states predictions are scored against. `train.accumulation_steps` averages the gradients of that many batches per optimizer step, leaving the global batch unchanged.
2. **Validate** on process zero. For a latent model, a closed-form ridge probe (`eval/probe.py`) maps rollout latents of `eval.probe_windows` fixed training windows to fMRI; validation windows are rolled out from `eval.prefix_trs` observed states over `eval.horizon_trs` steps and decoded by that probe (the forecaster decodes with its own readout). The selection score is the validation parcel Pearson (`val/selection_score`).
3. **Save** `latest.pt` every epoch and `best.pt` when the selection score improves. With `train.patience: n`, training stops after n epochs without a better score; epochs keep their place in the learning-rate schedule.

### Stage 2: the fMRI readout

`train/readout.py::fit_readout` loads `best.pt`, freezes the world model and checks at the end that none of its weights changed. The readout (`models/readout.py`) is a shared `Linear(D, readout.hidden_dim)` + GELU trunk and one `Linear(·, 1000)` head per training subject (`readout.hidden_dim: 0` leaves one linear map per subject). It trains on fMRI error from rollout latents (`readout.train_on: rollout`) or encoded latents (`encoded`); validation always decodes rollouts and reports fMRI MSE and parcel Pearson, and `readout.select_by` picks `readout_best.pt`. `readout.cache_rollouts: true` computes the rollouts once and keeps them in CPU memory. Stage 2 has no resume; each invocation fits a new readout. The `fmri` designs have no stage 2.

A run is scored on the benchmark with `--model run:<run dir>` (the fMRI forecaster, or a world model through its stage-2 readout) or `--model latent:run:<run dir>` (a frozen linear probe of its predicted states, plus the readout's row when `readout/readout_best.pt` exists); see `benchmark/adapters.py::load_run` and [BrainStateBench](benchmark.md#commands).

### Launching

```bash
./neuroworld/scripts/launch_stage1.sh neuroworld/configs/algonauts/stage0_forecaster.yaml
./neuroworld/scripts/launch_stage1.sh neuroworld/configs/algonauts/transformer_world_model.yaml
RUN_DIR=checkpoints/algonauts/<run name printed by stage 1> ./neuroworld/scripts/launch_stage2.sh
```

- Both scripts change to the repository root, run `${PYTHON:-python}`, and start `python -m accelerate.commands.launch` with `configs/accelerate.yaml`: one process per GPU visible to PyTorch (respecting `CUDA_VISIBLE_DEVICES`), or one CPU process when none is visible. `OMP_NUM_THREADS` (default 4) sets the CPU threads per process.
- Mixed precision follows `train.precision`. The bases set `bf16`, which needs CUDA; on CPU, set `PRECISION=fp32`.
- Stage 1 takes the config as its first argument; further arguments (`--set ...`, `--resume ...`) are passed on. `EPOCHS`, `BATCH_SIZE`, `LR` and `PRECISION` override the config when set; `RUN_TAG` adds a suffix to the run name and `RUN_DIR` chooses the directory.
- Stage 2 requires `RUN_DIR`, reads `$RUN_DIR/config.yaml` (or `CONFIG`), loads `CHECKPOINT` (default `$RUN_DIR/best.pt`) and writes to `OUTPUT_DIR` (default `$RUN_DIR/readout`). `EPOCHS`, `BATCH_SIZE`, `LR` and `PRECISION` apply to the readout; extra arguments such as `--set readout.hidden_dim=0` are passed on.
- Batch sizes are global: each of P processes loads `batch_size / (P × accumulation_steps)` windows, which must divide evenly, and SIGReg sees each process's own batch. The training loader drops its final partial batch; Accelerate seeds its sampler by epoch. The bases use 1,024 windows, 128 per GPU on 8 GPUs.
- `python -m neuroworld.train.latent <config>` runs the same loop in one process without the launcher.

**Resume.** Restart stage 1 from its last epoch with the saved config:

```bash
./neuroworld/scripts/launch_stage1.sh checkpoints/algonauts/<run>/config.yaml --resume checkpoints/algonauts/<run>/latest.pt
```

Resume restores the model, optimizer, scheduler and every process's random state, and continues in the checkpoint's directory. It requires the same process count, seed, data selection, batch size and training settings (`train/checkpoint.py::check_compatible`). A changed `train.epochs` recomputes the schedule over the new total and is recorded as `schedule_change` in `run.jsonl`.

### Run directories and checkpoints

Stage 1 names a new run `checkpoints/<dataset>/<run name>` relative to the working directory (the launchers run from the repository root; `checkpoints` may be a symbolic link, for example to `$NEUROWORLD_OUTPUT/checkpoints`). The name is built by `utils/config.py::run_name` from the settings runs differ in, for example `latent-stimulus_k3c3d1024l6_seq19_lag3_roll16w1_sig0.09_lr5e-05_bs1024_ep100_seed0`, and printed at start as `{"run_dir": ...}`. A new run refuses a directory that already holds `latest.pt`.

```text
<run>/
  config.yaml          fully resolved config; stage 2 and the benchmark load it
  data.json            selected runs, window counts, TR, effective stimulus delay
  run.jsonl            per invocation: source commit and dirty state, versions, device, schedule, resume origin
  metrics.jsonl        per epoch: training losses and validation metrics
  performance.jsonl    per epoch: timing, throughput, peak GPU memory
  latest.pt, best.pt   checkpoints (format neuroworld-latent-v2)
  wandb-run.json, wandb/
  readout/             stage 2: config.yaml, data.json, run.jsonl, metrics.jsonl,
                       readout_best.pt, readout_latest.pt (format neuroworld-readout-v1), wandb-run.json, wandb/
```

Checkpoints hold only tensors and plain values and load with `weights_only=True`. A world-model checkpoint stores the config, weights, optimizer and scheduler state, epoch, global step, history, best score, whether the encoder is frozen, and each process's random state. A readout checkpoint stores the SHA-256 of the world-model checkpoint it was trained on; `train/readout.py::load_readout` refuses a readout whose world model has changed.

The project's finished runs keep their weights only: their optimizer and scheduler state was removed to save space (`$NEUROWORLD_OUTPUT/archive/cleanup-20261119/stripped.tsv`), so they load and score but cannot resume training. The readouts trained on them record the stripped files' SHA-256 (`refingerprinted.tsv` there).

### W&B logging

Both stages log through Accelerate's W&B tracker from process zero. The `wandb` section sets `project`, `entity`, `name` (default `<run>-latent` or `<run>-readout`) and `mode` (`online`, `offline` or `disabled`); `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME` and `WANDB_MODE` override them. The bases use `online`; Spectator's configs and `scripts/spectator.sh` use `offline`.

```bash
WANDB_MODE=offline ./neuroworld/scripts/launch_stage1.sh neuroworld/configs/algonauts/joint_world_model.yaml
wandb sync checkpoints/algonauts/<run>/wandb/offline-run-*     # upload later
```

Offline runs are written under `<run>/wandb/` and need no login or network. Nested metrics are flattened into scalar curves (for example `val/fmri/model/parcel_pearson`, `train/rollout_loss`) on a `global_step` axis. An online stage-1 resume continues the W&B run whose ID the checkpoint holds. JSON logs and checkpoints are written in every mode.

### Data flow

With K frames per state, S states per window, C states of history, R rollout steps, N stimulus rows per transition and D state values (Algonauts base: K = 3, S = 19, C = 3, R = 16, N = 1, D = 1024):

```text
batch["brain"]    [B, K+S-1, 1000]  --encoder-->  states z [B, S, D]          (one state per frame: the K frames ending there)
batch["stimuli"]  text [B, S+N-2, 2, 3072], audio [.., 2, 1024], video [.., 2, 1408]
                    --frozen TRIBE v2 projections-->  [B, S+N-2, 1152]
              or  context [B, S+N-2, 2048]            (model.stimulus_features: tribev2_context)
                    --stimulus context (none | concat | linear)-->  conditions a [B, S-1, W]   (W = 1152 or 2048; N·W with concat)
predictor(z[:, :-1], a)                  --> predicted next states [B, S-1, D]   compared with z[:, 1:]
rollout from z[:, :C], a[:, :C+R-1]      --> [B, R, D]                           compared with z[:, C:C+R]
validation, stage 2: S = eval.prefix_trs + H; rollout from the first K+prefix-1 frames --> [B, H, D] --readout or probe--> [B, H, 1000]
```

The row conditioning the step from state r to r+1 comes from stimulus frame `a+K+r-lag` for a window starting at frame a, with `lag = floor(data.hrf_delay_seconds / data.tr_seconds + 0.5)` (3 frames on Algonauts). Windows never cross invalid frames (except the oldest frames of a window with `data.history_fill`); `data/schaefer_v2_dataset.py::SchaeferWindowDataset` also returns `frame_indices`, `state_frame_indices`, `physical_time_s`, `run_id` and `subject_id`. `WorldModel.rollout` receives only the observed prefix and the stimulus rows.

## How Spectator uses this runtime

Spectator's two gradient steps run through `scripts/launch_stage1.sh` and `train/latent.py`; its closed-form steps and scoring are in `spectator/`. [Spectator](spectator.md) describes the steps and their settings.

- **Stage 0** is `configs/spectator/algonauts-stage0.yaml`, a `stage0_forecaster` run.
- **The JEPA phase** is `configs/spectator/algonauts-jepa.yaml`, a `transformer_world_model_finetuned` run with an EMA target (`train.target: ema`) and a one-matrix predictor (`model.predictor_kind: mlp`, `mlp_depth: 0`). Its encoder and start-matrix files are set by `python -m neuroworld.spectator config`, which writes a copy of the config with `train.encoder_checkpoint` and `train.predictor_checkpoint` replaced; `--set` cannot change the encoder file, since the design fixes it.
- **The final model** is a run directory written by `python -m neuroworld.spectator assemble`: `config.yaml` and a `best.pt` in the stage-1 checkpoint format, so it loads with `train/readout.py::load_world_model` like any run. It has no stage 2; `spectator score` reads it through its own probe.
- `scripts/spectator.sh` runs each gradient step with `RUN_DIR=<out>/<step>` and `--set wandb.mode=offline`, resumes from `latest.pt` when one exists, and marks a finished step with `stage1.done`.

## Developing

```bash
python -m pip install -r neuroworld/requirements-dev.txt
WANDB_MODE=offline CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

`pytest.ini` sets `testpaths = tests`. The tests build synthetic inputs and need no dataset, weights or GPU; `tests/test_spectator.py` also compares the code with the frozen Spectator weights when `$NEUROWORLD_OUTPUT/spectator-v1/` exists.

To inspect one training batch and the model's shapes on CPU (needs the prepared bundle):

```bash
python -B - <<'PY'
from neuroworld.utils.config import load_config
from neuroworld.train.stage_utils import build_dataset, build_loader
from neuroworld.models import build_model
cfg = load_config("neuroworld/configs/algonauts/joint_world_model.yaml", ["data.num_workers=0", "data.preload=false"])
data = build_dataset(cfg, "train")
batch = next(iter(build_loader(data, cfg, batch_size=2)))
model = build_model(cfg, data.subject_ids)
states, rows = model.encode(batch["brain"]), model.encode_stimulus(batch["stimuli"])
print(tuple(batch["brain"].shape), tuple(states.shape), tuple(rows.shape), tuple(model.predict(states[:, :-1], rows).shape))
PY
```

It prints `(2, 21, 1000) (2, 19, 1024) (2, 18, 1152) (2, 18, 1024)`.

| Change | Where |
|---|---|
| A different setting | `--set section.key=value` on a design YAML; fields in `DEFAULTS`, constraints in `validate_config` |
| A new design | an entry in `DESIGNS` (and in `PHASE0_DESIGNS` if its encoder comes from a stage-0 run) and a design YAML next to the dataset's base |
| A new config field | `DEFAULTS` and `validate_config` in `utils/config.py`, the module that reads it, `run_name` if runs should differ by it, and `train/checkpoint.py::check_compatible` if it changes loading or resume |
| The encoder | `models/encoder.py` (`frame_encoder`, `HistoryEncoder`), built in `models/world_model.py` |
| The predictor | `models/predictor.py`, `models/lewm_blocks.py`; keep the interfaces of `WorldModel.predict` and `WorldModel.rollout_states` |
| A loss term | `train/losses.py`; `train_one_epoch` averages and logs every metric the loss returns |
| The readout | `models/readout.py`, `train/readout.py` |
| Validation metrics | `eval/rollout.py::evaluate`, `eval/fmri.py` |
| Windows, splits, stimulus alignment | `data/schaefer_v2_dataset.py`, `data/splits.py`, `train/stage_utils.py` |
| Logging | `train/tracking.py` |
| Benchmark scoring of a run | `benchmark/adapters.py::load_run` |
| Prepared arrays | `nbs_data/` ([data preparation](data-preparation.md)) |

Keep stimulus alignment in the loader, keep the observed prefix as the only fMRI input to a rollout, and keep source notes next to adapted code.

## Sources and adaptations

- [LeWM](https://github.com/lucas-maes/le-wm), commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`: SIGReg (`models/collapse.py`) and the conditional Transformer blocks (`models/lewm_blocks.py`) from `module.py`, loss behaviour from `train.py`, rollout organization from `jepa.py`. The local blocks replace einops with PyTorch reshapes, use `dim_head = D / heads` and take an explicit attention mask. Training batches independent C-state windows. The relative-position bias (in place of LeWM's absolute position embeddings), the zero-initialized additive output and the learned constant for brain-only training are local adaptations; `model.predictor_mlp_dim` sets the feed-forward width. License: [MIT](../neuroworld/licenses/LeWM.txt).
- [TRIBE v2](https://github.com/facebookresearch/tribev2), commit `38b3db073a7fa5bfbfc7fdd894b1a3536e4553e3`, released checkpoint snapshot `f894e783020944dcd96e5568550afe2aa9743f9f`: the per-modality flattening and projection and the missing-modality handling of `FmriEncoderModel.aggregate_features` (`models/stimulus.py`); the network as released (`models/tribev2.py`: projections, time embedding, 8-block Transformer built with `neuraltrain`'s `TransformerEncoder`, low-rank head and the average-subject brain head). With `model.stimulus_features: projections` the runtime loads only the projection weights and applies them to features already pooled to the TR; with `tribev2_context` it reads precomputed rows. The contextual rows (`nbs_data/stimuli/tribev2_context.py`) run the projections, Transformer and low-rank head offline with causal attention in place of the trained bidirectional attention. The brain head is used only for the zero-shot reference (`eval/tribev2_reference.py`). License: [CC BY-NC 4.0](../neuroworld/licenses/TRIBEv2.txt).
- The NeuroWorld preprint (Dong et al., 2026): the two-stage protocol, the prefix/horizon evaluation and the shared-trunk readout with one head per subject. No code or weights from it are loaded.

The encoders, the predictors' history handling, the matrix predictor, the readout, the dataset code and the training orchestration are local code. Apart from TRIBE v2's, no external pretrained weights are loaded.
