# State of Mind

Code for **State of Mind: Learning and Evaluating World Models of Naturalistic Brain Dynamics**.

A brain world model keeps an internal state built from a person's measured fMRI and simulates how that state evolves while they watch a movie. This repository contains:

- **BrainStateBench**, a benchmark that scores such models on three tasks over the next minute: *Forecast* (predict the fMRI from the past and the movie), *State* (predict how two viewings of the same scene differ) and *Updating* (use a few new measurements to stay on track). See [docs/benchmark.md](docs/benchmark.md).
- **Spectator**, the paper's model: a linear encoder that summarizes the last minutes of fMRI into a 2,048-number state, trained in the JEPA sense against its own slowly updated copy and then frozen, and one matrix, solved in closed form, that rolls the state forward under the movie. See [docs/spectator.md](docs/spectator.md).
- The code of the paper's analyses and baselines, mapped table by table in [docs/reproduce.md](docs/reproduce.md).

## Results

All Algonauts 2025 test data (*Friends* S6 and the four Movie10 films), skill in % (higher is better), with 180 s of fMRI history; r is the parcel-averaged Pearson correlation over the 40 steps of the minute. The paper also reports 7.5 and 120 s, HCP and the ablations.

| Model | Forecast | State | Updating | r |
|---|---:|---:|---:|---:|
| TRIBE v2 (stimulus only) | 4.2 | 0.0 | 4.1 | 0.173 |
| MIRAGE (stimulus only) | 8.8 | 0.0 | 8.8 | 0.320 |
| Linear forecaster | 11.6 | 2.6 | 12.7 | 0.294 |
| IPSID | 8.7 | 1.2 | 12.5 | 0.243 |
| NeuroWorld | 8.3 | 0.4 | 13.4 | 0.239 |
| BrainVista | 6.6 | 0.2 | 10.3 | 0.220 |
| **Spectator** | **16.0** | **5.5** | **20.3** | **0.349** |

## Setup

Python 3.11. From the repository root:

```bash
python -m pip install -r neuroworld/requirements-dev.txt
WANDB_MODE=offline CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The tests build synthetic inputs; they need no dataset, weights or GPU (the Spectator reproduction test is skipped when the frozen weights are absent). The paper's analysis scripts also use scikit-learn, nibabel and nilearn.

**Data and outputs** are not included. Code finds them through three roots ([`neuroworld/paths.py`](neuroworld/paths.py)), each an environment variable or, if unset, a folder or symbolic link in the repository root:

| Variable | Default | Holds |
|---|---|---|
| `NEUROWORLD_DATA` | `data/` | prepared datasets ([docs/data-preparation.md](docs/data-preparation.md)), atlases, the TRIBE v2 checkpoint |
| `NEUROWORLD_OUTPUT` | `outputs/` | runs, benchmark result stores, paper analyses, the frozen Spectator weights (`spectator-v1/`) |
| `NEUROWORLD_EXTERNAL` | `external/` | other baselines' outputs (BrainVista, MIRAGE, the original NeuroWorld run), read only |

Configs refer to them as `${NEUROWORLD_DATA}` and so on. Training runs are written to `checkpoints/<dataset>/` in the working directory; a `checkpoints` link to `$NEUROWORLD_OUTPUT/checkpoints` keeps them with the other outputs. Algonauts 2025 comes from the [Algonauts challenge](https://algonautsproject.com/) (CNeuroMod), HCP from the [Human Connectome Project](https://www.humanconnectome.org/) (7T movie runs).

## Usage

```bash
# Train Spectator end to end (stage 0, lag weights, JEPA phase, final matrix) and score it on every task
OUT=outputs/my-spectator neuroworld/scripts/spectator.sh algonauts

# Score a model on BrainStateBench (any class with predict(QueryBatch); see examples/persistence.py)
python -m neuroworld.benchmark suite --dataset algonauts --model py:examples/persistence.py:DecayingPersistence
python -m neuroworld.benchmark profile

# The paper's main table from the stored results
python -m neuroworld.paper.history_table
```

## Repository layout

| Path | Contents |
|---|---|
| [`neuroworld/benchmark/`](neuroworld/benchmark) | BrainStateBench: queries, tasks, scoring, reference models (linear forecaster, IPSID, TRIBE v2), latent probes |
| [`neuroworld/spectator/`](neuroworld/spectator) | Spectator's closed-form steps, run directory and scoring (`python -m neuroworld.spectator`) |
| [`neuroworld/models/`](neuroworld/models), [`neuroworld/train/`](neuroworld/train) | encoders, predictors and their training (Spectator's stage 0 and JEPA phase, the Transformer and NeuroWorld-style comparisons); [training runtime](docs/code.md) |
| [`neuroworld/configs/`](neuroworld/configs) | dataset configs and Spectator's frozen configs |
| [`neuroworld/nbs_data/`](neuroworld/nbs_data), [`neuroworld/data/`](neuroworld/data) | data preparation and loading |
| [`neuroworld/experiments/`](neuroworld/experiments) | the linear forecaster by history length |
| [`neuroworld/paper/`](neuroworld/paper) | the paper's analyses |
| [`neuroworld/eval/`](neuroworld/eval) | evaluation helpers and TRIBE v2's zero-shot predictions |
| [`examples/`](examples), [`tests/`](tests), [`docs/`](docs) | an external-model example, tests, documentation |

## Sources and licenses

Parts of the code are adapted from [LeWM](https://github.com/lucas-maes/le-wm) (SIGReg and conditional Transformer blocks; MIT) and [TRIBE v2](https://github.com/facebookresearch/tribev2) (feature projection and contextual rows; CC BY-NC 4.0); their licenses are in [`neuroworld/licenses/`](neuroworld/licenses), and the adaptations are described in [docs/code.md](docs/code.md#sources-and-adaptations).
