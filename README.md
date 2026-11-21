# State of Mind

Official code for **State of Mind: Learning and Evaluating World Models of Naturalistic Brain Dynamics**.

**[Pretrained Weights & Stored Scores (Hugging Face)](https://huggingface.co/stanjsx/Spectator)** | **[Data Preparation](DATA.md)**

A brain world model infers an internal latent state from a person's measured fMRI, predicts how that state evolves under an incoming naturalistic stimulus (a movie), and updates the state as new observations arrive. This repository provides:

- **Spectator**: a linear two-timescale latent brain world model trained with a joint-embedding predictive objective (JEPA), followed by a closed-form linear transition matrix.
- **BrainStateBench**: a benchmark probing latent brain states via frozen linear readouts across three tasks over a 60-second horizon (*Forecast*, *State*, and *Updating*).
- **Baselines & Paper Analyses**: reference baselines (TRIBE v2, linear forecaster, IPSID, and Transformer/world-model training), plus scripts reproducing the paper's tables and state–stimulus decomposition analyses.

---

## Main Results

Algonauts 2025 evaluation data (*Friends* S6 and the four Movie10 films), 180 s of fMRI history. *Forecast*, *State*, and *Updating* are variance explained in % (higher is better); $r$ is parcel-averaged Pearson correlation over the 40-step (60 s) horizon.

| Model | Forecast (%) | State (%) | Updating (%) | $r$ |
|---|---:|---:|---:|---:|
| TRIBE v2 (stimulus only) | 4.2 | 0.0 | 4.1 | 0.173 |
| MIRAGE (stimulus only)* | 8.8 | 0.0 | 8.8 | 0.320 |
| Linear forecaster | 11.6 | 2.6 | 12.7 | 0.294 |
| IPSID | 8.7 | 1.2 | 12.5 | 0.243 |
| NeuroWorld | 8.3 | 0.4 | 13.4 | 0.239 |
| BrainVista* | 6.6 | 0.2 | 10.3 | 0.220 |
| **Spectator** | **16.0** | **5.5** | **20.3** | **0.349** |

*\*MIRAGE and BrainVista were scored externally by a collaborator and are not included in the released score store.*

---

## Installation

Python 3.11 is recommended. Run commands from the repository root:

```bash
python -m pip install -r neuroworld/requirements-dev.txt
WANDB_MODE=offline CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The test suite uses synthetic inputs and requires no datasets, weights, or GPU. Optional extra dependencies:
- **Paper analyses & IPSID baseline:** `python -m pip install -r neuroworld/requirements-analysis.txt`
- **Data preparation & TRIBE v2 contextual features:** `python -m pip install -r neuroworld/requirements-data.txt`

### Data and output directories

Paths are resolved through two environment variables ([`neuroworld/paths.py`](neuroworld/paths.py)), which default to folders (or symlinks) in the repository root if unset:

| Variable | Default | Contents |
|---|---|---|
| `NEUROWORLD_DATA` | `data/` | Prepared fMRI bundles, Schaefer-1000 atlas, TRIBE v2 checkpoint ([DATA.md](DATA.md)) |
| `NEUROWORLD_OUTPUT` | `outputs/` | Downloaded Spectator weights, stored benchmark scores, training runs, and analysis outputs |

---

## Pretrained Weights & Quick Start

Download the released Spectator checkpoints, frozen linear probes, and stored BrainStateBench scores from Hugging Face into `outputs/` (or `$NEUROWORLD_OUTPUT`):

```bash
python -m pip install huggingface_hub
hf download stanjsx/Spectator --local-dir outputs
```

This creates:
- `outputs/spectator-v1/`: Spectator on Algonauts 2025 (180 s — the paper's main model — plus 7.5 s and 120 s) and on HCP 7T movies, each with its frozen probes;
- `outputs/benchmark-results/`: stored Algonauts BrainStateBench scores for the main table and history-length sweep.

With the weights downloaded, `tests/test_spectator.py` automatically verifies that the current code reproduces Spectator's stored reference rollouts on both Algonauts and HCP.

### Running a 60-second rollout in Python

```python
import json
import numpy as np
import torch
from neuroworld.train.readout import load_world_model
from neuroworld.utils.config import load_config

folder = "outputs/spectator-v1/algonauts/model"
model, _ = load_world_model(load_config(f"{folder}/config.yaml"), f"{folder}/best.pt", "cpu")

# brain: [B, 120, 1000] (last 180 s of fMRI at TR = 1.49 s)
# stimuli["context"]: [B, 44, 2048] (TRIBE v2 causal contextual movie features)
ref = torch.load(f"{folder}/reference-rollout.pt", weights_only=True)
with torch.no_grad():
    states = model.rollout(ref["brain"], ref["stimuli"], horizon_trs=40)  # [B, 40, 2048]

probe = np.load(f"{folder}/probe.npz")
i = json.loads(str(probe["meta"]))["groups"].index("Algonauts2025/01")
parcels = states.numpy() @ probe[f"W {i}"] + probe[f"b {i}"]              # [B, 40, 1000]
```

---

## Data

Neither dataset is redistributed here. [`neuroworld/nbs_data/download.sh`](neuroworld/nbs_data/download.sh) downloads both, and [DATA.md](DATA.md) covers sources, preprocessing, the preparation commands, and train/test splits:

- **Algonauts 2025** ([CNeuroMod](https://github.com/courtois-neuromod/algonauts_2025.competitors)): 4 participants watching *Friends* S1–6 and four Movie10 films, parcellated into Schaefer-1000 parcels (TR 1.49 s). *Friends* S1–5 are used for training; S6 and all four Movie10 films are held out for evaluation.
- **HCP 7T Movies** ([Human Connectome Project](https://www.humanconnectome.org/)): 184 participants parcellated into the same 1,000 Schaefer parcels (TR 1 s), split into 148 training and 36 unseen participants.

Movie stimuli enter as causal contextual features from [TRIBE v2](https://huggingface.co/facebook/tribev2). Note that TRIBE v2 was pretrained on CNeuroMod (which includes Algonauts recordings), whereas HCP lies outside TRIBE v2's training data. Both fMRI datasets are preprocessed over full runs, so benchmark scores reflect offline forecasting rather than real-time streaming.

---

## Spectator: Architecture & Training

Spectator models naturalistic brain dynamics in a 2,048-dimensional two-timescale latent state:

1. **Two-timescale encoder.** A linear map projects the newest 5 fMRI frames (7.5 s) into a 1,024-D fast state $f_t$. Per-channel learned lag weights summarize $f_t$ over the full 180 s history into a 1,024-D slow state $u_t$, giving $s_t = [f_t; u_t] \in \mathbb{R}^{2048}$.
2. **Linear residual transition.** $s_{t+1} = s_t + W[s_t; \bar{a}_{t+1}; 1]$, where $\bar{a}_{t+1}$ concatenates the newest 5 frames of causal TRIBE v2 stimulus features. Forty steps forecast one minute.
3. **Frozen linear probe.** One linear map per individual (or one pooled probe on HCP) maps latent states to 1,000 cortical parcels strictly for evaluation.

Training on Algonauts (*Friends* S1–5) proceeds in four stages:
1. **Stage 0 (encoder pretraining):** train the fast map $g_\phi$ by forecasting fMRI without movie input.
2. **Slow lag weights (closed form):** ridge fit from past fast states to the mean of the next 24 s of fast states.
3. **JEPA phase:** jointly train the full encoder and predictor over 40-step latent rollouts against an EMA target encoder ($\tau = 0.998$) with 30% stimulus dropout.
4. **Final transition matrix (closed form):** freeze the encoder and solve for $W$ by ridge regression with differential penalties on state ($10\times$) and stimulus ($100\times$; $10{,}000\times$ on HCP) columns.

```bash
# Train Spectator end-to-end on Algonauts and score on BrainStateBench
OUT=outputs/my-spectator neuroworld/scripts/spectator.sh algonauts

# Other history lengths (seconds; default 180, the main table also reports 7.5 and 120)
HISTORY=120 OUT=outputs/my-spectator-120s neuroworld/scripts/spectator.sh algonauts

# Transfer to HCP (reuses the Algonauts fast encoder and refits slow lags + transition matrix)
OUT=outputs/my-spectator neuroworld/scripts/spectator.sh hcp

# Score the downloaded pretrained model on every BrainStateBench task
python -m neuroworld.spectator score outputs/spectator-v1/algonauts/model \
    --seconds 180 --dataset algonauts --store outputs/my-results --name spectator-180s
```

Frozen configs live in [`neuroworld/configs/spectator/`](neuroworld/configs/spectator), and individual step commands are documented in [`neuroworld/spectator/__init__.py`](neuroworld/spectator/__init__.py).

---

## BrainStateBench

Each task reports variance explained over a 60-second horizon, $\text{skill} = 100 \times \left(1 - \sum(y - \hat{y})^2 / \sum(y - \bar{y})^2\right)$, where the task's reference prediction $\bar{y}$ scores 0%:

| Task | Question | Reference $\bar{y}$ |
|---|---|---|
| **Forecast** | Can the model predict the next minute from fMRI history and the movie? | Parcel mean over the recording |
| **State** | Does fMRI history predict trial-to-trial differences between two viewings of the same movie? | Zero difference |
| **Updating** | Do 4 newly revealed fMRI frames spaced across the minute improve predictions? | Parcel mean over the recording |

### Evaluating your own model

Implement a class with a `name` attribute and a `predict(batch) -> [B, horizon, 1000]` method (see [`examples/persistence.py`](examples/persistence.py) for a complete working example):

```python
class MyModel:
    def __init__(self, dataset, device):
        self.name = "my model"

    def predict(self, batch):  # QueryBatch -> float32 array [B, horizon, 1000]
        ...
```

Score it into the downloaded result store and generate a comparison card with paired 95% bootstrap confidence intervals:

```bash
python -m neuroworld.benchmark suite --dataset algonauts --model py:my_model.py:MyModel \
    --output outputs/benchmark-results
python -m neuroworld.benchmark profile --output outputs/benchmark-results \
    --reference "ridge-jepa-180s-std-partial (latent)"
```

Built-in models: `direct-linear`, `ipsid`, `tribev2`, `persistence`, `run:<dir>`, and `latent:run:<dir>`. For the history-length sweeps of the linear forecaster and IPSID, see [`neuroworld/experiments/history_length.py`](neuroworld/experiments/history_length.py). Pass `--help` to any command for full options.

---

## Reproducing Paper Tables & Analyses

All analysis commands run from the repository root and read from `$NEUROWORLD_OUTPUT` (and `$NEUROWORLD_DATA` where raw fMRI/stimuli are required). Full module list in [`neuroworld/paper/__init__.py`](neuroworld/paper/__init__.py):

| Paper Result | Command | Requires Data? |
|---|---|---|
| **Table 2 & App. B.3** (Main results & paired CIs from stored scores) | `python -m neuroworld.paper.history_table` | No (uses HF store) |
| **Table 3** (Input perturbations: w/o stimulus, w/o fMRI, other viewer) | `python -m neuroworld.paper.inputs run --condition all` | Yes (Algonauts) |
| **Fig. 3, Fig. 4, App. D.1–D.2** (Exact decomposition, swaps, cortical gains, GradientSHAP) | `python -m neuroworld.paper.collect spectator --every 1` followed by `python -m neuroworld.paper.decomposition {swap-table,state-swap,parcel-benefit,gradshap}` | Yes (Algonauts) |
| **App. B.2 & App. D.3** (Horizon decay, updating budgets, eigenmodes, impulse response, ISC) | `python -m neuroworld.paper.analysis {benchmark,collected,isc,maps}` | `benchmark`: No; others: Yes |
| **App. B.5** (Standard metrics: Parcel $r$, Global $r$, Cls@Top10) | `python -m neuroworld.paper.standard_metrics save ...` then `compute` | Yes (Algonauts) |
| **App. B.7** (Unseen HCP viewer calibration, 0–15 min) | `python -m neuroworld.paper.new_viewer {select,select-run1,fit,score,report} --out outputs/paper-analysis/new-viewer` | Yes (HCP) |
| **App. C.2** (5-minute open-loop rollouts) | `python -m neuroworld.paper.long_horizon run --part {latent,ridge} --out outputs/paper-analysis/long-horizon` then `report` | Yes (Algonauts) |
| **App. E.2** (Linear forecaster trained on Spectator trajectories) | `python -m neuroworld.paper.generated fit --seconds 180 --teacher spectator --alpha 1 --out outputs/paper-analysis/generated` | Yes (Algonauts) |

---

## Repository Layout

| Path | Contents |
|---|---|
| [`neuroworld/spectator/`](neuroworld/spectator) | Spectator closed-form fitting (slow lags, transition matrix), assembly, and scoring |
| [`neuroworld/benchmark/`](neuroworld/benchmark) | BrainStateBench tasks, scoring, reference baselines, latent probes, and benchmark card |
| [`neuroworld/models/`](neuroworld/models), [`neuroworld/train/`](neuroworld/train) | Encoders, predictors, world model, and gradient training loops |
| [`neuroworld/configs/`](neuroworld/configs) | Dataset base configs and frozen Spectator configs |
| [`neuroworld/scripts/`](neuroworld/scripts) | End-to-end training and launch scripts (`spectator.sh`, `launch_stage1.sh`, `launch_stage2.sh`) |
| [`neuroworld/data/`](neuroworld/data), [`neuroworld/nbs_data/`](neuroworld/nbs_data) | Dataset loaders and Schaefer-1000 / TRIBE v2 data preparation pipeline |
| [`neuroworld/experiments/`](neuroworld/experiments) | History-length scaling experiments for baselines |
| [`neuroworld/paper/`](neuroworld/paper) | Scripts reproducing paper tables and post-hoc analyses |
| [`examples/`](examples), [`tests/`](tests) | Minimal custom-model example and synthetic-data unit tests |

*(The Python package is named `neuroworld` for historical reasons; it is distinct from the NeuroWorld baseline reported in the results table.)*

---

## License & Attribution

Parts of the codebase are adapted from [LeWM](https://github.com/lucas-maes/le-wm) (SIGReg and conditional Transformer blocks; MIT) and [TRIBE v2](https://github.com/facebookresearch/tribev2) (stimulus projections and contextual encoder; CC BY-NC 4.0). Each adapted file documents its provenance and modifications, and full third-party license texts are in [`neuroworld/licenses/`](neuroworld/licenses).

The code and the pretrained Spectator weights are released under **CC BY-NC 4.0** (matching TRIBE v2; [LICENSE](LICENSE)).
