# State of Mind

Code for **[State of Mind: Learning and Evaluating World Models of Naturalistic Brain Dynamics](https://arxiv.org/abs/2611.28508)**.

Yuxiang Wei<sup>1,2,&#42;</sup>, Zundong Wu<sup>1,2,&#42;</sup>, Weiyu Sun<sup>1,&#42;</sup>, Qizhen Lan<sup>3</sup>, Yanteng Zhang<sup>2</sup>, Vince D. Calhoun<sup>2</sup>

<sup>1</sup>Georgia Institute of Technology · <sup>2</sup>Center for Translational Research in Neuroimaging and Data Science · <sup>3</sup>University of Texas Health Science Center at Houston  
<sub><sup>&#42;</sup>Equal contribution</sub>

**[Paper (arXiv)](https://arxiv.org/abs/2611.28508)** | **[PDF](https://arxiv.org/pdf/2611.28508)** | **[Project page](https://yuxiangwei0808.github.io/State-of-Mind/)** | **[Weights and stored scores](https://huggingface.co/stanjsx/Spectator)** | **[Data](DATA.md)**

This repository contains:

- **BrainStateBench**, a benchmark that scores brain world models on three tasks over the next minute;
- **Spectator**, the paper's model: a linear encoder that turns the last minutes of fMRI into a 2,048-number state, and one matrix that rolls the state forward under the movie;
- the data preparation and loading pipeline for Algonauts 2025 and HCP 7T movies.

## Results

All Algonauts 2025 test data (*Friends* S6 and the four Movie10 films), skill in % (higher is better), with 180 s of fMRI history; r is the parcel-averaged Pearson correlation over the 40 steps of the minute.

| Model | Forecast | State | Updating | r |
|---|---:|---:|---:|---:|
| TRIBE v2 (stimulus only) | 4.2 | 0.0 | 4.1 | 0.173 |
| MIRAGE (stimulus only) | 8.8 | 0.0 | 8.8 | 0.320 |
| Linear forecaster | 11.6 | 2.6 | 12.7 | 0.294 |
| IPSID | 8.7 | 1.2 | 12.5 | 0.243 |
| NeuroWorld | 8.3 | 0.4 | 13.4 | 0.239 |
| BrainVista | 6.6 | 0.2 | 10.3 | 0.220 |
| **Spectator** | **16.0** | **5.5** | **20.3** | **0.349** |

## Installation

Python 3.11. From the repository root:

```bash
python -m pip install -r state_of_mind/requirements-dev.txt
WANDB_MODE=offline CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The tests use synthetic inputs and need no data, weights or GPU. Data preparation also needs [`requirements-data.txt`](state_of_mind/requirements-data.txt).

The code finds data and outputs through two roots ([`state_of_mind/paths.py`](state_of_mind/paths.py)). Each is an environment variable or, if unset, a folder or link in the repository root:

| Variable | Default | Holds |
|---|---|---|
| `STATE_OF_MIND_DATA` | `data/` | prepared datasets, the atlas, the TRIBE v2 checkpoint ([DATA.md](DATA.md)) |
| `STATE_OF_MIND_OUTPUT` | `outputs/` | Spectator's weights, training runs, benchmark results |

Legacy `NEUROWORLD_DATA` / `NEUROWORLD_OUTPUT` variables and `${NEUROWORLD_*}` placeholders in released configs are still accepted as fallbacks.

## Pretrained model

```bash
python -m pip install huggingface_hub
hf download stanjsx/Spectator --local-dir outputs
```

This gives:
- `outputs/spectator-v1/`: Spectator on Algonauts 2025 with 180 s of fMRI history (the paper's model), with 7.5 and 120 s, and on HCP, each with its probes;
- `outputs/benchmark-results/`: the stored scores of the main table's rows.

With the weights in place, `tests/test_spectator.py` also checks the code against the model's stored rollout.

```python
import json
import numpy as np
import torch
from state_of_mind.train.readout import load_world_model
from state_of_mind.utils.config import load_config

folder = "outputs/spectator-v1/algonauts/model"
model, _ = load_world_model(load_config(f"{folder}/config.yaml"), f"{folder}/best.pt", "cpu")

# brain [B, 120, 1000]: the last 180 s of fMRI (TR 1.49 s); stimuli["context"] [B, 44, 2048]: the movie's TRIBE v2 rows.
ref = torch.load(f"{folder}/reference-rollout.pt", weights_only=True)   # a fixed synthetic input of these shapes
with torch.no_grad():
    states = model.rollout(ref["brain"], ref["stimuli"], horizon_trs=40)  # [B, 40, 2048]: the next minute

probe = np.load(f"{folder}/probe.npz")                                    # one linear map per participant
i = json.loads(str(probe["meta"]))["groups"].index("Algonauts2025/01")
parcels = states.numpy() @ probe[f"W {i}"] + probe[f"b {i}"]              # [B, 40, 1000]
```

## Data

Two fMRI datasets of people watching movies, neither included here:
- **Algonauts 2025** ([CNeuroMod](https://github.com/courtois-neuromod/algonauts_2025.competitors)): 4 participants watching *Friends* seasons 1–6 and four films, as released in Schaefer-1000 parcels (TR 1.49 s). *Friends* S1–5 are used for training.
- **HCP 7T movies** ([Human Connectome Project](https://www.humanconnectome.org/)): 184 participants, which we average into the same parcels (TR 1 s).

The movies enter as TRIBE v2's stimulus features. [`state_of_mind/nbs_data/download.sh`](state_of_mind/nbs_data/download.sh) downloads both datasets, and [DATA.md](DATA.md) covers preparing them as for the paper and which recordings each split holds.

## Spectator

1. **State.** A fast latent is one linear map of the newest 5 fMRI frames (7.5 s) to 1,024 values. A slow latent is, for each of those values, a weighted sum of the fast latent over the whole history. The state is both, 2,048 values.
2. **Step.** `s_{t+1} = s_t + W [s_t ; m_t ; 1]`, where `m_t` is the movie's TRIBE v2 contextual rows of the newest 5 frames. Forty steps forecast a minute.
3. **Reading.** A linear probe per participant maps states to parcels. It is fitted for scoring only; the model is never trained on parcels.

Training uses Algonauts' *Friends* S1–5 and has four steps:
- **stage 0** learns the fast map by forecasting fMRI without the movie;
- the slow latent's weights are fitted in closed form;
- in the **JEPA phase**, the whole encoder trains to predict its own future states, given by a slowly updated (EMA) copy of itself;
- with the encoder frozen, the final matrix is solved in closed form, with a strong penalty on the movie part.

```bash
# Train every step and score the result on every task (HISTORY: seconds of fMRI history, default 180; 7.5 and 120 in the main table)
OUT=outputs/my-spectator state_of_mind/scripts/spectator.sh algonauts
HISTORY=120 OUT=outputs/my-spectator-120s state_of_mind/scripts/spectator.sh algonauts
# On HCP: keep the Algonauts encoder (yours, or the downloaded one) and refit the closed-form steps
OUT=outputs/my-spectator state_of_mind/scripts/spectator.sh hcp

# Score the downloaded model
python -m state_of_mind.spectator score outputs/spectator-v1/algonauts/model --seconds 180 --dataset algonauts \
    --store outputs/my-results --name spectator-180s
```

The frozen configs are in [`state_of_mind/configs/spectator/`](state_of_mind/configs/spectator). Each step's command is listed in [`state_of_mind/spectator/__init__.py`](state_of_mind/spectator/__init__.py).

## BrainStateBench

Each task scores forecasts of measured fMRI against the recording, as skill = 100 × (1 − Σ(y − ŷ)² / Σ(y − ȳ)²) over every scored frame and parcel. Here ȳ is the task's reference prediction, which scores 0.

| Task | Question | Reference |
|---|---|---|
| **Forecast** | Can the model predict the next minute from the fMRI history and the movie? | each parcel's mean over the run |
| **State** | Does the history help predict how two viewings of the same scene differ? | no difference |
| **Updating** | Do four new fMRI frames spread over the minute keep the forecast on track? | each parcel's mean over the run |

A model receives the fMRI before the start of the forecast (with a mask of valid frames) and the movie's features, past and future. It returns the next minute of fMRI, shaped `[queries, frames, 1000]`. A model that exposes its states instead is scored through a linear probe fitted on its own rollouts on the training recordings. The benchmark never retrains a model.

To add a model, write a class with a `name` and a `predict` method ([`examples/persistence.py`](examples/persistence.py) is a complete example):

```python
class MyModel:
    def __init__(self, dataset, device):
        self.name = "my model"
    def predict(self, batch):    # QueryBatch -> [queries, horizon, 1000]
        ...
```

```bash
python -m state_of_mind.benchmark suite --dataset algonauts --model py:my_model.py:MyModel --output outputs/benchmark-results
python -m state_of_mind.benchmark profile --output outputs/benchmark-results --reference "my model"
```

`profile` prints the benchmark card: every row of the store, with paired 95% intervals against the reference row. Scoring into the downloaded store puts a model next to the paper's stored rows. Built-in model specs: `py:<file.py>:<Class>`, `run:<dir>`, `latent:run:<dir>`, and `persistence`. Each command takes `--help`.

TRIBE v2 was trained on CNeuroMod, which includes every Algonauts recording, so models that read its features inherit that exposure. The fMRI is preprocessed over whole runs, so the scores are for offline forecasting, not real time.

## Repository layout

| Path | Contents |
|---|---|
| [`state_of_mind/spectator/`](state_of_mind/spectator) | Spectator's closed-form steps, assembly, and scoring |
| [`state_of_mind/models/`](state_of_mind/models), [`state_of_mind/train/`](state_of_mind/train) | encoders, predictors, and gradient training |
| [`state_of_mind/benchmark/`](state_of_mind/benchmark) | BrainStateBench: tasks, scoring, latent probes, and benchmark card |
| [`state_of_mind/configs/`](state_of_mind/configs), [`state_of_mind/scripts/`](state_of_mind/scripts) | dataset/Spectator configs and training scripts |
| [`state_of_mind/nbs_data/`](state_of_mind/nbs_data), [`state_of_mind/data/`](state_of_mind/data) | data preparation and loading |

## License

Parts of the code are adapted from [LeWM](https://github.com/lucas-maes/le-wm) (conditional Transformer blocks; MIT) and [TRIBE v2](https://github.com/facebookresearch/tribev2) (stimulus projections and the network that makes the contextual rows; CC BY-NC 4.0). Each adapted file says what it took and what changed, and the licenses are in [`state_of_mind/licenses/`](state_of_mind/licenses). The code and the weights are released under CC BY-NC 4.0 ([LICENSE](LICENSE)).

## Citation

```bibtex
@article{wei2026stateofmind,
  title   = {State of Mind: Learning and Evaluating World Models
             of Naturalistic Brain Dynamics},
  author  = {Yuxiang Wei and Zundong Wu and Weiyu Sun and
             Qizhen Lan and Yanteng Zhang and Vince D. Calhoun},
  journal = {arXiv preprint arXiv:2611.28508},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.28508}
}
```
