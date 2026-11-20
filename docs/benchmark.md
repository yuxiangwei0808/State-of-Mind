# BrainStateBench

BrainStateBench evaluates brain world models: models that keep an internal state built from measured fMRI and simulate how it evolves while a person watches a movie. It asks how much a person's observed brain activity improves prediction beyond what the movie already tells, how long that benefit lasts, and how well a few new measurements restore it. The code is in [`neuroworld/benchmark`](../neuroworld/benchmark); the paper's results are mapped in [reproduce](reproduce.md).

## Tasks

Every task scores forecasts of measured fMRI (Schaefer-1000 parcels) against the recorded targets. Skill is

  skill = 100 × (1 − Σ (y − ŷ)² / Σ (y − ȳ)²),

summed over every scored frame and parcel, where ȳ is the reference prediction named below. 0% is the reference, and higher is better. Every scored frame weighs the same.

| Task | Question | Reference (0%) |
|---|---|---|
| **Forecast** | Can the model predict the next minute (40 frames on Algonauts, 60 on HCP) from the fMRI history and the movie, without new fMRI? | predicting the run's mean per parcel |
| **State** | Does the observed history help predict how two viewings of the same scene differ? | predicting no difference |
| **Updating** | Can a few new measurements in the minute (4, one every 12–15 s) keep the forecast on track? | predicting the run's mean per parcel |

- **Forecast** ([`forecast.py`](../neuroworld/benchmark/forecast.py)). Origins every 3 s; the last 30 s before each origin must be measured. Curves give the skill at each step ahead; the paper also reports the 0–7.5, 7.5–30 and 30–60 s bands.
- **State** ([`state.py`](../neuroworld/benchmark/state.py)). For two viewings of the same material, each forecast from its own past, the score is 1 − Σ(Δŷ − Δy)² / Σ(Δy)² over the minute, with Δ = viewing 1 − viewing 2. It rewards a predicted difference aligned with the measured one and penalizes an exaggerated one; a stimulus-only model scores exactly 0. For a matched pair the total squared error is exactly 2‖m̂ − m‖² + ½‖d̂ − d‖² (m the mean, d the difference of the viewings), and State scores the second part. Pairs: the Movie10 films watched twice (*Hidden Figures*, *Life*; the same movie history, so only the brain differs) and the repeated clip that ends every HCP run (different preceding clips; predicted and measured differences are centered across participants at each origin first). State is noise-limited: measurement noise in both viewings enters the difference, so a few percent is not trivial.
- **Updating** ([`updating.py`](../neuroworld/benchmark/updating.py)). After a measured history, the model runs through the next minute with fMRI hidden. A schedule reveals B frames, evenly spaced (B = 0, 2, 4, 8, 16, every frame, and a "gap" schedule with no measurement for 25 s); between reveals the model forecasts the next segment from everything before it, with a mask marking only the revealed frames as valid. Each frame is scored before it is revealed. The headline uses B = 4. Episodes start every 30 s. Each model bridges the gaps its own way (a latent model rolls its state through them; IPSID skips the measurement update). For a model that would enter a missing frame as a constant, such as the linear forecaster, `--own-fill` fills every frame it is not shown with its own earlier forecast; the paper scores the linear forecaster this way.
- **Predictor nonanticipation** ([`nonanticipation.py`](../neuroworld/benchmark/nonanticipation.py)), a pass/fail check: two queries that share the brain history and the movie up to t₀ + k must give the same forecast before t₀ + k (k = 5, 10, 20 frames).

## How a model is queried

A query names a participant, an opaque run handle, a stimulus handle and an origin t₀. The model receives:
- the **measured fMRI** before t₀, with a mask of valid frames (frames before the run start, and frames Updating hides, are invalid);
- the **movie**, through the stimulus handle: the stimulus features before t₀ and for the frames to forecast (the future stimulus is given, so the model simulates the brain, not the movie). Repeated viewings share one handle, and the run handle is opaque, so neither names the viewing;
- a **horizon**: the next minute.

It returns forecast fMRI for frames [t₀, t₀ + horizon), shaped [queries, horizon, 1000]: `predict(QueryBatch)` in [`interface.py`](../neuroworld/benchmark/interface.py). A model never sees fMRI at or after t₀. The tasks change one part of a query at a time: another past (State), a sparse past (Updating), another future movie from t₀ + k on (nonanticipation).

**Latent models** ([`latent.py`](../neuroworld/benchmark/latent.py)). A model that exposes its predicted states (`predict_states(QueryBatch)` → `(fMRI or None, states [queries, horizon, D])`) is scored through a frozen linear probe of those states, as the row `<name> (latent)`. The probe is an affine ridge map from the predicted latent to measured fMRI, fitted on the model's own rollouts on training recordings (open loop, and with 16 measurements in the minute), one per participant on Algonauts and one pooled on HCP. Each latent coordinate is standardized, and the penalty is chosen by cross-validation across movies. The probe is stored with its provenance (a digest of the model's weights and of the benchmark code, the recordings, the penalty) in `<output>/<dataset>/probes/` and reused while they match. Because the probe is fitted on the model's own rollouts, the score is the information the latent holds about the future, not the accuracy of the latent trajectory in its own coordinates. Spectator is scored this way.

**The benchmark never retrains a model.** Around it, it fits only the latent probe and, for models with participant-specific heads, a linear head per new HCP participant from that participant's run 1 ([`adapt.py`](../neuroworld/benchmark/adapt.py)).

## Adding a model

A model is one Python class with a `name` and a `predict` method, built as `Class(dataset=..., device=...)`; [`examples/persistence.py`](../examples/persistence.py) is a complete example.

```python
class MyModel:
    def __init__(self, dataset, device):
        self.name = "my model"                    # the result name
    def predict(self, batch):                     # QueryBatch -> [queries, horizon, 1000] fMRI
        ...
```

```bash
python -m neuroworld.benchmark suite --dataset algonauts --model py:my_model.py:MyModel --device cuda
python -m neuroworld.benchmark profile --reference "my model"
```

A `QueryBatch` holds `prefix` (the fMRI history, NaN where invalid), `prefix_valid`, `horizon`, `tr`, `subject_ids`, `timeline_ids` (stimulus handles), `origins` (frame indices in the stimulus timeline) and, for remapped queries, `splices` and `histories`. A stimulus-conditioned model reads the stimulus rows through `interface.stimulus_source`, which applies splices and history swaps. A query can hold up to 256 origins, so a model should process it in chunks that fit its memory.

## Data

Every recording not used for training is test data, used both to choose settings and to report results.

| Evaluation set | Data | Role |
|---|---|---|
| Algonauts training | *Friends* S1–5, 4 participants, TR 1.49 s | training |
| Algonauts | *Friends* S6 (`val`); Movie10 *Hidden Figures* and *Life*, each watched twice (`dev`); *Bourne* and *Wolf* (`test`) | test |
| HCP training | 148 participants, runs 1–3, TR 1 s, training frames | training |
| HCP known participants | the same 148: one held-out clip of each movie (`val`) and run 4 (`test`) | test |
| HCP new participants | 36 participants: run 1 calibrates participant-specific heads (`heldout-fit`); runs 2–3 (`heldout-adapt`) and run 4 (`heldout-test`) | test |

- **fMRI:** Schaefer-1000 parcel time series, as released (Algonauts, CNeuroMod's preprocessing) or prepared by us (HCP; [data preparation](data-preparation.md)).
- **Stimulus features:** V-JEPA 2 (video), w2v-BERT (audio) and Llama-3.2-3B (text over timed transcripts), as TRIBE v2's frozen projections or its contextual rows (`--features projections|context|raw`). HCP has no transcripts.
- **Resampling units:** content parts on Algonauts (23 *Friends* S6 episodes and 43 Movie10 segments, the 66 units of the paper's intervals) and participants on HCP. Queries overlap in time, so intervals resample units, never queries.

## Statistics and provenance

- Scores are pooled within a dataset by adding per-unit sums; datasets are never pooled.
- Intervals are 95% bootstrap intervals over units; comparisons are paired differences on the same units ([`report.py`](../neuroworld/benchmark/report.py), and `_paired` in [`profile.py`](../neuroworld/benchmark/profile.py)).
- Every result file carries the protocol version, the source commit and a digest of the benchmark code. The fitted references (IPSID, the linear forecaster) are stored with their settings and refitted when they change ([`baselines.py`](../neuroworld/benchmark/baselines.py), [`references.py`](../neuroworld/benchmark/references.py)).

## Limits

- **Offline preprocessing.** fMRI is preprocessed over whole runs (Algonauts: confound regression, a cosine high-pass filter and z-scoring per run; HCP: z-scoring and detrending per run), so an observed history depends slightly on the run's later frames. Every model receives the same inputs; the results are for forecasting withheld, offline-preprocessed frames, not real-time forecasting.
- **Stimulus-feature lookahead** is not tested: TRIBE v2's audio features use future context within 60-s chunks. The future movie is given by design.
- **Exposure.** TRIBE v2 was trained on CNeuroMod, which includes every Algonauts recording; models using its features or predictions inherit that exposure.
- **Participants.** Algonauts has 4 participants, so its intervals cover content, not new people. HCP's family structure is not used in the split (twins and siblings may fall on both sides).

## Commands

```bash
python -m neuroworld.benchmark suite --dataset {algonauts,hcp_movies} --model <model> [--features ...]   # every task for one model
python -m neuroworld.benchmark forecast | state | updating | nonanticipation --dataset ... --model ...
python -m neuroworld.benchmark updating ... --own-fill                                                 # fill unseen frames with the model's own forecast
python -m neuroworld.benchmark profile [--reference <row>]                                             # the card: every scored row
python -m neuroworld.benchmark report --dataset ... --split ... [--task state|updating]
```

- **Models:** `py:<file.py>:<Class>` (an added model), `run:<dir>` (a run trained with `neuroworld.train` that has its own fMRI readout; for a latent model without one, use `latent:run:<dir>`), `latent:<model>` (a latent model's probe row, plus its own decoder's row when it has one), `direct-linear` (the linear forecaster), `ipsid` (IPSID), `tribev2` (TRIBE v2's saved predictions), `persistence`. Spectator is scored with `python -m neuroworld.spectator score` ([Spectator](spectator.md)).
- **Results:** under `--output` (default `artifacts/benchmark`): `<dataset>/<split>/<row>/results.json` (Forecast), `<dataset>/{state,updating,nonanticipation}/<row>.json|npz`, `<dataset>/probes/`, and `card.md`.

## Tasks that were tried and dropped

Retention (decoding past stimulus content from forecasts) and Content (decoding the current scene): the past movie is given to the model, so neither isolates what the brain holds. Which-viewing choice as State's score: every fMRI-reading model scored 86–97%, because the first second nearly copies the last frame. Updating with 16 measurements as the headline: at one measurement every 4 s, a model reading neither the movie nor anything but each parcel's own past matched the Kalman filter. Intervention on synthetic data: it retrains each model on a generator, so it tests a training recipe rather than the model.
