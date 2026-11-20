# Spectator

This page describes the paper's model, Spectator, end to end: what it is, its architecture, how it is trained and how it is scored. Its run folders and benchmark rows are named `ridge-jepa` (at 180 s: run `ridge-jepa-180s-std-partial`, row `ridge-jepa-180s-std-partial (latent)`). The paper's tables and the runs behind them are mapped in [reproduce](reproduce.md).

**The model is frozen.** Its configs are in [`neuroworld/configs/spectator/`](../neuroworld/configs/spectator), its weights and every intermediate file in `$NEUROWORLD_OUTPUT/spectator-v1/` (with a README mapping each file to the run it came from), and its code is in this repository. [`tests/test_spectator.py`](../tests/test_spectator.py) checks that the configs keep their settings and that the code reproduces the frozen model's stored rollout.

## In plain words

The model watches someone's brain while they watch a movie, and predicts what their brain will do over the next minute.

1. **It summarizes the last 3 minutes of fMRI into a brain state of 2,048 numbers.** Half of the state is a snapshot of the newest 7.5 s (the fast part). The other half is a weighted summary of everything in the 3 minutes (the slow part). Both are plain weighted sums of the measured signal.
2. **It steps that state forward with one matrix.** Each step adds a correction to the state, computed from the state itself and the movie's last 7.5 s. One step is one fMRI frame (1.49 s). The new state is fed back in, so 40 steps forecast a minute.
3. **It reads brain activity off the state with a linear probe**, one per participant, only for scoring. The model itself never predicts brain regions directly.

It is trained in four steps (the detailed list below numbers six: it adds freezing the encoder and fitting the probe). Two train by gradient descent, and two are single regressions:
- **Stage 0** learns the fast part by forecasting fMRI without the movie.
- **The slow part's weights** come from one regression: which mix of the past best predicts the next 24 s.
- **The JEPA phase** fine-tunes both parts so that the state's own future is easy to predict: the state is trained to forecast later states, which are given by a slowly updated copy of the model. This is the step that adds most.
- **The final matrix** is solved in one regression, with a strong penalty on the movie part so it cannot memorize the training movie.

Everything in the model is linear. Every nonlinear part we tried (Transformers in the encoder or the predictor, extra nonlinear corrections) scored lower on new recordings; the Transformer predictor also fits the training recordings much more closely (training gap 2.9 against 1.8). On all Algonauts test data at 180 s, Spectator scores 16.0 / 5.5 / 20.3 on Forecast / State / Updating, against 11.6 / 2.6 / 12.7 for the linear forecaster and 13.2 / 3.4 / 17.9 for its Transformer counterpart.

## What the model is

`ridge-jepa` is a latent world model of a person watching a movie, in the JEPA sense:

- An encoder turns the recent fMRI into a latent state.
- A predictor rolls that state forward under the movie, which plays the role of the action.
- Every training target is an encoded latent, never the parcels.
- Parcels appear only at scoring, through a frozen linear probe per participant.

```
last H seconds of fMRI ──► encoder ──► state s_t = [fast latent; slow latent]
                                                │
movie features m_t (newest 5 frames) ───────────┤
                                                ▼
                    s_{t+1} = s_t + W · [s_t ; m_t ; 1]          (rolled forward frame by frame)
                                                │
                                                ▼
                        parcels = probe_participant(s_{t+k})    (scoring only)
```

**History length H is a setting of the model, not part of it.** We report H = 180 s, the length that scores highest on all test data; the paper's main table also gives 7.5 and 120 s, and its appendix 15 to 90 s. Below, the numbers that depend on H are given at 180 s.

## Architecture

**Input.** fMRI averaged in the 1,000 Schaefer parcels, one frame every 1.49 s, with Algonauts' own preprocessing. Frames before a run start, and unmeasured frames, are zeros.

**Encoder.** It has two parts:
- **Fast latent.** One linear map from the newest 5 frames (5 × 1,000 values) to 1,024 values, about 5.1 M parameters.
- **Slow latent.** For each of the 1,024 dimensions, a weighted sum of the fast latent at every position in the window, measured from the latent that empty frames encode to, times one shared scale.
  - The number of positions is the window's frames minus 4. At 180 s (120 frames) that is 116, so 116 × 1,024 lag weights.
  - At 7.5 s the window holds only the 5 frames, so there is no slow latent.
- **State.** The two latents side by side, 2,048 values.

**Movie.** TRIBEv2's contextual features, computed with causal attention, give 2,048 values per frame. Each feature is standardized by its mean and spread over all training frames. The predictor reads the rows of the newest 5 frames (7.5 s), 10,240 values.

**Predictor.** One matrix, which predicts the change of the state from the state, the movie rows and a constant: `s_{t+1} = s_t + W [s_t ; m_t ; 1]`. W has 2,048 × 12,289 entries, about 25 M. A forecast applies it frame by frame and feeds each predicted state back in. At 180 s, its largest eigenvalue is 0.9951, so rollouts are stable.

## Training, end to end

All training uses Algonauts: Friends seasons 1–5 for training, and season 6 for every epoch selection. Movie10's films are never trained on. Settings are picked by the best score on all data not used for training, on every dataset (Friends S6 and all four Movie10 films; HCP's held-out clips, run 4 and new participants). HCP uses the same encoder, with steps 2 and 5 refitted on HCP ([below](#on-hcp)).

1. **Stage 0: the fast map, from brain data alone.**
   - The linear map trains by forecasting fMRI with no movie input. A 6-block Transformer predictor and a readout to parcels are used here and then discarded.
   - Settings: AdamW (rate 5e-5, weight decay 0.001), batch 1,024, a 100-epoch cosine schedule.
   - It runs without early stopping, because it is unstable near epoch 7 and recovers later.
   - The epoch is selected by forecast correlation on Friends S6. The best is epoch 23, at 0.221.
2. **The slow latent's lag weights, in closed form.**
   - For each dimension, a ridge regression from the lagged fast latents to the mean of the next 16 fast latents (24 s ahead), on the training recordings.
   - The penalty is chosen on Friends S6. At 180 s it is 1e-4, and the fit explains 78% of that target there.
   - The scale is then set so the slow latent spreads as much as the fast latent.
3. **The JEPA phase: the encoder trains against its own targets.**
   - The whole encoder (the fast map and the lag weights) trains at rate 1e-5, against an EMA copy of itself (momentum 0.998) that gives the targets.
   - Its predictor is a matrix of the same form, trained at 5e-5 from a closed-form start. It reads one layer-normalized movie row, shifted 3 s for the haemodynamic delay, and the movie is dropped from 30% of sequences.
   - The loss is the squared error between predicted and target latents: one step ahead from encoded states, plus a 40-step rollout (60 s), weighted equally.
   - Same optimizer and schedule as stage 0, with gradient clipping at 1 and bfloat16 everywhere except the encoder, which stays in 32-bit floats.
   - It stops after 10 epochs without a better probe score on Friends S6. At 180 s it ran 69 epochs and kept epoch 58, about 1.3 minutes per epoch on 2 A100s.
4. **Freeze the encoder.** The JEPA phase's matrix is discarded.
5. **The final matrix, in closed form.**
   - A ridge regression of one-step state changes on the training recordings.
   - It uses every moment whose window holds H of measured frames. It also uses moments with less history, whose window starts before the run, with those frames zero as at scoring. Each keeps at least 30 s inside the run, as every forecast start does. At 180 s that is 1,920,128 transitions (104,696 full and 15,312 partial windows).
   - The movie rows are the newest 5, standardized per feature, with no delay shift.
   - The penalty is 1e-3 times the mean diagonal of the Gram matrix. It is multiplied by 10 on the state block, and the constant is not penalized.
   - On the movie block it is multiplied by a factor chosen on each dataset's test data: 100 on Algonauts, 10,000 on HCP. The Gram matrix grows with every transition, but the movie map learns only from distinct movie moments, and HCP's 148 participants all watched the same 45 minutes of movie, so HCP needs a much stronger penalty. Chosen on all test data.
   - The previous fit (layer-normalized rows, full windows only; run `D180-win5`) scored 0.3 lower on Forecast. The paper's ablations of the encoder and predictor are paired against it.
6. **The probe, for scoring only.**
   - For each participant, a ridge map from predicted states to parcels, fitted on the model's own rollouts on the training recordings, under the benchmark's protocols.
   - Each latent coordinate is standardized, and the penalty is chosen by cross-validation across movies.
   - It is then frozen before any scored recording.

No decoder is trained. In Updating, a measured frame enters through the encoder. Hidden frames in the window are filled by decoding predicted states through the probe and encoding them again.

**Cost at 180 s.** Stage 0 takes about 3 minutes per epoch on one A100. The lag weights and the final matrix take minutes. The JEPA phase takes about 1.5 hours on 2 A100s. Scoring the whole panel takes about 3 hours on one GPU.

## On HCP

HCP's frames are 1 s apart. The model keeps the Algonauts encoder's fast map, which reads 5 frames, and refits two closed-form parts on HCP's 148 training participants:
- **The lag weights** (step 2): 180 s of history (176 lags), and the mean of the next 24 s as the target.
- **The final matrix** (step 5): with movie factor 10,000.

The probe is pooled over participants, as for every HCP model, so new participants use it unchanged. HCP removes frames for head motion; in the oldest frames of a training window those enter as zero, as the benchmark gives them (`data.history_fill`). Retraining the whole recipe on HCP scored lower at every penalty.

## Code and runs

- **Configs:** [`neuroworld/configs/spectator/`](../neuroworld/configs/spectator): `algonauts-stage0.yaml` (step 1), `algonauts-jepa.yaml` (step 3), `algonauts.yaml` and `hcp.yaml` (the models). They are fully resolved, so later changes to defaults cannot change them.
- **End to end:** `OUT=<dir> neuroworld/scripts/spectator.sh algonauts` (or `hcp`) runs every step and the scoring.
- **Closed-form steps, run directory and scoring:** [`neuroworld/spectator/`](../neuroworld/spectator), run as `python -m neuroworld.spectator`: `lags.py` (step 2), `matrix.py` (step 5, and the JEPA phase's start matrix), `build.py` (the encoder file and the run directory), `score.py` (every benchmark task with the model's own probe).
- **Model and gradient steps:** `HistoryEncoder` and `BrainEncoder` in [`models/encoder.py`](../neuroworld/models/encoder.py), the predictor in [`models/predictor.py`](../neuroworld/models/predictor.py); stage 0 and the JEPA phase in [`train/latent.py`](../neuroworld/train/latent.py), launched by `neuroworld/scripts/launch_stage1.sh`.
- **Probe and latent scoring:** [`benchmark/latent.py`](../neuroworld/benchmark/latent.py).

**Frozen files** (`$NEUROWORLD_OUTPUT/spectator-v1/`), and the runs they came from (run root `$NEUROWORLD_OUTPUT/checkpoints/algonauts/wave2_20261102` unless noted):

| Step | Run or file |
|---|---|
| 1, stage 0 | run `phase0-k5` (`checkpoints/algonauts/state_20261030`); epoch 23 |
| 2, lag weights | `history-start-180.pt` |
| 3, JEPA phase | run `D180`, epoch 58; encoder `history-D180.pt` |
| 5, final matrix | run `ridge-jepa-180s-std-partial`, file `direct-start-ridge-jepa-180s-std-partial.pt`; row `ridge-jepa-180s-std-partial (latent)` |
| On HCP | run `ridge-jepa-180s-hcp-refit-m1e4`, row `ridge-jepa-180s-hcp-refit-m1e4 (latent)` |

**How run names read** (`<predictor>-<encoder>-<history>[-<change>]`):
- **Predictor.** `ridge` is a closed-form matrix, `transformer` a Transformer trained by gradient, and `matrix` a matrix trained by gradient (the JEPA-phase runs).
- **Encoder.** `jepa` means after the JEPA phase, `frozen` stage 0 only. `tfforecast` is a Transformer encoder trained by forecasting, and `linpath` adds a linear path to it.
- **Changes.** `std`: standardized movie rows. `partial`: partial-history windows in the final fit. `m<N>`: N movie rows (on the final fit's own rows) or movie factor N. `nomovie`, `nofmri`: one input removed. `hcp-refit`: refitted on HCP.
- Older run folders keep letter names: `D180` is the 180-s JEPA phase (`matrix-jepa-180s-m1-r40`), `D180-win5` is `ridge-jepa-180s` with the previous fit, and `D180-start-win5` is `ridge-frozen-180s`.
