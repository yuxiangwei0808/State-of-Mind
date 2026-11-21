"""Spectator, the model the paper reports (`ridge-jepa` in the docs and on disk), frozen.

A linear latent world model of a person watching a movie:

    state      s_t = [fast latent; slow latent]           2,048 values from the last 180 s of fMRI
    step       s_{t+1} = s_t + L s_t + W m_t + c           m_t: the movie's TRIBE v2 contextual rows of the newest 5 frames
    reading    parcels = s_t P_viewer + b_viewer           one linear probe per viewer, fitted for scoring only

Training, in six steps (the configs are in neuroworld/configs/spectator/, the frozen weights in
$NEUROWORLD_OUTPUT/spectator-v1/):

1. Stage 0 (gradient descent): the fast map, by forecasting fMRI without the movie.
       neuroworld/scripts/launch_stage1.sh neuroworld/configs/spectator/algonauts-stage0.yaml
2. Lag weights (closed form): the slow latent, a ridge fit of the mean of the next 24 s of fast latents.
       python -m neuroworld.spectator lags <stage-0 best.pt> <out .pt> --frames 120 --ahead 16
3. JEPA phase (gradient descent): the whole encoder against its EMA copy, from a closed-form start matrix.
       python -m neuroworld.spectator matrix neuroworld/configs/spectator/algonauts-jepa.yaml <start .pt> \
           --movie-factor 10 --state-factor 1 --full-windows-only --encoder <step-2 .pt>
       python -m neuroworld.spectator config neuroworld/configs/spectator/algonauts-jepa.yaml <jepa .yaml> \
           --encoder <step-2 .pt> --predictor <start .pt>
       RUN_DIR=<JEPA run> neuroworld/scripts/launch_stage1.sh <jepa .yaml>
4. Freeze the encoder, in the format the next steps load:
       python -m neuroworld.spectator encoder <JEPA run best.pt> <step-2 .pt> <encoder .pt>
5. Final matrix (closed form): a ridge fit of one-step state changes, the movie block penalized 100 times more.
       python -m neuroworld.spectator matrix neuroworld/configs/spectator/algonauts.yaml <matrix .pt> --encoder <encoder .pt>
6. The model's run directory, and its scores on every benchmark task (the probe is fitted here):
       python -m neuroworld.spectator assemble neuroworld/configs/spectator/algonauts.yaml <matrix .pt> <run dir> --encoder <encoder .pt>
       python -m neuroworld.spectator score <run dir> --seconds 180 --dataset algonauts --store <result store>

neuroworld/scripts/spectator.sh runs all of it. On HCP, steps 2 and 5 are refitted on HCP from the Algonauts encoder
(--frames 180 --ahead 24; movie factor 10,000), with neuroworld/configs/spectator/hcp.yaml. Other history lengths change
only the window (HISTORY=<seconds> in spectator.sh: --frames and --set model.state_window_trs=<frames> in steps 2-6); at
7.5 s the window holds only the fast latent's 5 frames, so step 2 is skipped and the JEPA phase starts from stage 0.
"""
