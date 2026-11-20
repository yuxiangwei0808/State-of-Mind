"""Direct fMRI forecaster (the fMRI forecaster and stage-0 forecaster): the world model without latent training or Stage 2.

Uses the same encoder, predictor and stimulus conditioning as WorldModel, plus
a per-subject readout trained jointly on next-frame fMRI MSE. Rollout decodes
each predicted frame and feeds it back through the encoder.
"""
import torch
from neuroworld.models.readout import Readout
from neuroworld.models.world_model import WorldModel


class FmriForecaster(WorldModel):
    forecasts_fmri = True

    def __init__(self, cfg, subjects, *, initialize_stimulus=True):
        super().__init__(cfg, initialize_stimulus=initialize_stimulus)
        self.readout = Readout(cfg["model"]["state_dim"], cfg["readout"]["hidden_dim"], subjects)

    def predict_frames(self, brain, stimuli, subject_index, phase=None):
        """Teacher-forced frames K..end of each window from the states before them: [B, S-1, 1000]. With a state
        step s, those of one chain per window, from frame phase[b]: frames K - 1 + phase + s * m, m = 1..S/s - 1."""
        states, conditions = self.encode(brain), self.encode_stimulus(stimuli)
        if phase is not None:
            states, conditions = self.chain(states, conditions, phase)
        hidden = self.predict(states[:, :-1], conditions)
        return self.readout(hidden, subject_index)

    def rollout_frames(self, observed_brain, stimuli, subject_index, *, horizon_trs):
        """Autoregressive fMRI rollout from K+H0-1 observed frames: [B, horizon, 1000].

        stimuli hold H0+horizon-1 rows; row r conditions the transition from state r to r+1. With a state step s,
        each frame is decoded from the state predicted from the state s frames before it (that state's chain:
        context_states states s frames apart, re-encoded from the observed and decoded frames).
        """
        if type(horizon_trs) is not int or horizon_trs < 1:
            raise ValueError("horizon must be a positive integer")
        if self.mode == "stimulus" and stimuli is None:
            raise ValueError("Stimulus-conditioned rollout requires stimulus conditions")
        k = self.encoder.state_window_trs
        prefix = observed_brain.shape[1] - k + 1
        if prefix < 1:
            raise ValueError("The observed prefix must hold at least one state")
        conditions = self.encode_stimulus(stimuli)
        if conditions is not None and conditions.shape[1] != prefix + horizon_trs - 1:
            raise ValueError(f"Conditions must hold {prefix + horizon_trs - 1} rows for this prefix and horizon")
        s = self.state_step
        if prefix < s:
            raise ValueError(f"A state step of {s} needs at least {s} observed states")
        rows = self.step_rows(conditions)
        frames, predicted = observed_brain, []
        for step in range(horizon_trs):
            end = prefix + step  # the predicted state's index
            count = min(self.context_states, end // s)  # the chain's states before it
            first = end - count * s  # the chain's oldest state
            states = self.encode(frames[:, first:first + k + (count - 1) * s])[:, ::s]
            condition = None if rows is None else rows[:, first:end - s + 1:s]
            hidden = self.predictor(states, condition)[:, -1:]
            frame = self.readout(hidden, subject_index)
            predicted.append(frame)
            frames = torch.cat((frames, frame), dim=1)
        return torch.cat(predicted, dim=1)


def build_model(cfg, subjects=None, *, initialize_stimulus=True):
    """WorldModel for train.objective latent; FmriForecaster (needs subjects) for fmri."""
    if cfg["train"]["objective"] == "fmri":
        if not subjects:
            raise ValueError("The fMRI forecaster needs the training subject IDs")
        return FmriForecaster(cfg, subjects, initialize_stimulus=initialize_stimulus)
    return WorldModel(cfg, initialize_stimulus=initialize_stimulus)
