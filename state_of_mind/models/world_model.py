"""Latent-only world model: encoder, bounded-history transition, explicit rollout."""
import torch
from torch import nn
from state_of_mind.models.encoder import HistoryEncoder, frame_encoder
from state_of_mind.models.predictor import CONDITION_DIM, CorrectedMatrix, MLPPredictor, Predictor
from state_of_mind.models.stimulus import STIMULUS_DIMS, PrecomputedStimulus, Standardize, StimulusEncoder
from state_of_mind.models.stimulus_context import StimulusContext


class WorldModel(nn.Module):
    def __init__(self, cfg, *, initialize_stimulus=True):
        super().__init__()
        m = cfg["model"]
        self.context_states = m["context_states"]
        # A state is computed at every frame. With a state step s > 1, each transition moves s frames: the predictor
        # reads states s frames apart and predicts the state s frames on, and its movie row for that step joins the
        # rows of the s frames the step covers. The states then form s interleaved chains, one per phase.
        self.state_step = m.get("state_step", 1)
        self.mode = cfg["train"]["mode"]
        self.encoder_frozen = False
        # model.state_tokens 2 with model.fast_window_trs: a state of fast and slow latents (HistoryEncoder).
        tokens = m.get("state_tokens", 1)
        def make_encoder():
            if m.get("fast_window_trs", 0):
                return HistoryEncoder(m["state_window_trs"], m["fast_window_trs"], m["state_dim"] // tokens, m.get("slow_per_dimension", False),
                                      trainable_lags=not cfg["train"].get("freeze_encoder", False), extra_branches=m.get("slow_extra_branches", 0),
                                      fast=frame_encoder(m, m["fast_window_trs"], m["state_dim"] // tokens))
            return frame_encoder(m, m["state_window_trs"], m["state_dim"])
        self.encoder = make_encoder()
        if cfg["train"].get("freeze_fast_encoder", False):
            self.encoder.fast.requires_grad_(False)  # only the lag weights train
        # train.freeze_encoder_body: a Transformer fast encoder's body (its embedding, tokens, blocks and the LayerNorm before the output
        # map) stays fixed and runs without dropout, as a cache of its outputs would; its output map, linear path and the lag weights train.
        if cfg["train"].get("freeze_encoder_body", False):
            self.encoder.fast.fix_body()
        # train.target: what predictions are scored against. "encoder": the encoder's own states (the Transformer world model's frozen
        # encoder; the joint world model's, attached as in LeWM). "initial": a frozen copy of the encoder as initialized (the fine-tuned
        # Transformer world model: the stage-0 encoder, while the encoder trains on). "ema": an EMA copy of the encoder, moving
        # train.target_momentum of the way per optimizer step (the frame's learned teacher).
        self.target = cfg["train"].get("target", "encoder")
        self.target_momentum = cfg["train"].get("target_momentum", 0.998)
        self.target_encoder = None
        if self.target != "encoder":
            self.target_encoder = make_encoder().requires_grad_(False)
            if cfg["train"].get("freeze_encoder_body", False):
                self.target_encoder.fast.fix_body()
        # A training-only linear head from each predicted state to its newest frame's parcels (train.parcel_weight).
        self.parcel_head = nn.Linear(m["state_dim"], 1000) if cfg["train"].get("parcel_weight", 0.0) else None
        self.stimulus = None
        self.stimulus_scale = None
        condition_width = CONDITION_DIM
        if self.mode == "stimulus":
            features = m.get("stimulus_features", "projections")
            if features == "tribev2_context":
                self.stimulus = PrecomputedStimulus(modality_dropout=cfg["train"]["modality_dropout"], width=m.get("context_width", STIMULUS_DIMS[features]))
            else:
                path = m["tribev2_checkpoint"]
                if initialize_stimulus and not path:
                    raise ValueError("Stimulus training requires model.tribev2_checkpoint")
                # initialize_stimulus=False is only for restoring a complete checkpoint.
                self.stimulus = StimulusEncoder(path if initialize_stimulus else None,
                                                modality_dropout=cfg["train"]["modality_dropout"])
            self.stimulus_context = StimulusContext(m["stimulus_context"], m["stimulus_context_trs"],
                                                    width=m.get("context_width", STIMULUS_DIMS[features]) if features == "tribev2_context" else STIMULUS_DIMS[features])
            condition_width = self.stimulus_context.out_width
            # standardize: each feature scaled by training-row statistics (set by the trainer, or by a closed-form fit) in place of
            # the predictor's LayerNorm on each row. With concat, the frames' rows share one set of statistics.
            if m.get("stimulus_scaling", "layernorm") == "standardize":
                block = m.get("context_width", STIMULUS_DIMS[features]) if m["stimulus_context"] == "concat" else None
                self.stimulus_scale = Standardize(condition_width, block=block)
        # With a state step, each frame's row is layer-normalized on its own before the s rows are joined, as the
        # predictor normalizes the one row of a 1-frame step.
        self.row_norm = None
        if self.state_step > 1 and self.stimulus is not None and self.stimulus_scale is None:
            self.row_norm = nn.LayerNorm(condition_width)
        # model.predictor_kind mlp: a residual MLP takes the Transformer's place (MLPPredictor).
        self.predictor = MLPPredictor(m["state_dim"], condition_width, m["mlp_hidden_dim"], m["mlp_depth"], m["predictor_dropout"],
                                      condition_norm=self.stimulus_scale is None, state_tokens=tokens) if m.get("predictor_kind") == "mlp" else Predictor(
                                   m["state_dim"], m["context_states"], m["predictor_depth"],
                                   m["attention_heads"], m["predictor_dropout"],
                                   mlp_dim=m.get("predictor_mlp_dim", 4 * m["state_dim"]),
                                   condition_dim=condition_width * self.state_step,
                                   condition_norm=self.stimulus_scale is None and self.row_norm is None,
                                   history_blocks=m.get("history_blocks", 0),
                                   block_states=m.get("history_block_states", 0),
                                   history_spaced=m.get("history_spaced", False),
                                   conditioning=m.get("conditioning", "adaln"),
                                   chunk_frames=m.get("chunk_frames", 0),
                                   state_tokens=tokens, own_drive=m.get("own_drive", False),
                                   movie_rank=m.get("movie_rank", 0),
                                   output_norm=m.get("predictor_output_norm", True),
                                   movie_tokens=m["stimulus_context_trs"] if m.get("movie_tokens", False) else 0,
                                   token_positions=torch.load(m["token_positions_file"], map_location="cpu", weights_only=True)["positions"] if m.get("token_positions_file") else None,
                                   width=m.get("predictor_width", 0))
        # model.predictor_kind corrected: a fixed matrix plus that Transformer's correction (CorrectedMatrix).
        if m.get("predictor_kind") == "corrected":
            self.predictor = CorrectedMatrix(self.predictor, m["state_dim"], condition_width, condition_norm=self.stimulus_scale is None, state_tokens=tokens)
        # model.chunk_frames n > 0: the predictor returns the next n states at once, and they join the states it reads.
        self.chunk_frames = m.get("chunk_frames", 0)
        # With history blocks, model.history_measured_only keeps the model's own rollout predictions out of the
        # block means (they hold measured states only; a block of predictions drops out), so a long rollout never
        # reads its own predictions as history. The newest state is always read.
        self.history_measured_only = m.get("history_measured_only", False)
        self.zero_brain = m.get("zero_brain", False)

    @property
    def history_blocks(self):
        """History blocks of the predictor (other predictors swapped in by experiments have none)."""
        return getattr(self.predictor, "history_blocks", 0)

    @property
    def read_states(self):
        """The states each prediction reads: the newest context_states, plus any history blocks before them."""
        return getattr(self.predictor, "read_states", self.context_states)

    def encode(self, brain):
        # model.zero_brain (an ablation): the model is given no measured fMRI, so every state is what empty frames encode to.
        if getattr(self, "zero_brain", False):  # subclasses built without WorldModel's __init__ do not set it
            brain = torch.zeros_like(brain)
        return self.encoder(brain)

    def reset_target(self):
        """Start the target encoder as a copy of the encoder, once the encoder is initialized."""
        if self.target_encoder is not None:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def target_states(self, brain):
        """[B, S, D] states the predictions are scored against; None: the encoder's own states."""
        return None if self.target_encoder is None else self.target_encoder(brain)

    @torch.no_grad()
    def follow_target(self):
        """After each optimizer step: an "ema" target encoder moves toward the encoder."""
        if self.target == "ema":
            for mine, theirs in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                mine.lerp_(theirs.detach(), 1 - self.target_momentum)

    def encode_stimulus(self, features):
        """Frozen per-row stimulus rows, then the context block: one condition per transition."""
        if self.mode == "brain":
            return None
        rows = self.stimulus_context(self.stimulus(features))
        return rows if self.stimulus_scale is None else self.stimulus_scale(rows)

    def predict(self, states, conditions=None):
        """Teacher-forced predictions of states 1..T from observed states 0..T-1; with history blocks, of
        states read_states..T only, each from the read_states states before it."""
        if self.mode == "stimulus" and conditions is None:
            raise ValueError("Stimulus-conditioned prediction requires stimulus conditions")
        if getattr(self, "chunk_frames", 0):
            # One chunk from every chunk_frames-th state, so each state after the first read_states is predicted once.
            anchors = range(self.read_states - 1, states.shape[1], self.chunk_frames)
            return self.predictor.predict_chunks(states, conditions, None, anchors, self.chunk_frames).flatten(1, 2)
        if self.history_blocks:
            return self.predictor(states, conditions, positions=range(self.read_states - 1, states.shape[1]))
        return self.predictor(states, conditions)

    def state_valid(self, brain):
        """[B, S] whether each state's frames were all measured (the benchmark passes unmeasured frames as zero)."""
        return brain.ne(0).any(-1).unfold(1, self.encoder.state_window_trs, 1).all(-1)

    def rollout_states(self, history, conditions=None, *, horizon, valid=None, detach_every=0):
        """Autoregressive rollout from H0 latent states.

        conditions must hold H0+horizon-1 rows (or be None in brain mode); row r
        conditions the transition from state r to r+1. Returns [B, horizon, D].
        Only the last context_states predicted/observed states enter each step
        (with history blocks, the last read_states, of which only valid [B, H0] history states and the
        predictions; with model.history_measured_only, the predictions only as the newest state). With
        detach_every n > 0, the states fed back are detached after every n steps
        (truncated backpropagation through time); the returned predictions keep their own gradients.
        """
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if history.ndim != 3 or history.shape[1] < 1:
            raise ValueError("history must hold at least one state")
        if self.mode == "stimulus" and conditions is None:
            raise ValueError("Stimulus-conditioned rollout requires stimulus conditions")
        expected = history.shape[1] + horizon - 1
        if conditions is not None and conditions.shape[1] != expected:
            raise ValueError(f"Conditions must hold {expected} rows for this prefix and horizon")
        if getattr(self, "chunk_frames", 0):
            return self.rollout_chunks(history, conditions, horizon=horizon, valid=valid)
        states = history
        if getattr(self.predictor, "history_blocks", 0):  # self may be another rollout model reusing this method
            if valid is None:
                valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
            outputs = []
            for step in range(horizon):
                end = states.shape[1]
                start = max(0, end - self.read_states)
                condition = None if conditions is None else conditions[:, start:end]
                prediction = self.predictor(states[:, start:], condition, valid[:, start:],
                                            positions=[end - start - 1])[:, -1]
                outputs.append(prediction)
                states = torch.cat((states, prediction[:, None]), dim=1)
                measured = not getattr(self, "history_measured_only", False)
                valid = torch.cat((valid, valid.new_full((valid.shape[0], 1), measured)), dim=1)
                if detach_every and (step + 1) % detach_every == 0:
                    states = states.detach()
            return torch.stack(outputs, dim=1) if detach_every else states[:, history.shape[1]:]
        outputs = []
        for step in range(horizon):
            end = states.shape[1]
            window = states[:, -self.context_states:]
            condition = None if conditions is None else conditions[:, end - window.shape[1]:end]
            prediction = self.predictor(window, condition)[:, -1]
            outputs.append(prediction)
            states = torch.cat((states, prediction[:, None]), dim=1)
            if detach_every and (step + 1) % detach_every == 0:
                states = states.detach()
        return torch.stack(outputs, dim=1) if detach_every else states[:, history.shape[1]:]

    def rollout(self, observed_brain, stimuli=None, *, horizon_trs=1):
        """Only the observed prefix frames and the supplied stimuli enter inference."""
        history = self.encode(observed_brain)
        conditions = self.encode_stimulus(stimuli)
        if self.history_blocks:
            return self.rollout_states(history, conditions, horizon=horizon_trs, valid=self.state_valid(observed_brain))
        if getattr(self, "state_step", 1) > 1:  # self may be another rollout model reusing this method
            return self.rollout_chains(history, conditions, horizon=horizon_trs)
        return self.rollout_states(history, conditions, horizon=horizon_trs)

    def rollout_chunks(self, history, conditions=None, *, horizon, valid=None):
        """Chunked rollout from H0 states and their H0 + horizon - 1 rows, returning [B, horizon, D]: each chunk's states
        join the states the next chunk reads, as valid states, and the next chunk's anchor is the chunk's last state."""
        states, outputs = history, []
        for start in range(0, horizon, self.chunk_frames):
            chunk = self.predictor.predict_chunks(states, conditions, valid, [states.shape[1] - 1],
                                                  min(self.chunk_frames, horizon - start))[:, 0]
            outputs.append(chunk)
            states = torch.cat((states, chunk), dim=1)
            if valid is not None:
                valid = torch.cat((valid, valid.new_ones(chunk.shape[:2])), dim=1)
        return torch.cat(outputs, dim=1)

    def step_rows(self, conditions):
        """[B, R, W] rows, one per frame (row r conditions the frame of state r + 1), as [B, R - s + 1, s * W] step
        rows: row j joins the rows of the s frames of the step from state j to state j + s, oldest first."""
        if conditions is None or self.state_step == 1:
            return conditions
        rows = conditions if self.row_norm is None else self.row_norm(conditions)
        return rows.unfold(1, self.state_step, 1).transpose(-1, -2).flatten(2)

    def chain(self, states, conditions, phase):
        """One chain per window for training: from [B, S, D] states (one per frame; S a multiple of the step s) and
        their [B, S - 1, W] rows, the S / s states s frames apart from frame phase[b] on, with their step rows."""
        s = self.state_step
        index = phase[:, None] + s * torch.arange(states.shape[1] // s, device=states.device)
        states = states.gather(1, index[..., None].expand(-1, -1, states.shape[-1]))
        if conditions is not None:
            rows = self.step_rows(conditions)
            conditions = rows.gather(1, index[:, :-1, None].expand(-1, -1, rows.shape[-1]))
        return states, conditions

    def rollout_chains(self, history, conditions=None, *, horizon):
        """Rollout with a state step s from [B, H0, D] states, one per frame, and their H0 + horizon - 1 rows:
        s interleaved chains, each stepping s frames from one of the s newest states (its newest context_states
        states s frames apart), together predict every frame of the horizon. Returns [B, horizon, D]."""
        s, length = self.state_step, history.shape[1]
        if length < s:
            raise ValueError(f"A state step of {s} needs at least {s} history states")
        if conditions is not None and conditions.shape[1] != length + horizon - 1:
            raise ValueError(f"Conditions must hold {length + horizon - 1} rows for this prefix and horizon")
        rows = self.step_rows(conditions)
        out = history.new_empty(history.shape[0], horizon, history.shape[2])
        for phase in range(s):
            last = length - 1 - phase  # the chain's newest state
            steps = (horizon + phase) // s
            if not steps:
                continue
            index = torch.arange(last % s, last + 1, s, device=history.device)[-self.context_states:]
            first = int(index[0])
            chain_rows = None if rows is None else rows[:, first:first + (len(index) + steps - 1) * s:s]
            # The chain's m-th prediction is the state of frame last + m * s.
            out[:, s - 1 - phase::s][:, :steps] = self.rollout_states(history[:, index], chain_rows, horizon=steps)
        return out

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder_frozen = True
