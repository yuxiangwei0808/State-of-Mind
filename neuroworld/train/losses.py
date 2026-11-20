"""Training objectives: latent world model (next-latent MSE + SIGReg) or direct fMRI forecasting.

By default both prediction and target paths stay attached to the encoder, as in
le-wm/train.py::lejepa_forward, with no stop-gradient or EMA; train.target initial or ema scores
the predictions against a separate target encoder instead (models/world_model.py).
"""
import torch


def rollout_steps_at(train_cfg, epoch):
    """The rollout length at an epoch: train.rollout_steps, or the train.rollout_schedule entry reached by then."""
    steps = train_cfg["rollout_steps"]
    for start, length in train_cfg.get("rollout_schedule") or ():
        if epoch >= start:
            steps = length
    return steps


def rollout_step_weights(kind, steps, device):
    """Weights of the rollout steps 1..steps: uniform, or rising or falling linearly to steps times the first."""
    ranks = torch.arange(1, steps + 1, device=device, dtype=torch.float32)
    return {"uniform": torch.ones_like(ranks), "ramp_up": ranks, "ramp_down": ranks.flip(0)}[kind]


def latent_error(predicted, target, train_cfg):
    """[..., D] error per latent dimension: squared, or with train.latent_error l1 absolute times train.l1_scale; train.slow_loss_weight scales the second half of the state, its slow latent."""
    if train_cfg.get("latent_error", "squared") == "l1":
        error = train_cfg.get("l1_scale", 1.0) * (predicted - target).abs()
    else:
        error = (predicted - target).square()
    weight = train_cfg.get("slow_loss_weight", 1.0)
    if weight != 1.0:
        half = error.shape[-1] // 2
        error = torch.cat((error[..., :half], weight * error[..., half:]), dim=-1)
    return error


def load_teacher(run_dir):
    """The frozen world model of an earlier run (train.teacher_run), always in eval mode: its own predictions from the student's encoded
    states are extra targets (latent_loss). It must have the student's encoder, so both read the same latent states."""
    from pathlib import Path
    from neuroworld.train.readout import load_world_model
    from neuroworld.utils.config import load_config
    run = Path(run_dir)
    model, _ = load_world_model(load_config(run / "config.yaml"), run / "best.pt", "cpu")
    return model.eval().requires_grad_(False)


def teacher_losses(model, teacher, batch, states, predictions, rolled, rows, prefix, steps, train_cfg, extra):
    """(generated, pairs): the student against a frozen teacher world model on the same encoded states (train.teacher_run).
    generated: the student's teacher-forced predictions and rollout against the teacher's own, from the same states under the same
    movie rows, weighted as the real terms (train.prediction_weight, train.rollout_weight). pairs: the starting state of each window
    under the movie rows of the window before it in the batch (a shuffled batch, so another moment), the student's rollout against the
    teacher's. The teacher reads the true movie rows (no modality dropout), as the real targets come from the true movie."""
    zero = states.new_zeros(())
    with torch.no_grad(), torch.autocast(states.device.type, enabled=False):
        given = states.detach().float()
        taught = teacher.encode_stimulus(batch["stimuli"]).float()
        first, taught_rows = given[:, :prefix], taught[:, :prefix + steps - 1]
        if train_cfg["teacher_weight"]:
            teacher_next = teacher.predict(given[:, :-1], taught)[:, -predictions.shape[1]:]
            teacher_rolled = teacher.rollout_states(first, taught_rows, horizon=steps)
        if train_cfg["teacher_pair_weight"]:
            other = torch.roll(torch.arange(states.shape[0], device=states.device), 1)
            teacher_pairs = teacher.rollout_states(first, taught_rows[other], horizon=steps)
    generated = pairs = zero
    if train_cfg["teacher_weight"]:
        generated = (train_cfg.get("prediction_weight", 1.0) * latent_error(predictions, teacher_next, train_cfg).mean()
                     + train_cfg["rollout_weight"] * latent_error(rolled, teacher_rolled, train_cfg).mean())
    if train_cfg["teacher_pair_weight"]:
        paired = model.rollout_states(states[:, :prefix], rows[other], horizon=steps, **extra)
        pairs = train_cfg["rollout_weight"] * latent_error(paired, teacher_pairs, train_cfg).mean()
    return generated, pairs


def latent_loss(model, batch, regularizer, train_cfg, epoch=0, teacher=None):
    states = model.encode(batch["brain"])  # [B, S, D]
    # A separate target encoder (train.target initial or ema) gives the states predictions are scored against;
    # otherwise they are the encoder's own states.
    targets = model.target_states(batch["brain"]) if hasattr(model, "target_states") else None
    conditions = model.encode_stimulus(batch["stimuli"])
    step = getattr(model, "state_step", 1)
    if step > 1:
        # With a state step s, each window trains one chain of states s frames apart, from a random phase (the
        # objective below then sees each chain state as one state and each step row as one row).
        phase = torch.randint(step, (states.shape[0],), device=states.device)
        states, conditions = model.chain(states, conditions, phase)
        if targets is not None:
            targets, _ = model.chain(targets, None, phase)
    targets = states if targets is None else targets
    if states.shape[1] < 2:
        raise ValueError("Training windows need at least two states")
    predictions = model.predict(states[:, :-1], conditions)
    # With history blocks, the teacher-forced targets are the states after the first read_states.
    prediction_loss = latent_error(predictions, targets[:, targets.shape[1] - predictions.shape[1]:], train_cfg).mean()
    # prediction_weight 0 trains on the rollout alone (the teacher-forced error is still reported).
    loss = train_cfg.get("prediction_weight", 1.0) * prediction_loss
    rollout_loss = states.new_zeros(())
    steps = rollout_steps_at(train_cfg, epoch)
    if steps:
        # As in V-JEPA 2-AC, add an autoregressive loss to the teacher-forced one: from the first
        # C observed states, predict R states feeding predictions back, with gradients through the
        # whole rollout. Targets are those of the one-step loss.
        prefix = getattr(model, "read_states", model.context_states)
        if states.shape[1] < prefix + steps:
            raise ValueError("sequence_states must cover the states the predictor reads + rollout_steps")
        rows = None if conditions is None else conditions[:, :prefix + steps - 1]
        # rollout_detach_every n > 0 truncates backpropagation: every n steps, the fed-back states are detached.
        detach = train_cfg.get("rollout_detach_every", 0)
        extra = {"detach_every": detach} if detach else {}
        rolled = model.rollout_states(states[:, :prefix], rows, horizon=steps, **extra)
        error = latent_error(rolled, targets[:, prefix:prefix + steps], train_cfg)
        kind = train_cfg.get("rollout_step_weights", "uniform")
        if kind == "uniform":
            rollout_loss = error.mean()
        else:
            # A weighted mean over steps, so the loss keeps the scale of the uniform mean.
            weights = rollout_step_weights(kind, steps, error.device)
            rollout_loss = (weights * error.mean(dim=(0, 2))).sum() / weights.sum()
        loss = loss + train_cfg["rollout_weight"] * rollout_loss
    teacher_terms = {}
    if teacher is not None:
        generated, pairs = teacher_losses(model, teacher, batch, states, predictions, rolled, rows, prefix, steps, train_cfg, extra)
        loss = loss + train_cfg["teacher_weight"] * generated + train_cfg["teacher_pair_weight"] * pairs
        teacher_terms = {"teacher_loss": generated.detach(), "teacher_pair_loss": pairs.detach()}
    parcel_loss = states.new_zeros(())
    head = getattr(model, "parcel_head", None)
    if head is not None:
        # Each predicted state, teacher-forced and rolled out, read by the training-only parcel head against its
        # newest frame (state j ends at frame j + k - 1), with the two terms weighted as in the latent loss.
        brain, k, total = batch["brain"], model.encoder.state_window_trs, states.shape[1]
        first = k - 1 + total - predictions.shape[1]
        parcel_loss = (head(predictions) - brain[:, first:first + predictions.shape[1]]).square().mean()
        if steps:
            parcel_loss = parcel_loss + train_cfg["rollout_weight"] * (
                head(rolled) - brain[:, k - 1 + prefix:k - 1 + prefix + steps]).square().mean()
        loss = loss + train_cfg["parcel_weight"] * parcel_loss
    sigreg = states.new_zeros(())
    if not model.encoder_frozen:
        # As in LeWM: one sample per window at each state position, averaged over positions.
        sigreg = regularizer(states.transpose(0, 1))
        ramp = train_cfg.get("sigreg_ramp_epochs", 0)  # the weight rises over this many epochs, then stays
        loss = loss + train_cfg["sigreg_weight"] * (min(1.0, (epoch + 1) / ramp) if ramp else 1.0) * sigreg
    spread = states.flatten(0, 1).std(dim=0, unbiased=False).mean()
    return loss, {"prediction_loss": prediction_loss.detach(), "rollout_loss": rollout_loss.detach(),
                  "sigreg": sigreg.detach(), "latent_spread": spread.detach(), **teacher_terms,
                  **({"parcel_loss": parcel_loss.detach()} if head is not None else {})}


def fmri_loss(model, batch, train_cfg):
    """Next-frame fMRI MSE for the direct fMRI forecaster, plus the optional rollout term.

    The rollout term mirrors latent_loss: from the first C observed states, predict R frames,
    feeding each decoded frame back through the encoder, with gradients through the rollout.
    """
    brain = batch["brain"]
    k, s = model.encoder.state_window_trs, getattr(model, "state_step", 1)
    subjects = model.readout.subject_indices(batch["subject_id"], brain.device)
    if s > 1:
        # One chain per window, from a random phase, as in latent_loss; each chain state m > 0 predicts its frame.
        phase = torch.randint(s, (brain.shape[0],), device=brain.device)
        predicted = model.predict_frames(brain, batch["stimuli"], subjects, phase)
        frames = k - 1 + phase[:, None] + s * torch.arange(1, predicted.shape[1] + 1, device=brain.device)
        target = brain.gather(1, frames[..., None].expand(-1, -1, brain.shape[-1]))
    else:
        predicted, target = model.predict_frames(brain, batch["stimuli"], subjects), brain[:, k:]
    prediction_loss = (predicted - target).square().mean()
    loss = prediction_loss
    rollout_loss = prediction_loss.new_zeros(())
    steps = train_cfg["rollout_steps"]
    if steps:
        # With a state step s, the rollout covers rollout_steps steps of each of the s chains (s * rollout_steps
        # frames) from s * context_states observed states.
        prefix, steps = model.context_states * s, steps * s
        if brain.shape[1] - k + 1 < prefix + steps:
            raise ValueError("sequence_states must cover context_states + rollout_steps")
        stimuli = {name: value[:, :prefix + steps - 1] for name, value in batch["stimuli"].items()}
        rolled = model.rollout_frames(brain[:, :k + prefix - 1], stimuli, subjects, horizon_trs=steps)
        rollout_loss = (rolled - brain[:, k + prefix - 1:k + prefix - 1 + steps]).square().mean()
        loss = loss + train_cfg["rollout_weight"] * rollout_loss
    zero = prediction_loss.new_zeros(())
    return loss, {"prediction_loss": prediction_loss.detach(), "rollout_loss": rollout_loss.detach(),
                  "sigreg": zero, "latent_spread": zero}


class LatentObjective(torch.nn.Module):
    """Expose the whole differentiable objective as one DDP forward call."""

    def __init__(self, model, regularizer, train_cfg):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.train_cfg = train_cfg
        self.epoch = 0  # set by the training loop; train.rollout_schedule reads it
        self.teacher = load_teacher(train_cfg["teacher_run"]) if train_cfg.get("teacher_run") else None
        if self.teacher is not None:
            mine, theirs = model.encoder.state_dict(), self.teacher.encoder.state_dict()
            if mine.keys() != theirs.keys() or any(not torch.equal(mine[k].cpu(), theirs[k].cpu()) for k in mine):
                raise ValueError("train.teacher_run must have the student's encoder: its states are the student's targets")

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "teacher", None) is not None:
            self.teacher.eval()  # no dropout in the teacher's targets
        return self

    def forward(self, batch):
        if self.train_cfg["objective"] == "fmri":
            return fmri_loss(self.model, batch, self.train_cfg)
        return latent_loss(self.model, batch, self.regularizer, self.train_cfg, self.epoch, self.teacher)
