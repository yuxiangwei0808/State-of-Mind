"""Training objectives: Spectator's JEPA latent rollout objective and stage-0 fMRI forecasting."""
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
    """[..., D] error per latent dimension: squared, or absolute with train.latent_error='l1'."""
    if train_cfg.get("latent_error", "squared") == "l1":
        error = train_cfg.get("l1_scale", 1.0) * (predicted - target).abs()
    else:
        error = (predicted - target).square()
    weight = train_cfg.get("slow_loss_weight", 1.0)
    if weight != 1.0:
        half = error.shape[-1] // 2
        error = torch.cat((error[..., :half], weight * error[..., half:]), dim=-1)
    return error


def latent_loss(model, batch, train_cfg, epoch=0):
    states = model.encode(batch["brain"])  # [B, S, D]
    targets = model.target_states(batch["brain"]) if hasattr(model, "target_states") else None
    conditions = model.encode_stimulus(batch["stimuli"])
    step = getattr(model, "state_step", 1)
    if step > 1:
        phase = torch.randint(step, (states.shape[0],), device=states.device)
        states, conditions = model.chain(states, conditions, phase)
        if targets is not None:
            targets, _ = model.chain(targets, None, phase)
    targets = states if targets is None else targets
    if states.shape[1] < 2:
        raise ValueError("Training windows need at least two states")
    predictions = model.predict(states[:, :-1], conditions)
    prediction_loss = latent_error(predictions, targets[:, targets.shape[1] - predictions.shape[1]:], train_cfg).mean()
    loss = train_cfg.get("prediction_weight", 1.0) * prediction_loss
    rollout_loss = states.new_zeros(())
    steps = rollout_steps_at(train_cfg, epoch)
    if steps:
        prefix = getattr(model, "read_states", model.context_states)
        if states.shape[1] < prefix + steps:
            raise ValueError("sequence_states must cover the states the predictor reads + rollout_steps")
        rows = None if conditions is None else conditions[:, :prefix + steps - 1]
        detach = train_cfg.get("rollout_detach_every", 0)
        extra = {"detach_every": detach} if detach else {}
        rolled = model.rollout_states(states[:, :prefix], rows, horizon=steps, **extra)
        error = latent_error(rolled, targets[:, prefix:prefix + steps], train_cfg)
        kind = train_cfg.get("rollout_step_weights", "uniform")
        if kind == "uniform":
            rollout_loss = error.mean()
        else:
            weights = rollout_step_weights(kind, steps, error.device)
            rollout_loss = (weights * error.mean(dim=(0, 2))).sum() / weights.sum()
        loss = loss + train_cfg["rollout_weight"] * rollout_loss
    spread = states.flatten(0, 1).std(dim=0, unbiased=False).mean()
    return loss, {"prediction_loss": prediction_loss.detach(), "rollout_loss": rollout_loss.detach(),
                  "latent_spread": spread.detach()}


def fmri_loss(model, batch, train_cfg):
    """Next-frame fMRI MSE for stage-0 forecaster training, plus the optional rollout term."""
    brain = batch["brain"]
    k, s = model.encoder.state_window_trs, getattr(model, "state_step", 1)
    subjects = model.readout.subject_indices(batch["subject_id"], brain.device)
    if s > 1:
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
        prefix, steps = model.context_states * s, steps * s
        if brain.shape[1] - k + 1 < prefix + steps:
            raise ValueError("sequence_states must cover context_states + rollout_steps")
        stimuli = {name: value[:, :prefix + steps - 1] for name, value in batch["stimuli"].items()}
        rolled = model.rollout_frames(brain[:, :k + prefix - 1], stimuli, subjects, horizon_trs=steps)
        rollout_loss = (rolled - brain[:, k + prefix - 1:k + prefix - 1 + steps]).square().mean()
        loss = loss + train_cfg["rollout_weight"] * rollout_loss
    zero = prediction_loss.new_zeros(())
    return loss, {"prediction_loss": prediction_loss.detach(), "rollout_loss": rollout_loss.detach(),
                  "latent_spread": zero}


class LatentObjective(torch.nn.Module):
    """Expose the differentiable objective as one DDP forward call."""

    def __init__(self, model, train_cfg):
        super().__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.epoch = 0

    def forward(self, batch):
        if self.train_cfg["objective"] == "fmri":
            return fmri_loss(self.model, batch, self.train_cfg)
        return latent_loss(self.model, batch, self.train_cfg, self.epoch)
