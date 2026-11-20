"""Causal next-state prediction over at most context_states states, built from LeWM blocks."""
import torch
from torch import nn
from neuroworld.models.lewm_blocks import ConditionalBlock, banded_causal_mask

CONDITION_DIM = 1152  # projected TRIBEv2 text, audio, video


class Predictor(nn.Module):
    def __init__(self, state_dim, context_states=3, depth=6, heads=16, dropout=0.1,
                 mlp_dim=None, condition_dim=CONDITION_DIM, output_norm=True, condition_norm=True,
                 history_blocks=0, block_states=0, conditioning="adaln", history_spaced=False, chunk_frames=0,
                 state_tokens=1, own_drive=False, movie_rank=0, movie_tokens=0, token_positions=None, width=0):
        super().__init__()
        if type(context_states) is not int or context_states < 1:
            raise ValueError("context_states must be a positive integer")
        if history_blocks and block_states < 1:
            raise ValueError("History blocks need at least one state each")
        if conditioning not in ("adaln", "input", "output"):
            raise ValueError(f"Unknown conditioning {conditioning!r}")
        if state_dim % state_tokens:
            raise ValueError("state_dim must split evenly into state_tokens")
        if movie_tokens and (condition_dim % movie_tokens or conditioning != "input" or context_states != 1 or history_blocks
                             or chunk_frames or own_drive or movie_rank):
            raise ValueError("movie_tokens splits concatenated rows, with input conditioning and one context state")
        self.context_states = context_states
        # state_tokens n > 1: a state is n latents side by side, read as n tokens that attend to each other, each
        # predicted from itself. The Transformer then has state_dim / n as its width. state_dim below is that width.
        self.state_tokens, state_dim = state_tokens, state_dim // state_tokens
        self.token_dim = state_dim
        # width w > 0: the blocks run at width w, between a projection in from each token and the output layer back to it
        # (input conditioning, one state per prediction). The tokens, their embeddings and the movie drive keep the token's width.
        if width and (conditioning != "input" or history_blocks or chunk_frames or movie_tokens):
            raise ValueError("width applies to input conditioning on one state, with no history blocks, chunks or movie tokens")
        inner = width or state_dim
        self.inward = nn.Linear(state_dim, inner) if inner != state_dim else nn.Identity()
        # own_drive: each state token has its own projection of the movie row, instead of all sharing one.
        self.own_drive = own_drive
        # Optional longer history: the history_blocks * block_states states before the newest
        # context_states enter as one mean per block, each block marked by a learned embedding. With history_spaced,
        # a block enters as its first state instead, so the tokens are single states block_states apart.
        self.history_blocks, self.block_states, self.history_spaced = history_blocks, block_states, history_spaced
        self.read_states = context_states + history_blocks * block_states
        # Optional chunked query: chunk_frames learned query tokens that return the next chunk_frames states at once
        # (predict_chunks).
        self.chunk_frames = chunk_frames
        # 1152 per stimulus row (2048 with tribev2_context); N rows side by side with concat.
        self.condition_dim = condition_dim
        # Frozen projections of raw features vary widely in scale (video outliers reach ~500);
        # normalize the concatenated row as TRIBEv2's pre-norm Transformer does. condition_norm=False
        # leaves the rows as given (for rows already standardized per feature).
        self.condition_norm = nn.LayerNorm(condition_dim) if condition_norm else nn.Identity()
        drive_width = state_dim * state_tokens if own_drive else state_dim
        # movie_rank r > 0: the row reaches the state through r dimensions, the additive projection's last layer starting at zero.
        self.condition = nn.Linear(condition_dim, drive_width) if not movie_rank else nn.Sequential(
            nn.Linear(condition_dim, movie_rank, bias=False), nn.Linear(movie_rank, drive_width))
        # movie_tokens n > 0: the n concatenated movie rows enter as n tokens beside the state, one projection for all of them, each
        # marked by a learned lag embedding (predict_movie_tokens). The state's output still starts at zero, so this too starts as the identity.
        self.movie_tokens = movie_tokens
        if movie_tokens:
            self.condition = nn.Linear(condition_dim // movie_tokens, state_dim)
            self.lag_embedding = nn.Parameter(torch.zeros(movie_tokens, state_dim))
        # How the projected movie row enters. adaln: it modulates every block (shift, scale and gate of the
        # normalized state stream, a state x movie interaction). input: it is added to its state's token, and the
        # blocks' modulation reads the learned constant below. output: the blocks read the states alone, and the
        # row is added to the prediction, x' = x + f(states) + W u. The additive projections start at zero, so
        # every form starts as the identity, as AdaLN-zero does.
        self.conditioning = conditioning
        if conditioning != "adaln" and not movie_tokens:
            last = self.condition[-1] if movie_rank else self.condition
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # Brain-only training, and the additive conditionings, condition every block on one learned constant.
        self.null_condition = nn.Parameter(torch.zeros(inner))
        # One learned logit offset per history distance (in tokens: blocks, then single states, then queries);
        # irrelevant when context_states=1 and there are no blocks or queries.
        self.relative_bias = nn.Parameter(torch.zeros(context_states + history_blocks + chunk_frames))
        if history_blocks:
            self.block_embedding = nn.Parameter(torch.zeros(history_blocks, state_dim))
        if chunk_frames:
            self.query_embedding = nn.Parameter(torch.zeros(chunk_frames, state_dim))
        if state_tokens > 1:
            self.token_embedding = nn.Parameter(torch.zeros(state_tokens, state_dim))
        # token_positions [state_tokens, k]: fixed coordinates of the tokens (such as functional gradients), each token's embedding a
        # learned linear map of its coordinates, in place of a free embedding per token.
        self.register_buffer("token_positions", None if token_positions is None else torch.as_tensor(token_positions, dtype=torch.float32))
        if token_positions is not None:
            del self.token_embedding
            self.token_position = nn.Linear(self.token_positions.shape[-1], state_dim, bias=False)
        self.blocks = nn.ModuleList([
            ConditionalBlock(inner, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        # The final LayerNorm bounds the increment whatever the state's amplitude. Without it
        # (output_norm=False) the output layer reads the raw residual stream, so the increment
        # can grow with the state and the predictor contains an affine transition of the
        # latest state as a special case.
        self.norm = nn.LayerNorm(inner) if output_norm else nn.Identity()
        self.out = nn.Linear(inner, state_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, states, conditions=None, valid=None, positions=None):
        """Teacher-forced next-state prediction at every position.

        states: [B, T, D]; conditions: [B, T, condition_dim] stimulus conditions or None.
        Row t conditions the transition from state t to t+1. Returns [B, T, D].
        Each prediction uses at most context_states raw input states across all
        blocks. Early positions use the shorter available history.

        With history blocks, each prediction reads the read_states states up to its position,
        and only the states flagged in valid ([B, T], default all) enter; positions (default all)
        chooses the predictions returned, [B, len(positions), D].
        """
        if states.ndim != 3:
            raise ValueError("states must have shape [batch, time, state_dim]")
        batch, length, width = states.shape
        if conditions is not None and (conditions.shape[:2] != (batch, length) or conditions.shape[-1] != self.condition_dim):
            raise ValueError(f"conditions must supply one {self.condition_dim}-wide row per state")
        if self.movie_tokens:
            return self._predict_movie_tokens(states, conditions)
        condition, drive = self._project(conditions, batch, length)
        if self.history_blocks:
            if valid is None:
                valid = torch.ones(batch, length, dtype=torch.bool, device=states.device)
            return self._predict_history(states, condition, valid,
                                         range(length) if positions is None else positions, drive)
        if length <= self.context_states:
            return self._predict_window(states, condition, drive)

        # Process each window of context_states independently. A band mask on a longer
        # sequence would let stacked blocks indirectly read more than context_states states.
        count = length - self.context_states + 1
        windows = lambda x: x.unfold(1, self.context_states, 1).transpose(-1, -2).reshape(
            batch * count, self.context_states, x.shape[-1])
        predictions = self._predict_window(
            windows(states), windows(condition), None if drive is None else windows(drive),
        ).reshape(batch, count, self.context_states, width)
        # The first window also supplies predictions with fewer than context_states inputs.
        return torch.cat((predictions[:, 0, :-1], predictions[:, :, -1]), dim=1)

    def _predict_history(self, states, condition, valid, positions, drive=None):
        """[B, P, D] predictions at the given positions, each from its read_states states: one token per
        block mean (over its valid states; a block with none is left out), then one per newest state."""
        reach, edge = self.read_states, self.history_blocks * self.block_states
        index = torch.as_tensor(list(positions), device=states.device)[:, None] + torch.arange(reach, device=states.device)
        pad = lambda x: torch.cat((x.new_zeros(x.shape[0], reach - 1, *x.shape[2:]), x), 1)[:, index].flatten(0, 1)
        x, c, v = pad(states), pad(condition), pad(valid)  # [B * P, reach, ...]: older states first
        weight = v[:, :edge].unflatten(1, (self.history_blocks, self.block_states)).to(x.dtype)
        if self.history_spaced:
            weight = weight * (torch.arange(self.block_states, device=x.device) == 0)
        count = weight.sum(-1)
        mean = lambda y: (y[:, :edge].unflatten(1, (self.history_blocks, self.block_states)) * weight[..., None]
                          ).sum(2) / count.clamp(min=1)[..., None]
        tokens = torch.cat((mean(x) + self.block_embedding, x[:, edge:]), 1)
        condition = torch.cat((mean(c), c[:, edge:]), 1)
        if drive is not None:
            d = pad(drive)
            drive = torch.cat((mean(d), d[:, edge:]), 1)
        present = torch.cat((count > 0, v[:, edge:]), 1)
        length = tokens.shape[1]
        mask = banded_causal_mask(length, length, self.relative_bias, device=x.device, dtype=x.dtype)
        # A missing token is read by no other token (it still reads itself, so no row is empty).
        hidden = ~present[:, None, :] & ~torch.eye(length, dtype=torch.bool, device=x.device)
        mask = mask.masked_fill(hidden, float("-inf"))[:, None]
        h = self._read(tokens, drive)
        for block in self.blocks:
            h = block(h, condition, attn_mask=mask)
        out = self._drive(tokens[:, -1] + self.out(self.norm(h[:, -1])), None if drive is None else drive[:, -1])
        return out.unflatten(0, (states.shape[0], -1))

    def _predict_window(self, states, condition, drive=None):
        if self.state_tokens > 1:
            return self._predict_tokens(states, condition, drive)
        mask = banded_causal_mask(states.shape[1], self.context_states, self.relative_bias,
                                  device=states.device, dtype=states.dtype)
        x = self._read(states, drive)
        for block in self.blocks:
            x = block(x, condition, attn_mask=mask)
        return self._drive(states + self.out(self.norm(x)), drive)

    def _predict_tokens(self, states, condition, drive=None):
        """[N, 1, D] the next state from [N, 1, D] states of state_tokens latents side by side. Each latent is a token with
        its embedding and the state's row, and the tokens attend to each other; each prediction is its token's latent plus
        its output."""
        x = states.squeeze(1).unflatten(-1, (self.state_tokens, self.token_dim))
        if drive is not None:
            drive = drive.squeeze(1).unflatten(-1, (self.state_tokens, self.token_dim)) if self.own_drive else drive[:, :1]
        h = self._read(x + self._state_embedding(), drive)
        condition = condition.expand(-1, self.state_tokens, -1)
        everyone = h.new_zeros(self.state_tokens, self.state_tokens)  # no mask would mean causal attention
        for block in self.blocks:
            h = block(h, condition, attn_mask=everyone)
        return self._drive(x + self.out(self.norm(h)), drive).flatten(1)[:, None]

    def _state_embedding(self):
        return self.token_embedding if self.token_positions is None else self.token_position(self.token_positions)

    def _predict_movie_tokens(self, states, conditions):
        """[B, T, D] the next state from [B, T, D] states, each read with the movie rows of its transition as separate tokens, and all
        tokens attending to each other. The rows are normalized together, as in the concatenated form. Each prediction is the
        state plus its state tokens' outputs."""
        if conditions is None:
            raise ValueError("Movie tokens need conditions")
        batch, length = states.shape[:2]
        rows = self.condition_norm(conditions).flatten(0, 1).unflatten(-1, (self.movie_tokens, -1))
        x = states.flatten(0, 1).unflatten(-1, (self.state_tokens, self.token_dim))
        identity = self._state_embedding() if self.state_tokens > 1 else 0
        h = torch.cat((x + identity, self.condition(rows) + self.lag_embedding), 1)
        count = h.shape[1]
        condition = self.null_condition.expand(h.shape[0], count, -1)
        everyone = h.new_zeros(count, count)  # no mask would mean causal attention
        for block in self.blocks:
            h = block(h, condition, attn_mask=everyone)
        out = x + self.out(self.norm(h[:, :self.state_tokens]))
        return out.flatten(1).unflatten(0, (batch, length))

    def predict_chunks(self, states, conditions, valid, anchors, count):
        """[B, A, count, D] the next count states after each of the A anchor positions, returned at once.

        states: [B, T, D]; conditions: [B, R, condition_dim], row r conditioning the transition from state r to r + 1,
        through the last query's frame; valid: [B, T] (None: all). An anchor reads the read_states states up to it, as
        tokens: the newest context_states and, with spaced history, the single states block_states apart before them;
        only the valid ones enter. Each query token is the newest state plus its query's embedding. Tokens carry their
        row, as one-step predictions do, so the j-th query has the row of the transition to its frame. Attention is
        causal: a query reads the context and the earlier queries, so a shorter chunk is a longer one's first frames.
        Each prediction is the newest state plus its query's output.
        """
        batch, device = states.shape[0], states.device
        reach, edge = self.read_states, self.history_blocks * self.block_states
        first = torch.as_tensor(list(anchors), device=device)[:, None] - reach + 1
        if first.min() < 0:
            raise ValueError(f"A chunk needs {reach} states up to its anchor")
        spaced = torch.arange(0, edge, max(self.block_states, 1), device=device)
        read = first + torch.cat((spaced, torch.arange(edge, reach, device=device)))  # [A, context tokens]
        x = states[:, read].flatten(0, 1)
        newest = x[:, -1:]
        tokens = [newest + self.query_embedding[:count]]
        if self.history_blocks:
            tokens = [x[:, :self.history_blocks] + self.block_embedding, x[:, self.history_blocks:], *tokens]
        else:
            tokens = [x, *tokens]
        tokens = torch.cat(tokens, 1)
        length = tokens.shape[1]
        rows = None
        if conditions is not None:
            rows = conditions[:, torch.cat((read, first + reach - 1 + torch.arange(count, device=device)), 1)].flatten(0, 1)
        condition, drive = self._project(rows, tokens.shape[0], length)
        present = tokens.new_ones((tokens.shape[0], length), dtype=torch.bool)
        if valid is not None:
            present[:, :read.shape[1]] = valid[:, read].flatten(0, 1)
        mask = banded_causal_mask(length, len(self.relative_bias), self.relative_bias, device=device, dtype=tokens.dtype)
        # A missing token is read by no other token (it still reads itself, so no row is empty).
        hidden = ~present[:, None, :] & ~torch.eye(length, dtype=torch.bool, device=device)
        mask = mask.masked_fill(hidden, float("-inf"))[:, None]
        h = self._read(tokens, drive)
        for block in self.blocks:
            h = block(h, condition, attn_mask=mask)
        out = self._drive(newest + self.out(self.norm(h[:, -count:])), None if drive is None else drive[:, -count:])
        return out.unflatten(0, (batch, first.shape[0]))

    def _project(self, conditions, batch, length):
        """The blocks' modulation [B, T, D] and, for an additive conditioning, the projected rows (else None)."""
        if conditions is None:
            return self.null_condition.expand(batch, length, -1), None
        condition = self.condition(self.condition_norm(conditions))
        if self.conditioning != "adaln":
            return self.null_condition.expand(batch, length, -1), condition
        return condition, None

    def _read(self, tokens, drive):
        """The tokens the blocks read: with input conditioning, each plus its projected row, at the blocks' width."""
        return self.inward(tokens + drive if drive is not None and self.conditioning == "input" else tokens)

    def _drive(self, prediction, drive):
        """With output conditioning, the prediction plus its transition's projected row."""
        return prediction + drive if drive is not None and self.conditioning == "output" else prediction


class MLPPredictor(nn.Module):
    """A residual MLP in the Predictor's place, reading one state with no attention and no normalization of the state:
    x' = x + L x + W u + f(x, u), where u is the normalized movie row. L, W and f's last layer start at zero, so it starts
    as the identity. Depth 0 leaves a trained matrix, x' = x + L x + W u + c, as a control."""

    def __init__(self, state_dim, condition_dim, hidden_dim, depth, dropout, condition_norm=True, state_tokens=1):
        super().__init__()
        self.context_states, self.condition_dim, self.state_tokens, self.depth = 1, condition_dim, state_tokens, depth
        self.history_blocks = 0
        self.condition_norm = nn.LayerNorm(condition_dim) if condition_norm else nn.Identity()
        self.condition = nn.Linear(condition_dim, state_dim)
        self.linear = nn.Linear(state_dim, state_dim, bias=False)
        for layer in (self.condition, self.linear):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.condition.bias)
        if depth:
            self.embed = nn.Linear(state_dim + condition_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(depth)])
            self.out = nn.Linear(hidden_dim, state_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, states, conditions=None, valid=None, positions=None):
        """[B, T, D] next states from [B, T, D] states and [B, T, condition_dim] rows; row t conditions the step from state t to t + 1."""
        row = self.condition_norm(conditions)
        prediction = states + self.linear(states) + self.condition(row)
        if self.depth:
            h = self.embed(torch.cat((states, row), dim=-1))
            for block in self.blocks:
                h = h + block(h)
            prediction = prediction + self.out(h)
        return prediction


class CorrectedMatrix(MLPPredictor):
    """A matrix held fixed, with a Transformer's correction on top: x' = x + L x + W u + c + T(x, u). L, W and c come from a closed-form fit
    (train.predictor_checkpoint), and T's output starts at zero, so the model starts as the matrix and T learns only what it leaves."""

    def __init__(self, correction, state_dim, condition_dim, condition_norm=True, state_tokens=1):
        super().__init__(state_dim, condition_dim, 0, 0, 0.0, condition_norm, state_tokens)
        self.requires_grad_(False)
        self.correction = correction

    def forward(self, states, conditions=None, valid=None, positions=None):
        return super().forward(states, conditions) + self.correction(states, conditions) - states
