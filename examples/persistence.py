"""An external model, scored through the benchmark's public interface only (docs/benchmark.md#adding-a-model).

A model is any class with a ``name`` and ``predict(QueryBatch) -> [B, horizon, 1000]``, built as
``Class(dataset=..., device=...)``. This one forecasts every parcel by its last measured value, relaxing toward the
run's average (zero in the benchmark's z-scored units) with a fixed time constant. It reads only the query's fMRI
history and its validity mask: frames before the run start and frames Updating hides are marked invalid.

    python -m neuroworld.benchmark suite --dataset algonauts --model py:examples/persistence.py:DecayingPersistence
"""
import numpy as np

from neuroworld.benchmark.interface import last_valid


class DecayingPersistence:
    """The last valid frame, decaying toward zero with time constant tau_frames."""

    def __init__(self, dataset=None, device="cpu", tau_frames=8.0):
        self.tau = float(tau_frames)
        self.name = f"decaying persistence (tau {self.tau:g} frames)"

    def predict(self, batch):
        last = last_valid(batch)                                          # [B, 1000]
        decay = np.exp(-np.arange(1, batch.horizon + 1) / self.tau)       # [horizon]
        return (last[:, None, :] * decay[None, :, None]).astype(np.float32)
