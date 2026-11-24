"""Dataset-organized stimulus preparation APIs.

``common`` and ``timing`` own event identity, validation, stitching, and
native-clock pooling; ``raw_cache`` owns the shared raw 2-Hz reader/exporter;
``materialize`` is its command-line wrapper. The two dataset modules own
clock, modality, count, and text policy.
"""

from __future__ import annotations

from . import algonauts, common, hcp_movies, raw_cache
from .common import *  # noqa: F401,F403 - public package surface
from .raw_cache import *  # noqa: F401,F403 - public package surface


__all__ = [
    "algonauts",
    "common",
    "hcp_movies",
    "raw_cache",
]
__all__ += [name for name in common.__all__ if name not in __all__]
__all__ += [name for name in raw_cache.__all__ if name not in __all__]
