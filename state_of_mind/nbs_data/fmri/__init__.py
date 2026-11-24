"""Native-clock fMRI preparation organized by shared and dataset code.

The numerical kernels and structural contracts are shared. Dataset modules
own discovery and the dataset-specific preparation recipes:

* :mod:`.algonauts` for the official Schaefer-1000 HDF5 arrays; and
* :mod:`.hcp_movies` for corrected HCP movie volumes.

Common numerical, source-discovery, and preparation symbols are re-exported
at this package boundary. Dataset modules are loaded lazily to avoid import
cycles and unnecessary optional dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

from .common import *  # noqa: F401,F403 - shared public numerical API
from .common import __all__ as _COMMON_ALL
from .common import POWER_FD_RADIUS_MM
from ._adapters import PreparedFmriRun, prepare_run
from ._sources import (
    EXPECTED_RUN_COUNTS,
    NativeRunSource,
    SourceDiscoveryError,
    audit_run_sources,
    discover_runs,
)


_DATASET_MODULES = frozenset({"algonauts", "hcp_movies"})


def __getattr__(name: str) -> Any:
    """Load a dataset module only when a caller asks for it."""

    if name in _DATASET_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DATASET_MODULES))


__all__ = [
    *_COMMON_ALL,
    "EXPECTED_RUN_COUNTS",
    "NativeRunSource",
    "POWER_FD_RADIUS_MM",
    "PreparedFmriRun",
    "SourceDiscoveryError",
    "audit_run_sources",
    "discover_runs",
    "prepare_run",
    *_DATASET_MODULES,
]
