"""Where the code reads data and writes results: roots set by environment variables.

    NEUROWORLD_DATA      prepared datasets and assets: <dataset>/derivatives/neuroworld-schaefer1000-v2* (the fMRI
                         bundles, TRIBE v2 feature rows and predictions), atlases/, the TRIBE v2 checkpoint, and
                         cached baseline fits (neuralworld-checkpoints/baselines)
    NEUROWORLD_OUTPUT    released weights and outputs: spectator-v1/, benchmark-results/, paper-analysis/, and
                         training runs (checkpoints/)
    NEUROWORLD_EXTERNAL  optional read-only root for external baseline outputs

Unset, each defaults to a folder (or a symbolic link) in the repository root: data/, outputs/ and external/. Paths are
resolved, so a link and its target give the same strings (some cached fits record the paths they were made from).
YAML configs refer to the roots as ${NEUROWORLD_DATA}, ${NEUROWORLD_OUTPUT} and ${NEUROWORLD_EXTERNAL}.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _root(variable, default):
    return Path(os.environ.get(variable) or REPO / default).expanduser().resolve()


DATA = _root("NEUROWORLD_DATA", "data")
OUTPUT = _root("NEUROWORLD_OUTPUT", "outputs")
EXTERNAL = _root("NEUROWORLD_EXTERNAL", "external")
ROOTS = {"NEUROWORLD_DATA": DATA, "NEUROWORLD_OUTPUT": OUTPUT, "NEUROWORLD_EXTERNAL": EXTERNAL}


def expand(value):
    """value with ${NEUROWORLD_DATA}, ${NEUROWORLD_OUTPUT} and ${NEUROWORLD_EXTERNAL} replaced by the roots, in every
    string of a nested dict or list."""
    if isinstance(value, str):
        for name, root in ROOTS.items():
            value = value.replace("${" + name + "}", str(root))
        return value
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value
