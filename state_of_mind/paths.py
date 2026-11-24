"""Where the code reads data and writes results: roots set by environment variables.

    STATE_OF_MIND_DATA      prepared datasets and assets: <dataset>/derivatives/neuroworld-schaefer1000-v2* (the fMRI
                            bundles and TRIBE v2 contextual feature rows), atlases/, and the TRIBE v2 checkpoint
    STATE_OF_MIND_OUTPUT    released weights and outputs: spectator-v1/, benchmark-results/, and training runs
                            (checkpoints/)

Unset, each falls back to the legacy NEUROWORLD_* variable if set, and otherwise defaults to a folder (or a symbolic
link) in the repository root: data/ and outputs/. Paths are resolved, so a link and its target give the same strings
(some cached fits record the paths they were made from).
YAML configs refer to the roots as ${STATE_OF_MIND_DATA} and ${STATE_OF_MIND_OUTPUT};
legacy ${NEUROWORLD_*} placeholders (as in the released Hugging Face configs) expand to the same roots.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _root(variable, legacy, default):
    return Path(os.environ.get(variable) or os.environ.get(legacy) or REPO / default).expanduser().resolve()


DATA = _root("STATE_OF_MIND_DATA", "NEUROWORLD_DATA", "data")
OUTPUT = _root("STATE_OF_MIND_OUTPUT", "NEUROWORLD_OUTPUT", "outputs")
ROOTS = {
    "STATE_OF_MIND_DATA": DATA,
    "STATE_OF_MIND_OUTPUT": OUTPUT,
    "NEUROWORLD_DATA": DATA,
    "NEUROWORLD_OUTPUT": OUTPUT,
}


def expand(value):
    """value with ${STATE_OF_MIND_*} (and legacy ${NEUROWORLD_*}) placeholders replaced by the roots, in every
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
