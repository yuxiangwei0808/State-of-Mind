"""Where the code reads data and writes results: three roots, each set by an environment variable.

    NEUROWORLD_DATA      prepared datasets and assets: <dataset>/derivatives/neuroworld-schaefer1000-v2* (the fMRI
                         bundles, TRIBE v2 feature rows and predictions), atlases/, templateflow/, the TRIBE v2
                         checkpoint, and the cached reference fits (neuralworld-checkpoints/baselines)
    NEUROWORLD_OUTPUT    what the project produced: checkpoints/<dataset>/<run root>/<run>/, the benchmark result stores
                         (benchmark-final, benchmark-test, benchmark-iter-overflow, benchmark-ablations,
                         benchmark-history-context), paper-analysis/, and the released model spectator-v1/
    NEUROWORLD_EXTERNAL  other baselines' outputs, read only: BrainVista's and MIRAGE's benchmark stores and saved
                         forecasts (baselines/...) and the original NeuroWorld k1 run (NeuralWorld/checkpoints/...)

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
