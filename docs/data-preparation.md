# Data preparation

The models and the benchmark read prepared bundles: Schaefer-1000 parcel time series on each dataset's own clock, the stimulus features on the same clock, and masks. The exact commands, flags, dry runs and resume behaviour are in the runbook [`neuroworld/nbs_data/README.md`](../neuroworld/nbs_data/README.md). Bundles are read by [`neuroworld/data/schaefer_v2_dataset.py`](../neuroworld/data/schaefer_v2_dataset.py) and the benchmark's [`data.py`](../neuroworld/benchmark/data.py); configs name them under `${NEUROWORLD_DATA}`.

## Sources and processing

| Dataset | Source | TR | Processing |
|---|---|---:|---|
| Algonauts 2025 (*Friends* S1–6, Movie10; 4 participants) | `$NEUROWORLD_DATA/algonauts/dataset` | 1.49 s | The official Schaefer-1000 arrays as released (CNeuroMod's preprocessing); no further cleaning |
| HCP movies (184 participants) | `$NEUROWORLD_DATA/hcp_movies/dataset` | 1.0 s | Corrected S1200 MNI movie volumes and the official brain mask; TRIBE v2's recipe: parcel mean of unsmoothed voxels, then per-run z-score and linear detrend on valid frames. Motion (FD, DVARS) only flags frames. Movie runs only |

The bundles are offline artifacts (`prospective_ready=false`): whole-run regression, detrending and z-scoring use every frame of a run, so no real-time forecasting claim is supported ([benchmark limits](benchmark.md#limits)).

**Stimulus features.** TRIBE v2's feature extractors (V-JEPA 2 video, w2v-BERT audio, Llama-3.2-3B text over timed transcripts) are exported as raw chunks at 2 Hz and pooled once onto the fMRI clock (`schaefer run`). Two derived sets are used:
- **projections:** TRIBE v2's frozen per-modality projections of the features;
- **context:** TRIBE v2's contextual rows (2,048 per frame, causal attention), which Spectator and the paper's linear forecaster and IPSID read, built by [`stimuli/tribev2_context.py`](../neuroworld/nbs_data/stimuli/tribev2_context.py) into `<dataset>/derivatives/neuroworld-schaefer1000-v2-tribev2-context`.

TRIBE v2's zero-shot predictions (the stimulus-only reference) are cached by [`eval/tribev2_reference.py`](../neuroworld/eval/tribev2_reference.py) into `<dataset>/derivatives/neuroworld-schaefer1000-v2-tribev2-predictions`.

## Workflow

Run from the repository root:

```bash
python -m neuroworld.nbs_data.schaefer --help
python -m neuroworld.nbs_data.stimuli.materialize --help
```

1. Use the two-template Schaefer-1000 atlas (MNI2009c and HCP/FSL-MNI label images with one ordered 1,000-row table).
2. If a raw stimulus index is absent, export it with `stimuli.materialize --export-raw-only`.
3. Run `schaefer prepare`, a read-only dry run over the full inventory.
4. Optionally run `schaefer run --limit N` into a separate directory and inspect it.
5. Run `schaefer run`, pointing `--output` at the dataset's derivatives folder. An interrupted run resumes with the same arguments.

The work is CPU- and storage-bound; there is no GPU path.

## Bundles

```text
$NEUROWORLD_DATA/algonauts/derivatives/neuroworld-schaefer1000-v2
$NEUROWORLD_DATA/hcp_movies/derivatives/neuroworld-schaefer1000-v2
```

Each holds `bundle.json`, `runs/{runs.jsonl,records/}`, `fmri/projected_native/`, `masks/{valid_time,nonsteady,motion_qc,stimulus_support}/`, `features/multimodal_native_tr/{features.jsonl,records/,arrays/}`, `qc/runs/` and `preparation/atlas_grids/`: 1,392 Algonauts runs and 725 HCP movie runs, plus eight HCP feature timelines. The loaders always apply the `valid_time` mask. Training and test splits are in each dataset's base config ([`configs/algonauts/base.yaml`](../neuroworld/configs/algonauts/base.yaml), [`configs/hcp_movies/base.yaml`](../neuroworld/configs/hcp_movies/base.yaml)).
