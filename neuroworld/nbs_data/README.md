# Offline Schaefer-1000 preparation

This directory contains the one-off offline preparation code for Algonauts
and HCP Movies. The fMRI command has exactly two operations:

```text
prepare   read-only dry-run of the complete input inventory
run       write or resume prepared data directly at --output
```

Use `run --limit N` for a small manual smoke test. There is no build stage,
validation command, or publish step. A full run writes to its final location
from the start.

## Code layout

```text
neuroworld/nbs_data/
├── fmri/
│   ├── {_sources,_adapters,common,atlas,hcp_registry}.py
│   └── {algonauts,hcp_movies}.py
├── stimuli/
│   ├── {common,timing,raw_cache,config,materialize}.py
│   └── {algonauts,hcp_movies}.py
├── bundle.py
├── pipeline.py
└── schaefer.py
```

`fmri/` owns source discovery and fMRI preprocessing. `stimuli/` owns the
independent raw 2-Hz stimulus export and native-TR timing policy. `pipeline.py`
joins one fMRI run with its stimulus timeline; `schaefer.py` is the two-command
entry point.

## Data contract

Every completed bundle declares:

```text
training_semantics=offline_tmi_encoding
prospective_ready=false
hashes_computed=false
```

Brain arrays are finite float32 `[T,1000]`, masks are bool `[T]`, and stimulus
arrays are finite float32 `[T,2,D]`. The pipeline applies no feature
normalization and no baked haemodynamic lag. Whole-run detrending and temporal
z-scoring make the HCP outputs future-dependent, so these bundles do not
support causal or prospective claims.

The fMRI recipes are:

- **Algonauts:** read the official Schaefer-1000 HDF5 arrays unchanged at TR
  1.49 s. Do not smooth, regress, z-score, or project them again.
- **HCP Movies:** use corrected S1200 `MNINonLinear` movie volumes at TR 1.0 s and
  official brain masks, with TRIBEv2's recipe (d'Ascoli et al. 2026, sections
  5.7–5.8): Schaefer-1000 means of the unsmoothed in-mask voxels, then per run a
  z-score and a linear detrend on valid rows. HCP motion gives FD, and the volume
  gives standardized DVARS; they only flag invalid frames. No smoothing, nuisance,
  global-signal or high-pass regression: the first bundle applied 5-mm smoothing
  and nuisance plus 0.009-Hz DCT regression, which removed movie-driven signal
  (TRIBEv2's group correlation on MOVIE2 fell from 0.38 to 0.29). Do not repeat
  motion/distortion correction. HCP rest is out of scope.

## Inputs

```bash
PYTHON=python
DATA="$NEUROWORLD_DATA"   # the data root of neuroworld/paths.py
ATLAS="$DATA/atlases/neuroworld-schaefer1000-v2/manifest.json"

ALG_SOURCE="$DATA/algonauts/dataset"
ALG_RAW="$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-preparation/raw/chunks.jsonl"
ALG_OUT="$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2"

HCP_SOURCE="$DATA/hcp_movies/dataset"
HCP_RAW="$DATA/hcp_movies/derivatives/neuroworld-schaefer1000-v2-preparation/raw/chunks.jsonl"
HCP_OUT="$DATA/hcp_movies/derivatives/neuroworld-schaefer1000-v2"
```

The atlas is an existing input asset. Algonauts does not use it because its
official arrays are already Schaefer-1000. HCP requires it, and also needs a
run-local `brainmask_fs.1.60.nii.gz` for every run; source discovery
still checks the subject-level `MNINonLinear/ROIs/wmparc.1.60.nii.gz`, which
processing no longer uses.

The two `*_RAW` indexes come from the independent
`neuroworld.nbs_data.stimuli.materialize --export-raw-only` command. Run that
first only when the corresponding `chunks.jsonl` does not exist. It exports
pre-projector TRIBEv2 arrays at 2 Hz; `schaefer run` performs the sole
2-Hz-to-native-TR pooling pass.

## 0. Generate the Algonauts stimulus index only if absent

HCP already has its raw index. Check `$ALG_RAW` before
continuing; if `chunks.jsonl` already exists, leave it unchanged and skip this
stage. Otherwise, generate it with:

```bash
  TRIBE_CONFIG="$DATA/hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/config.yaml"
  ALG_CACHE="$DATA/algonauts/tribev2_stimuli/algonauts"
  ALG_RAW_DIR="$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-preparation/raw"
  ALG_EVENTS=(
    "$DATA/algonauts/tribev2_stimuli/neuroworld-v1/full/algonauts_events_with_context_shard00-of-04.tsv"
    "$DATA/algonauts/tribev2_stimuli/neuroworld-v1/full/algonauts_events_with_context_shard01-of-04.tsv"
    "$DATA/algonauts/tribev2_stimuli/neuroworld-v1/full/algonauts_events_with_context_shard02-of-04.tsv"
    "$DATA/algonauts/tribev2_stimuli/neuroworld-v1/full/algonauts_events_with_context_shard03-of-04.tsv"
  )

  "$PYTHON" -m neuroworld.nbs_data.stimuli.materialize \
    --export-raw-only \
    --dataset algonauts \
    --source-root "$ALG_SOURCE" \
    --extractor neuroworld.nbs_data.stimuli.raw_cache:iter_tribev2_raw_chunks \
    --extractor-kwargs \
      "{\"events_path\":[\"${ALG_EVENTS[0]}\",\"${ALG_EVENTS[1]}\",\"${ALG_EVENTS[2]}\",\"${ALG_EVENTS[3]}\"],\"cache_dir\":\"$ALG_CACHE\",\"config_path\":\"$TRIBE_CONFIG\"}" \
    --generated-index-dir "$ALG_RAW_DIR" \
    --workers 4
```

The expected result is `$ALG_RAW_DIR/chunks.jsonl`, which is the path already
stored in `$ALG_RAW`. `--workers` partitions timelines across spawned
processes; each process keeps NeuralSet's internal loader at `num_workers=0`,
and only the parent publishes the final index after all partitions succeed. Set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to `1` for a multi-process run. If interrupted, rerun this
exact command; completed raw arrays are reused from the deterministic
`.raw.staging` sibling directory.

## 1. Prepare: full dry-run

`prepare` reads every expected source and the complete raw-feature inventory.
It writes nothing. A zero exit status and `"ready": true` mean the inputs are
ready for `run`.

```bash
"$PYTHON" -m neuroworld.nbs_data.schaefer prepare \
  --dataset algonauts --source-root "$ALG_SOURCE" --raw-index "$ALG_RAW"

"$PYTHON" -m neuroworld.nbs_data.schaefer prepare \
  --dataset hcp_movies --source-root "$HCP_SOURCE" \
  --atlas-manifest "$ATLAS" --raw-index "$HCP_RAW"
```

## 2. Optional manual smoke test

`--limit N` selects the first N canonical run IDs deterministically. Use a
separate output directory because a limited bundle is intentionally complete
for only those N runs.

```bash
"$PYTHON" -m neuroworld.nbs_data.schaefer run \
  --dataset algonauts --source-root "$ALG_SOURCE" --raw-index "$ALG_RAW" \
  --output "${ALG_OUT}-smoke" --limit 3

/usr/bin/time -v "$PYTHON" -m neuroworld.nbs_data.schaefer run \
  --dataset hcp_movies --source-root "$HCP_SOURCE" \
  --atlas-manifest "$ATLAS" --raw-index "$HCP_RAW" \
  --output "${HCP_OUT}-smoke" --limit 3
```

Inspect the smoke directory manually, or hand its path to Codex for a manual
audit. This is deliberately not another pipeline command.

## 3. Full run

After `prepare` succeeds and the smoke output looks correct, write directly to
the final roots. The worker counts below are the tested concurrent allocation
for `arctrdcn017.rs.gsu.edu` (32 CPUs, 754 GiB RAM):

```bash
"$PYTHON" -m neuroworld.nbs_data.schaefer run \
  --dataset algonauts --source-root "$ALG_SOURCE" --raw-index "$ALG_RAW" \
  --output "$ALG_OUT" --workers 1

"$PYTHON" -m neuroworld.nbs_data.schaefer run \
  --dataset hcp_movies --source-root "$HCP_SOURCE" \
  --atlas-manifest "$ATLAS" --raw-index "$HCP_RAW" \
  --output "$HCP_OUT" --workers 24 --eager-volumes
```

To stream-validate every array referenced by the Algonauts raw index, run:

```bash
python -m neuroworld.nbs_data.validate_raw --help
```

If interrupted, rerun the exact same command with the same `--output`. Feature
timelines and fMRI runs are committed atomically and already committed records
are checked and skipped. `bundle.json` is written last, so its presence marks
a completed output.

There is no GPU path. `--workers N` uses spawned processes only for independent
per-run fMRI computation; the parent remains the sole feature, run-record, and
bundle writer. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1` whenever `N > 1`.
`--eager-volumes` reads each compressed HCP float32 volume once and then
reuses it for every processing pass. On this
host, one eager HCP worker used about 7 GiB RSS in the smoke test and up to
about 12 GiB during the first full-run wave. Choose workers from both CPU and
memory limits rather than CPU count alone. Worker processes are recycled after
each run so eager-volume allocations and native-library caches cannot
accumulate across the full inventory.

The three-run HCP smoke output was reproduced with three workers and eager
loading in 9m17s versus 12h09m through the proxy-streaming path. All 21 stored
arrays were bit-for-bit identical and the full bundle audit passed.

Expected full inventories are 1,392 Algonauts runs (641,568 frames) and 725 HCP
runs sharing eight feature timelines.

The verified 2026-09-06 full run on `arctrdcn017` passed every reconciled audit.
It contains four Algonauts runs with no stimulus-supported frames and no
excluded HCP runs. Those four runs remain present in the canonical inventory
but contribute zero training samples through `valid_time`.

## Output layout

```text
neuroworld-schaefer1000-v2/
├── bundle.json
├── runs/{runs.jsonl,records/}
├── fmri/projected_native/
├── masks/{valid_time,nonsteady,motion_qc,stimulus_support}/
├── features/multimodal_native_tr/{features.jsonl,records/,arrays/}
├── qc/runs/
└── preparation/atlas_grids/
```

Training must apply the stored `valid_time` mask on every read. Invalid HCP
rows contain finite storage fillers and are not training samples.

## Contextual stimulus features

For `model.stimulus_features: tribev2_context`, TRIBEv2's frozen stimulus model
(`models/tribev2.py`: projections, 8-block temporal Transformer, low-rank head) runs once on
the raw 2-Hz cache with attention masked to past and present samples, and its output is pooled
to the TR grid with the bundle's rule. A modality missing from the cache (text, for HCP) counts
as missing, as in TRIBEv2. The run needs `neuraltrain` and `x_transformers`
(TRIBEv2's dependencies) and takes about 30 minutes on 12 CPU threads for Algonauts:

```bash
python -m neuroworld.nbs_data.stimuli.tribev2_context \
  --bundle $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2 \
  --checkpoint $DATA/hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/best.ckpt \
  --output $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-tribev2-context \
  --threads 12
```

Subjects who watched the same stimulus share one array. Before computing, it
checks that pooling the stitched raw cache reproduces every subject's stored
features, which confirms the alignment. The output holds `index.json` (format,
settings, checkpoint, and the array and row count for each timeline) and
`arrays/<stimulus>.npy` (float32, one 2048-value row per TR). Point
`data.context_features` at the output directory. Re-running skips finished arrays.
