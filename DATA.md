# Data

The models and the benchmark read prepared bundles under `$NEUROWORLD_DATA` (default `data/`): Schaefer-1000 parcel time series on each dataset's own clock, the movie's stimulus features on the same clock, and masks of valid frames. No data are included in this repository or in the [Hugging Face release](https://huggingface.co/stanjsx/Spectator); the steps below download the datasets and prepare them as for the paper.

## Sources

| What | Where it comes from | Place at |
|---|---|---|
| **Algonauts 2025**: 4 participants (sub-01, 02, 03, 05) watching *Friends* S1–6 and the Movie10 films; Schaefer-1000 arrays, TR 1.49 s; the movies and transcripts | [courtois-neuromod/algonauts_2025.competitors](https://github.com/courtois-neuromod/algonauts_2025.competitors) (DataLad; CC0) | `algonauts/dataset/` |
| **HCP 7T movies**: 184 participants, the four movie runs; per run `<subject>/MNINonLinear/Results/tfMRI_MOVIE{1-4}_7T_{AP,PA}/` (the volume, `brainmask_fs.1.60.nii.gz`, `Movement_Regressors.txt`), per participant `MNINonLinear/ROIs/wmparc.1.60.nii.gz`; TR 1 s | the [HCP Young Adult](https://www.humanconnectome.org/study/hcp-young-adult) S1200 release on HCP's AWS bucket, under HCP's data use terms | `hcp_movies/dataset/` |
| **HCP movie stimuli**: the run movies in both released versions, the clip timing table and the lists of participants per version | HCP's 7T movie stimulus package (`movie_stimulus_2025.zip`) | `hcp_movies/dataset/movie_stimulus/` |
| **Schaefer-1000 atlas** (7 networks; for HCP): the FSL-MNI152 1 mm label image from [CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal), the MNI152NLin2009cAsym 2 mm one from [TemplateFlow](https://www.templateflow.org/), and a table of the 1,000 parcels in that order (`index`, `name`, `color`) | step 1 below | `atlases/neuroworld-schaefer1000-v2/` |
| **TRIBE v2**, snapshot `f894e783020944dcd96e5568550afe2aa9743f9f` | [facebook/tribev2](https://huggingface.co/facebook/tribev2) | `hub/` (the Hugging Face cache layout) |

The fMRI downloads ([`neuroworld/nbs_data/download.sh`](neuroworld/nbs_data/download.sh); `DRY_RUN=1` prints the commands):

```bash
neuroworld/nbs_data/download.sh algonauts    # needs DataLad
neuroworld/nbs_data/download.sh hcp          # needs the aws CLI with HCP's credentials, and the stimulus package unpacked first
```

**fMRI processing.** Algonauts is used as released (CNeuroMod's preprocessing), with no further cleaning. HCP follows TRIBE v2's recipe: the mean of the unsmoothed voxels in each parcel, then a per-run z-score and linear detrend on valid frames; head motion (FD, DVARS) only marks frames invalid. Both are preprocessed over whole runs, so the bundles support offline forecasting, not real-time claims.

**Stimulus features.** TRIBE v2's feature extractors (V-JEPA 2 for video, w2v-BERT for audio, Llama-3.2-3B for text over timed transcripts) run on the movies at 2 Hz and are pooled to the fMRI clock. Spectator, the linear forecaster and IPSID read TRIBE v2's **contextual rows**: its projections, Transformer and low-rank head run with attention to the past and present only, giving 2,048 values per frame. HCP's features use video and audio only, as in TRIBE v2.

## Preparation

Install [`neuroworld/requirements-data.txt`](neuroworld/requirements-data.txt) and ffmpeg, and run from the repository root with `DATA=$NEUROWORLD_DATA`. Every step resumes when rerun with the same arguments. Step 2 runs TRIBE v2's feature extractors and wants a GPU; the other steps run on CPUs.

1. **Atlas** (HCP only):
   ```bash
   python -c "from neuroworld.nbs_data.fmri.atlas import build_schaefer1000_atlas as build; build(
       mni_label_image='tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz',
       hcp_label_image='Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
       common_lut='Schaefer2018_1000Parcels_7Networks_order.tsv', output_dir='$DATA/atlases/neuroworld-schaefer1000-v2')"
   ```
2. **Event tables and raw stimulus features** at 2 Hz. [`stimuli/events.py`](neuroworld/nbs_data/stimuli/events.py) lists each recording's video, audio and words, as TRIBE v2 reads them; the export runs TRIBE v2's extractors over that table. For Algonauts:
   ```bash
   python -m neuroworld.nbs_data.stimuli.events algonauts --source-root $DATA/algonauts/dataset \
       --output $DATA/algonauts/tribev2_stimuli/events.tsv
   python -m neuroworld.nbs_data.stimuli.materialize --export-raw-only --dataset algonauts --source-root $DATA/algonauts/dataset \
       --extractor neuroworld.nbs_data.stimuli.raw_cache:iter_tribev2_raw_chunks \
       --extractor-kwargs "{\"events_path\": \"$DATA/algonauts/tribev2_stimuli/events.tsv\", \"cache_dir\": \"$DATA/algonauts/tribev2_stimuli/algonauts\", \"config_path\": \"$DATA/hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/config.yaml\"}" \
       --generated-index-dir $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-preparation/raw --workers 4
   ```
   For HCP, the same two commands with `hcp_movies` and its paths (`$DATA/hcp_movies/...`; the events step also writes each clip's audio to `audio_clips/` beside the table). Rebuilt for two Algonauts recordings and two HCP runs, the tables equal the ones the paper used; the audio can differ by decoder rounding.
3. **Bundles**: a read-only dry run, then the run.
   ```bash
   python -m neuroworld.nbs_data.schaefer prepare --dataset algonauts --source-root $DATA/algonauts/dataset \
       --raw-index $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-preparation/raw/chunks.jsonl
   python -m neuroworld.nbs_data.schaefer run --dataset algonauts --source-root $DATA/algonauts/dataset \
       --raw-index $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-preparation/raw/chunks.jsonl \
       --output $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2
   ```
   For HCP, use `--dataset hcp_movies` with its own paths and add `--atlas-manifest $DATA/atlases/neuroworld-schaefer1000-v2/manifest.json`; `--workers 24 --eager-volumes` reads each volume once, at about 12 GB of memory per worker (set `OMP_NUM_THREADS=1` and the like). `bundle.json` is written last and marks a finished bundle: 1,392 Algonauts runs and 725 HCP runs.
4. **Contextual rows** (about 30 minutes on 12 CPU threads for Algonauts; needs TRIBE v2's dependencies `neuraltrain` and `x_transformers`):
   ```bash
   python -m neuroworld.nbs_data.stimuli.tribev2_context --bundle $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2 \
       --checkpoint $DATA/hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/best.ckpt \
       --output $DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-tribev2-context --threads 12
   ```
5. **TRIBE v2's zero-shot predictions**, the stimulus-only reference, cached in `<bundle>-tribev2-predictions/`: `python -m neuroworld.eval.tribev2_reference --help`.

The result:

```text
$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2/                        fMRI, masks, pooled features, run records
$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-tribev2-context/        contextual rows, one array per stimulus
$DATA/algonauts/derivatives/neuroworld-schaefer1000-v2-tribev2-predictions/    TRIBE v2's predictions
$DATA/hcp_movies/derivatives/...                                               the same for HCP
```

The loaders always apply the bundle's `valid_time` mask. The dataset configs ([`algonauts/base.yaml`](neuroworld/configs/algonauts/base.yaml), [`hcp_movies/base.yaml`](neuroworld/configs/hcp_movies/base.yaml)) name these folders and define the splits.

## Splits

Every recording not used for training is test data.

| Dataset | Training | Test |
|---|---|---|
| Algonauts | *Friends* S1–5 | *Friends* S6 (`val`); *Hidden Figures* and *Life*, each watched twice (`dev`); *The Bourne Supremacy* and *The Wolf of Wall Street* (`test`) |
| HCP | 148 participants, runs 1–3 except one clip of each movie | the same 148: one held-out clip of each movie (`val`) and run 4 (`test`); 36 new participants, whose run 1 fits participant-specific heads and whose runs 2–4 are scored |

Intervals resample content parts on Algonauts (23 *Friends* S6 episodes and 43 Movie10 segments) and participants on HCP.
