#!/usr/bin/env bash
# Download the datasets into $STATE_OF_MIND_DATA (fallback $NEUROWORLD_DATA, default data/ in the repository root), where DATA.md's steps read them.
#   state_of_mind/nbs_data/download.sh algonauts   Algonauts 2025 (CNeuroMod) with DataLad: fMRI, movies and transcripts
#   state_of_mind/nbs_data/download.sh hcp         HCP 7T movie fMRI from HCP's AWS bucket, with the aws CLI and the AWS
#                                                  credentials HCP gives after you accept its data use terms
# HCP first needs HCP's 7T movie stimulus package (movie_stimulus_2025.zip) unpacked into
# $STATE_OF_MIND_DATA/hcp_movies/dataset/movie_stimulus/: its subject lists say whom to download. Files already present are
# skipped, so a stopped download resumes. DRY_RUN=1 prints the commands instead.
set -euo pipefail
data="${STATE_OF_MIND_DATA:-${NEUROWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}}"
run() { if [[ -n "${DRY_RUN:-}" ]]; then echo "$*"; else "$@"; fi; }

case "${1:-}" in
  algonauts)
    root="$data/algonauts/dataset"
    [[ -d "$root/.datalad" ]] || run datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git "$root"
    run datalad get -r -J8 -d "$root" "$root/fmri" "$root/stimuli"
    ;;
  hcp)
    root="$data/hcp_movies/dataset"
    lists="$root/movie_stimulus/movie_stimulus_2025"
    [[ -f "$lists/HCP_7T_movie_V1_ReleasedSubjs.csv" ]] || {
      echo "unpack HCP's 7T movie stimulus package (movie_stimulus_2025.zip) into $root/movie_stimulus/ first" >&2; exit 1; }
    get() {  # get <bucket key> <destination>
      [[ -f "$2" ]] || run aws s3 cp "s3://hcp-openaccess/$1" "$2" || echo "not in the bucket: $1" >&2
    }
    for subject in $(awk -F, 'FNR > 1 {sub("\r", "", $1); print $1}' "$lists"/HCP_7T_movie_V{1,2}_ReleasedSubjs.csv); do
      mni="$subject/MNINonLinear"
      get "HCP_1200/$mni/ROIs/wmparc.1.60.nii.gz" "$root/$mni/ROIs/wmparc.1.60.nii.gz"
      for r in tfMRI_MOVIE1_7T_AP tfMRI_MOVIE2_7T_PA tfMRI_MOVIE3_7T_PA tfMRI_MOVIE4_7T_AP; do
        for f in "$r.nii.gz" brainmask_fs.1.60.nii.gz Movement_Regressors.txt; do
          get "HCP_1200/$mni/Results/$r/$f" "$root/$mni/Results/$r/$f"
        done
      done
    done
    ;;
  *) echo "usage: $0 algonauts|hcp" >&2; exit 2 ;;
esac
