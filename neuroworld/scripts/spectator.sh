#!/usr/bin/env bash
# Spectator end to end (neuroworld/spectator/__init__.py), from the frozen configs in neuroworld/configs/spectator/:
#   OUT=<dir> neuroworld/scripts/spectator.sh algonauts      stage 0, lag weights, JEPA phase, final matrix, model, scores
#   OUT=<dir> neuroworld/scripts/spectator.sh hcp            lag weights and final matrix refitted on HCP, model, scores
# HISTORY sets the seconds of fMRI history on Algonauts: 180 (default, the paper's model), 7.5 and 120 (the main
# table's other columns) or 15-90 (the appendix). Only the window changes. At 7.5 s the window holds just the 5 frames
# the fast latent reads, so that model has no slow latent: no lag weights, and its JEPA phase starts from stage 0.
#   HISTORY=120 OUT=<dir> neuroworld/scripts/spectator.sh algonauts
# Each step is skipped when its output exists, so a stopped run resumes. HCP (180 s) starts from $OUT/encoder.pt when
# the Algonauts steps ran there, otherwise from the frozen release's encoder (ENCODER overrides). Scores go to $STORE
# (default $OUT/benchmark). The reported model ran stage 0 on one GPU and the JEPA phase on two (CUDA_VISIBLE_DEVICES).
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
python="${PYTHON:-python}"
out="${OUT:?Set OUT to the output directory}"
store="${STORE:-$out/benchmark}"
release="${RELEASE:-$("$python" -c 'from neuroworld.paths import OUTPUT; print(OUTPUT / "spectator-v1")')}"
configs=neuroworld/configs/spectator
spectator() { "$python" -m neuroworld.spectator "$@"; }
history="${HISTORY:-180}"
frames="$("$python" -c "from neuroworld.benchmark.grid import frames_within; print(frames_within(float('$history'), 1.49))")"
if (( frames > 5 )); then  # the window's size; the frozen configs hold 120 frames (180 s at TR 1.49 s)
  window=(--set "model.state_window_trs=$frames"); start="$out/lags.pt"
else                       # 7.5 s: the fast latent alone, started from stage 0's encoder
  window=(--set model.state_window_trs=5 --set model.fast_window_trs=0 --set model.state_dim=1024 --set model.state_tokens=1
          --set model.slow_per_dimension=false); start="$out/stage0/best.pt"
fi
mkdir -p "$out"

train() {  # train <config> <run dir>: one gradient stage through launch_stage1.sh
  if [[ ! -f "$2/stage1.done" ]]; then
    resume=(); [[ -f "$2/latest.pt" ]] && resume=(--resume "$2/latest.pt")
    RUN_DIR="$2" neuroworld/scripts/launch_stage1.sh "$1" --set wandb.mode=offline "${resume[@]}"
    touch "$2/stage1.done"
  fi
}

case "${1:-algonauts}" in
  algonauts)
    train "$configs/algonauts-stage0.yaml" "$out/stage0"                                                    # 1. stage 0
    if (( frames > 5 )); then                                                                                 # 2. lag weights
      [[ -f "$out/lags.pt" ]] || spectator lags "$out/stage0/best.pt" "$out/lags.pt" --frames "$frames" --ahead 16
    fi
    [[ -f "$out/jepa-start-matrix.pt" ]] || spectator matrix "$configs/algonauts-jepa.yaml" "$out/jepa-start-matrix.pt" \
      --encoder "$start" "${window[@]}" --movie-factor 10 --state-factor 1 --full-windows-only             # 3. JEPA phase
    spectator config "$configs/algonauts-jepa.yaml" "$out/jepa.yaml" --encoder "$start" --predictor "$out/jepa-start-matrix.pt" \
      "${window[@]}"
    train "$out/jepa.yaml" "$out/jepa"
    [[ -f "$out/encoder.pt" ]] || spectator encoder "$out/jepa/best.pt" "$start" "$out/encoder.pt"         # 4. freeze the encoder
    [[ -f "$out/matrix.pt" ]] || spectator matrix "$configs/algonauts.yaml" "$out/matrix.pt" --encoder "$out/encoder.pt" \
      "${window[@]}"                                                                                         # 5. final matrix
    [[ -f "$out/model/best.pt" ]] || spectator assemble "$configs/algonauts.yaml" "$out/matrix.pt" "$out/model" \
      --encoder "$out/encoder.pt" "${window[@]}"
    spectator score "$out/model" --seconds "$history" --dataset algonauts --store "$store" --name "spectator-${history}s" --gap  # 6.
    ;;
  hcp)
    encoder="${ENCODER:-$out/encoder.pt}"; [[ -f "$encoder" ]] || encoder="$release/algonauts/encoder.pt"
    [[ -f "$out/hcp-lags.pt" ]] || spectator lags "$encoder" "$out/hcp-lags.pt" --frames 180 --ahead 24 \
      --dataset-config neuroworld/configs/hcp_movies/base.yaml
    [[ -f "$out/hcp-matrix.pt" ]] || spectator matrix "$configs/hcp.yaml" "$out/hcp-matrix.pt" --encoder "$out/hcp-lags.pt" --movie-factor 10000
    [[ -f "$out/hcp-model/best.pt" ]] || spectator assemble "$configs/hcp.yaml" "$out/hcp-matrix.pt" "$out/hcp-model" --encoder "$out/hcp-lags.pt"
    spectator score "$out/hcp-model" --seconds 180 --dataset hcp_movies --store "$store" --name spectator-hcp --gap
    ;;
  *) echo "usage: [HISTORY=<seconds>] OUT=<dir> $0 algonauts|hcp" >&2; exit 2 ;;
esac
