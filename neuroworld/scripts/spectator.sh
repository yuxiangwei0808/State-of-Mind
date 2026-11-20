#!/usr/bin/env bash
# Spectator end to end (neuroworld/spectator/__init__.py), from the frozen configs in neuroworld/configs/spectator/:
#   OUT=<dir> neuroworld/scripts/spectator.sh algonauts      stage 0, lag weights, JEPA phase, final matrix, model, scores
#   OUT=<dir> neuroworld/scripts/spectator.sh hcp            lag weights and final matrix refitted on HCP, model, scores
# Each step is skipped when its output exists, so a stopped run resumes. HCP starts from $OUT/encoder.pt when the
# Algonauts steps ran there, otherwise from the frozen release's encoder (ENCODER overrides). Scores go to $STORE
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
    [[ -f "$out/lags.pt" ]] || spectator lags "$out/stage0/best.pt" "$out/lags.pt" --frames 120 --ahead 16    # 2. lag weights
    [[ -f "$out/jepa-start-matrix.pt" ]] || spectator matrix "$configs/algonauts-jepa.yaml" "$out/jepa-start-matrix.pt" \
      --encoder "$out/lags.pt" --movie-factor 10 --state-factor 1 --full-windows-only                       # 3. JEPA phase
    spectator config "$configs/algonauts-jepa.yaml" "$out/jepa.yaml" --encoder "$out/lags.pt" --predictor "$out/jepa-start-matrix.pt"
    train "$out/jepa.yaml" "$out/jepa"
    [[ -f "$out/encoder.pt" ]] || spectator encoder "$out/jepa/best.pt" "$out/lags.pt" "$out/encoder.pt"   # 4. freeze the encoder
    [[ -f "$out/matrix.pt" ]] || spectator matrix "$configs/algonauts.yaml" "$out/matrix.pt" --encoder "$out/encoder.pt"   # 5. final matrix
    [[ -f "$out/model/best.pt" ]] || spectator assemble "$configs/algonauts.yaml" "$out/matrix.pt" "$out/model" --encoder "$out/encoder.pt"
    spectator score "$out/model" --seconds 180 --dataset algonauts --store "$store" --name spectator --gap      # 6. scores
    ;;
  hcp)
    encoder="${ENCODER:-$out/encoder.pt}"; [[ -f "$encoder" ]] || encoder="$release/algonauts/encoder.pt"
    [[ -f "$out/hcp-lags.pt" ]] || spectator lags "$encoder" "$out/hcp-lags.pt" --frames 180 --ahead 24 \
      --dataset-config neuroworld/configs/hcp_movies/base.yaml
    [[ -f "$out/hcp-matrix.pt" ]] || spectator matrix "$configs/hcp.yaml" "$out/hcp-matrix.pt" --encoder "$out/hcp-lags.pt" --movie-factor 10000
    [[ -f "$out/hcp-model/best.pt" ]] || spectator assemble "$configs/hcp.yaml" "$out/hcp-matrix.pt" "$out/hcp-model" --encoder "$out/hcp-lags.pt"
    spectator score "$out/hcp-model" --seconds 180 --dataset hcp_movies --store "$store" --name spectator-hcp --gap
    ;;
  *) echo "usage: OUT=<dir> $0 algonauts|hcp" >&2; exit 2 ;;
esac
