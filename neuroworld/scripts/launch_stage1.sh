#!/usr/bin/env bash
# Stage 1 for one model design:
#   ./neuroworld/scripts/launch_stage1.sh neuroworld/configs/algonauts/transformer_world_model.yaml [--set section.key=value ...]
# The design YAML fixes the objective, stimulus mode and encoder; --set overrides
# any other setting. The variables below override epochs, batch size, lr and precision when set.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"

python="${PYTHON:-python}"
config="${1:?Usage: launch_stage1.sh <design YAML, e.g. neuroworld/configs/algonauts/transformer_world_model.yaml> [--set ...]}"
shift
# The run directory is named automatically, checkpoints/<dataset>/<run name from the settings>,
# and printed at startup. RUN_TAG appends a suffix; RUN_DIR chooses the directory instead.
# Optional overrides: EPOCHS, BATCH_SIZE (global; split evenly across GPUs), LR, PRECISION.
# W&B uses the YAML settings; optional overrides:
# export WANDB_PROJECT=neuralworld
# export WANDB_MODE=offline  # online / offline / disabled
# export WANDB_NAME=my-run
# To resume: launch_stage1.sh <run directory>/config.yaml --resume <run directory>/latest.pt
# Select GPUs before running, for example CUDA_VISIBLE_DEVICES=0,1 ./launch_stage1.sh <YAML>.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
overrides=()
[[ -n "${EPOCHS:-}" ]] && overrides+=(--epochs "$EPOCHS")
[[ -n "${BATCH_SIZE:-}" ]] && overrides+=(--batch-size "$BATCH_SIZE")
[[ -n "${LR:-}" ]] && overrides+=(--lr "$LR")
[[ -n "${PRECISION:-}" ]] && overrides+=(--precision "$PRECISION")
[[ -n "${RUN_TAG:-}" ]] && overrides+=(--tag "$RUN_TAG")
[[ -n "${RUN_DIR:-}" ]] && overrides+=(--output-dir "$RUN_DIR")
precision="${PRECISION:-$("$python" -c 'import sys; from neuroworld.utils.config import load_config; print(load_config(sys.argv[1])["train"]["precision"])' "$config")}"
num_gpus=$("$python" -c 'import torch; print(torch.cuda.device_count())')
num_processes=$((num_gpus > 0 ? num_gpus : 1))
mixed_precision="$precision"
if [[ "$precision" == "fp32" ]]; then mixed_precision=no; fi
launch_args=(--config_file neuroworld/configs/accelerate.yaml
             --num_processes "$num_processes" --num_machines 1
             --mixed_precision "$mixed_precision"
             --num_cpu_threads_per_process "$OMP_NUM_THREADS")
if ((num_gpus > 1)); then
    launch_args+=(--multi_gpu --main_process_port 0)
elif ((num_gpus == 0)); then
    launch_args+=(--cpu)
fi

echo "Stage 1: $num_gpus visible GPU(s), $num_processes process(es), precision $precision"
exec "$python" -m accelerate.commands.launch "${launch_args[@]}" \
    --module neuroworld.train.latent "$config" \
    "${overrides[@]}" "$@"