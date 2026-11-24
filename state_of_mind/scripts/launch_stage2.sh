#!/usr/bin/env bash
# Stage 2: fit the readout with a frozen world model. Not used by the fMRI forecaster.
# The YAML holds the run settings; the variables below override them only when set.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"

python="${PYTHON:-python}"
# RUN_DIR is the stage-1 run directory printed by launch_stage1.sh; its saved config is used by default.
run_dir="${RUN_DIR:?Set RUN_DIR to the stage-1 run directory, e.g. checkpoints/algonauts/<run name>}"
config="${CONFIG:-$run_dir/config.yaml}"
# Optional overrides: EPOCHS, BATCH_SIZE (global; split evenly across GPUs), LR, PRECISION;
# other settings with arguments, e.g. --set readout.hidden_dim=0 readout.select_by=parcel_pearson.
# W&B uses the YAML settings; optional overrides:
# export WANDB_PROJECT=neuralworld
# export WANDB_MODE=offline  # online / offline / disabled
# export WANDB_NAME=my-run
checkpoint="${CHECKPOINT:-$run_dir/best.pt}"
output_dir="${OUTPUT_DIR:-$run_dir/readout}"
# Select GPUs before running, for example CUDA_VISIBLE_DEVICES=0,1 ./launch_stage2.sh.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
overrides=()
[[ -n "${EPOCHS:-}" ]] && overrides+=(--epochs "$EPOCHS")
[[ -n "${BATCH_SIZE:-}" ]] && overrides+=(--batch-size "$BATCH_SIZE")
[[ -n "${LR:-}" ]] && overrides+=(--lr "$LR")
[[ -n "${PRECISION:-}" ]] && overrides+=(--precision "$PRECISION")
precision="${PRECISION:-$("$python" -c 'import sys; from state_of_mind.utils.config import load_config; print(load_config(sys.argv[1])["train"]["precision"])' "$config")}"
num_gpus=$("$python" -c 'import torch; print(torch.cuda.device_count())')
num_processes=$((num_gpus > 0 ? num_gpus : 1))
mixed_precision="$precision"
if [[ "$precision" == "fp32" ]]; then mixed_precision=no; fi
launch_args=(--config_file state_of_mind/configs/accelerate.yaml
             --num_processes "$num_processes" --num_machines 1
             --mixed_precision "$mixed_precision"
             --num_cpu_threads_per_process "$OMP_NUM_THREADS")
if ((num_gpus > 1)); then
    launch_args+=(--multi_gpu --main_process_port 0)
elif ((num_gpus == 0)); then
    launch_args+=(--cpu)
fi

echo "Stage 2: $num_gpus visible GPU(s), $num_processes process(es), precision $precision"
exec "$python" -m accelerate.commands.launch "${launch_args[@]}" \
    --module state_of_mind.train.readout "$config" \
    "${overrides[@]}" --checkpoint "$checkpoint" --output-dir "$output_dir" "$@"
