#!/usr/bin/env bash
set -euo pipefail

# Edit these values, then run: bash test5.sh
dataset_name="gossipcop"
checkpoint_dir="checkpoints/gossipcop/20260406_163658_gossipcop_v1"
model_version="v1"
encoder_type="roberta"
gpu_id="2"

if [[ ! -d "$checkpoint_dir" ]]; then
  echo "[ERROR] checkpoint_dir not found: $checkpoint_dir"
  exit 1
fi

mkdir -p results

run_id="$(date +%Y%m%d_%H%M%S)"
base_name="evalall_${dataset_name}_${model_version}_${encoder_type}_${run_id}"
log_file="results/${base_name}.log"
status_file="results/${base_name}.status"

echo "start_time=$(date -Is)" > "$status_file"
echo "pid=$$" >> "$status_file"
echo "log_file=$log_file" >> "$status_file"
echo "checkpoint_dir=$checkpoint_dir" >> "$status_file"

echo "[START] $(date -Is)" > "$log_file"
echo "dataset_name=$dataset_name" >> "$log_file"
echo "checkpoint_dir=$checkpoint_dir" >> "$log_file"
echo "model_version=$model_version" >> "$log_file"
echo "encoder_type=$encoder_type" >> "$log_file"

set +e
CUDA_VISIBLE_DEVICES="$gpu_id" uv run src/eval_checkpoints.py \
  --dataset_name "$dataset_name" \
  --checkpoint_dir "$checkpoint_dir" \
  --model_version "$model_version" \
  --encoder_type "$encoder_type" \
  >> "$log_file" 2>&1
exit_code=$?
set -e

reason="unknown"
case "$exit_code" in
  0) reason="completed" ;;
  130) reason="interrupted_by_sigint" ;;
  137) reason="killed_by_sigkill_or_oom" ;;
  143) reason="terminated_by_sigterm" ;;
esac

echo "end_time=$(date -Is)" >> "$status_file"
echo "exit_code=$exit_code" >> "$status_file"
echo "reason=$reason" >> "$status_file"

echo "[END] $(date -Is)" >> "$log_file"
echo "[EXIT_CODE] $exit_code" >> "$log_file"
echo "[REASON] $reason" >> "$log_file"

echo "Done. Log: $log_file"
echo "Done. Status: $status_file"

exit "$exit_code"
