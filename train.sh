#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

dataset_name="lun"
encoder_type="roberta"
model_version="v2"
distorted=false
use_match_loss=true
disable_gate=false

run_id="$(date +%Y%m%d_%H%M%S)"
run_name="${run_id}_${dataset_name}_${model_version}"
if [[ "$encoder_type" == "bert" ]]; then
	run_name="${run_name}_bert"
fi
if [[ "$distorted" == true ]]; then
	run_name="${run_name}_distort"
fi
if [[ "$use_match_loss" == false ]]; then
	run_name="${run_name}_noloss"
fi
if [[ "$disable_gate" == true ]]; then
	run_name="${run_name}_nogate"
fi
log_file="results/${run_name}.log"
status_file="results/${run_name}.status"

mkdir -p "checkpoints/${dataset_name}/${run_name}"

echo "start_time=$(date -Is)" > "$status_file"
echo "pid=$$" >> "$status_file"
echo "log_file=$log_file" >> "$status_file"

echo "[START] $(date -Is)" > "$log_file"

set +e
CUDA_VISIBLE_DEVICES=0 uv run src/sheepdog.py \
	--dataset_name $dataset_name \
	--model_name sheepdog \
	--n_epochs 5 \
	--iters 5 \
	--batch_size 4 \
	--model_version $model_version \
	$( [[ "$distorted" == true ]] && echo "--distorted" ) \
	$( [[ "$use_match_loss" == true ]] && echo "--use_match_loss" ) \
	$( [[ "$disable_gate" == true ]] && echo "--disable_gate" ) \
	--run_name "$run_name" \
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

exit "$exit_code"

# [--dataset_name]: politifact / gossipcop / lun
