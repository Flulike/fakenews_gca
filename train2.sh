#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

run_id="$(date +%Y%m%d_%H%M%S)"
log_file="results/${run_id}_lun_v2__bert.log"
status_file="results/${run_id}_lun_v2__bert.status"

echo "start_time=$(date -Is)" > "$status_file"
echo "pid=$$" >> "$status_file"
echo "log_file=$log_file" >> "$status_file"

echo "[START] $(date -Is)" > "$log_file"

set +e
CUDA_VISIBLE_DEVICES=2 uv run src/sheepdog.py \
	--dataset_name lun \
	--model_name sheepdog \
	--iters 10 \
	--n_epochs 5 \
	--batch_size 4 \
	--model_version v2 \
	--use_match_loss \
	--encoder_type bert \
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