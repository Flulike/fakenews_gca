CUDA_VISIBLE_DEVICES=2 python src/sheepdog.py --dataset_name lun --model_name sheepdog --n_epochs 10 --iters 10 --batch_size 4 --model_version v1 --disable_gate > results/$(date +%Y%m%d_%H%M)_lun_v1_nogate.log 2>&1

# `[--dataset_name]`: politifact / gossipcop / lun
