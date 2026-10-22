#!/usr/bin/env bash
# A の追加学習 (比較用ではない)。A_s0_5k の step 5000 の重みから始め、LR スケジュールとオプティマイザは新しくする。
#  - 重みだけを読む (trainer.pretrained_checkpoint)。AdamW の状態は保存されていないので、再開ではなく
#    warmup 500 から始め直す。ステップ数は 0 から数える (ログの Step 1 = 通算 5001 相当)。
#  - サンプルの並びは seed に依存する。同じ seed だと最初の 5000 step と同じサンプルを繰り返すので、seed を変える。
# 起動: tmux の中で starVLA/ 直下から。 MAX_STEPS=15000 (既定) で約 15 h。
set -euo pipefail

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4

Framework_name=QwenPIAlign
freeze_module_list=''
base_vlm=/data/pretrained/Qwen3-VL-2B-Instruct
config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/data/datasets/libero
data_mix=libero_all
run_root_dir=/data/checkpoints
run_id=${RUN_ID:-A_s0_5k_cont}
init_ckpt=${INIT_CKPT:-/data/checkpoints/A_s0_5k/checkpoints/steps_5000_pytorch_model.pt}
max_steps=${MAX_STEPS:-15000}
warmup_steps=500
save_interval=${SAVE_INTERVAL:-5000}
per_device_batch=4
data_seed=${DATA_SEED:-43}   # 最初の学習は 42 (既定)
cache_dir=/data/cache/dinov3_b16
align_lambda=0.1
align_layer=18
align_head_lr=1.0e-4

[[ -f "${init_ckpt}" ]] || { echo "初期チェックポイントが無い: ${init_ckpt}" >&2; exit 1; }

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} /data/logs
cp "$0" ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file scripts_local/deepspeed/deepspeed_zero2_ga4.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch} \
  --datasets.vla_data.seed ${data_seed} \
  --trainer.vla_data.video_backend torchvision_av \
  --framework.align.enabled true \
  --framework.align.lambda ${align_lambda} \
  --framework.align.layer ${align_layer} \
  --datasets.vla_data.wm_align.enabled true \
  --datasets.vla_data.wm_align.cache_dir ${cache_dir} \
  --datasets.vla_data.wm_align.current_target false \
  --trainer.learning_rate.align_head ${align_head_lr} \
  --trainer.pretrained_checkpoint ${init_ckpt} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${max_steps} \
  --trainer.num_warmup_steps ${warmup_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity local \
  2>&1 | tee /data/logs/${run_id}.log
