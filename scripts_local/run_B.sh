#!/usr/bin/env bash
# 再現 B (ベースライン) の smoke run。starVLA/ 直下で実行する。
# 実効バッチ = per_device 4 × 勾配蓄積 4 × GPU 1 = 16 (80GB に収めるため分割。spec §6.2 の 64 とは別)
set -euo pipefail

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 蓄積回数は Accelerate の DeepSpeedPlugin が環境変数から読む (ds_config の値は効かなかった、ログで確認済み)
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4

Framework_name=QwenPI
freeze_module_list=''   # YAML は qwen_vl_interface を凍結と書くが、ここで空にして VLM も学習する。論文 Table A.2 の学習可能パラメータ 2.608B は VLM を含む量と読めるため【解釈】。起動ログの trainable パラメータ数で確認する
base_vlm=/data/pretrained/Qwen3-VL-2B-Instruct
config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/data/datasets/libero
data_mix=libero_all
run_root_dir=/data/checkpoints
run_id=B_smoke_bs4ga4
max_steps=300
per_device_batch=4

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
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
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${max_steps} \
  --trainer.save_interval ${max_steps} \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity local
