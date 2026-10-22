#!/usr/bin/env bash
# 再現 A (WM-Align: 未来 t+4/8/12/16 の DINOv3 特徴) の本学習。starVLA/ 直下で実行する。
# 実効バッチ = per_device 4 × 勾配蓄積 4 × GPU 1 = 16 (80GB に収めるため分割。spec §6.1 の 64 とは別。log.md に逸脱として記録)
# 5k step × 約 3.5 s/it ≈ 5 h。B / A / C の3アームを同じ 5k・warmup 500 で揃える。tmux の中で起動すること。
set -euo pipefail

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 蓄積回数は Accelerate の DeepSpeedPlugin が環境変数から読む (ds_config の値は効かない、smoke で確認済み)
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4

Framework_name=QwenPIAlign
freeze_module_list=''   # VLM も学習する (smoke と同じ。trainable 3.804B)
base_vlm=/data/pretrained/Qwen3-VL-2B-Instruct
config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/data/datasets/libero
data_mix=libero_all
run_root_dir=/data/checkpoints
run_id=${RUN_ID:-A_s0_5k}
max_steps=${MAX_STEPS:-5000}   # スモークは MAX_STEPS=20 RUN_ID=A_smoke で上書き
warmup_steps=500       # YAML は 5000 (100k 向け)。5k で完結させるため max_steps の 10% に揃える
save_interval=2500
per_device_batch=4
cache_dir=/data/cache/dinov3_b16   # build_dino_cache.py の出力
align_lambda=0.1
align_layer=18
align_head_lr=1.0e-4

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
  --trainer.vla_data.video_backend torchvision_av \
  --framework.align.enabled true \
  --framework.align.lambda ${align_lambda} \
  --framework.align.layer ${align_layer} \
  --datasets.vla_data.wm_align.enabled true \
  --datasets.vla_data.wm_align.cache_dir ${cache_dir} \
  --datasets.vla_data.wm_align.current_target false \
  --trainer.learning_rate.align_head ${align_head_lr} \
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
