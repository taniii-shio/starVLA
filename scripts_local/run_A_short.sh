#!/usr/bin/env bash
# A の失敗の切り分け用の短い学習 (診断)。run_A.sh と同じ設定で、環境変数で次を変えられる。starVLA/ 直下で、tmux の中で起動する。
#   RUN_ID         出力名 (必須。上書き防止のため既定なし)
#   MAX_STEPS      既定 5000 (5k 本番と同じ cosine スケジュール)。判定に十分な step 1500 (約 1.5 h) まで進んだら Ctrl+C で止めてよい
#                  (LR の経路が B/C/A の 5k と同一になるので、窓ごとに直接比べられる。WARMUP を変える実験だけは経路が変わる)
#   WARMUP         既定 500 (5k 本番と同じ)。論文は 5000 (50k の 10%)
#   ALIGN_LAMBDA   既定 0.1
#   DETACH_INPUT   既定 false。true なら補助損失の勾配を VLM に流さない (ヘッドだけ学習)。診断用で、手法の修正ではない
#   CURRENT_TARGET 既定 false (A)。true なら C の目標
#   SEED           既定 42 (5k 本番と同じ)。モデル側の乱数とデータの抽出順の両方を変える。変えると序盤の step は 5k 本番と比べられない
#   GRAD_LOG       既定 false。true なら ${run_root_dir}/${RUN_ID}/grad_log.jsonl に step ごとの勾配ノルムを書く (1 GPU 前提。学習の計算は変えない)
#                  集計: python3 scripts_local/summarize_gradnorm.py /data/checkpoints/<RUN_ID>/grad_log.jsonl
# データの抽出順は 5k 本番と同じ (seed 42) なので、序盤の step は B/C/A の 5k と直接比べられる。
# 使い方は impl/analysis_20k.md §8 を参照。
set -euo pipefail

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4
export GRAD_LOG=${GRAD_LOG:-false}

: "${RUN_ID:?RUN_ID を指定する (例: RUN_ID=A_short_detach)}"
run_id=${RUN_ID}
max_steps=${MAX_STEPS:-5000}
warmup_steps=${WARMUP:-500}
align_lambda=${ALIGN_LAMBDA:-0.1}
detach_input=${DETACH_INPUT:-false}
current_target=${CURRENT_TARGET:-false}

Framework_name=QwenPIAlign
base_vlm=/data/pretrained/Qwen3-VL-2B-Instruct
config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/data/datasets/libero
data_mix=libero_all
run_root_dir=/data/checkpoints
per_device_batch=4
cache_dir=/data/cache/dinov3_b16
align_layer=18
align_head_lr=1.0e-4

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} /data/logs
cp "$0" ${output_dir}/
echo "[run_A_short] run_id=${run_id} steps=${max_steps} warmup=${warmup_steps} lambda=${align_lambda} detach=${detach_input} current_target=${current_target}"

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
  --framework.align.detach_input ${detach_input} \
  --datasets.vla_data.wm_align.enabled true \
  --datasets.vla_data.wm_align.cache_dir ${cache_dir} \
  --datasets.vla_data.wm_align.current_target ${current_target} \
  --trainer.learning_rate.align_head ${align_head_lr} \
  --seed ${SEED:-42} \
  --datasets.vla_data.seed ${SEED:-42} \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps ${max_steps} \
  --trainer.num_warmup_steps ${warmup_steps} \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity local \
  2>&1 | tee /data/logs/${run_id}.log
