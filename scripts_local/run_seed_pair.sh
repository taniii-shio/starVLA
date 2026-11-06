#!/usr/bin/env bash
# seed を変えた A と C を順に 1 本ずつ走らせ、各 STOP_STEP で自動停止する (診断)。starVLA/ 直下で、tmux の中で起動する。
#   SEED       既定 43
#   STOP_STEP  既定 1500。ログにこの step が出たら学習を止める (cosine は MAX_STEPS=5000 のまま = 5k 本番と同じ LR の経路)
#   ARMS       既定 "A C"。順に走らせる (1 GPU 前提)
# 出力: /data/logs/{A,C}_seed${SEED}.log、/data/checkpoints/{A,C}_seed${SEED}/
# 1 本 約 1.5 h。2 本で約 3 h。
set -uo pipefail

SEED=${SEED:-43}
STOP_STEP=${STOP_STEP:-1500}
ARMS=${ARMS:-"A C"}

for arm in ${ARMS}; do
  run_id=${arm}_seed${SEED}
  if [ -e "/data/logs/${run_id}.log" ]; then
    echo "[seed_pair] /data/logs/${run_id}.log が既にある。上書きしないので中止する" >&2
    exit 1
  fi
  ct=false; [ "${arm}" = "C" ] && ct=true
  echo "[seed_pair] start ${run_id} (current_target=${ct}, seed=${SEED}, stop at step ${STOP_STEP})"

  RUN_ID=${run_id} SEED=${SEED} MAX_STEPS=5000 CURRENT_TARGET=${ct} bash scripts_local/run_A_short.sh &
  runner=$!

  # ログに "Step ${STOP_STEP}," が出るまで待って止める。学習が先に終わったら抜ける
  while kill -0 ${runner} 2>/dev/null; do
    if [ -f "/data/logs/${run_id}.log" ] && \
       sed 's/\x1b\[[0-9;]*m//g' "/data/logs/${run_id}.log" | grep -q "Step ${STOP_STEP}, Loss"; then
      echo "[seed_pair] ${run_id}: step ${STOP_STEP} に到達。停止する"
      pkill -INT -f "starVLA/training/train_starvla.py" || true
      sleep 20
      pkill -TERM -f "starVLA/training/train_starvla.py" || true
      break
    fi
    sleep 60
  done
  wait ${runner} 2>/dev/null || true
  sleep 30   # GPU メモリの解放待ち
done
echo "[seed_pair] done"
