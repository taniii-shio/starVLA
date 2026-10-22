#!/usr/bin/env bash
# 1 アーム分の LIBERO 評価: ポリシーサーバーを起動 → 指定スイートを順に評価 → サーバーを止める。
#
#   bash scripts_local/run_eval_arm.sh <B|A|C> <PORT> [GPU_ID]
#   SUITES="libero_goal" NUM_TRIALS_PER_TASK=1 bash scripts_local/run_eval_arm.sh A 6694   # 動作確認
#   bash scripts_local/run_eval_arm.sh A 6694 0                                             # 本番 (4 スイート × 50)
#
# 3 アームを並列にするときは、ポートを変える (6694 / 6695 / 6696) こと。
set -euo pipefail

ARM=${1:?B|A|C}
PORT=${2:?port}
GPU=${3:-0}
SUITES=${SUITES:-"libero_object libero_goal libero_spatial libero_10"}

STARVLA_DIR=/data/starVLA
EVAL_DIR=${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files
LOG_DIR=/data/logs

export STARVLA_DIR
# 評価用の LIBERO (別の conda 環境 "libero")。自分の環境に合わせて環境変数で上書きする
export LIBERO_HOME=${LIBERO_HOME:-/data/LIBERO}
export LIBERO_PYTHON=${LIBERO_PYTHON:-$HOME/miniforge3/envs/libero/bin/python}
export CKPT=/data/checkpoints/${ARM}_s0_5k/checkpoints/steps_5000_pytorch_model.pt
export PORT GPU_ID=${GPU} USE_BF16=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}

[[ -f "${CKPT}" ]] || { echo "チェックポイントが無い: ${CKPT}" >&2; exit 1; }

if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  echo "ポート ${PORT} が既に使われている。前のサーバーが残っていないか: pgrep -fa server_policy.py" >&2
  exit 1
fi

echo "== arm=${ARM} port=${PORT} gpu=${GPU} trials=${NUM_TRIALS_PER_TASK} suites=${SUITES}"
bash "${EVAL_DIR}/run_policy_server.sh" > "${LOG_DIR}/serve_${ARM}.log" 2>&1 &
SERVER_PID=$!
# run_policy_server.sh の下で python が子プロセスとして動くので、このポートのサーバーを名指しで止める
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  pkill -f "server_policy.py.*--port ${PORT}( |\$)" 2>/dev/null || true
}
trap cleanup EXIT

# サーバーがポートを開くまで待つ (最大 15 分)
for _ in $(seq 1 180); do
  if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then break; fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "サーバーが落ちた: ${LOG_DIR}/serve_${ARM}.log を見る" >&2; exit 1; }
  # 注: この疎通確認はサーバーのログに websockets の EOFError (HTTP 要求なしで切断) を 1 回ずつ残す。無害
  sleep 5
done
(echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null || { echo "ポート ${PORT} が開かない" >&2; exit 1; }
echo "server up: $(grep -m1 'WM-Align' ${LOG_DIR}/serve_${ARM}.log || echo 'WM-Align 行なし (B なら正常)')"

for suite in ${SUITES}; do
  echo "== ${ARM} ${suite} start $(date '+%F %T')"
  TASK_SUITE_NAME=${suite} bash "${EVAL_DIR}/eval_libero.sh" 2>&1 | tee -a "${LOG_DIR}/eval_${ARM}_${suite}.log"
  echo "== ${ARM} ${suite} end   $(date '+%F %T')"
done
echo "== ${ARM} all done"
