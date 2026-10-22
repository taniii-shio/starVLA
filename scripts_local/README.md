# scripts_local: WM-Align の再現実験 (LIBERO)

arXiv 2610.19825 の **WM-Align** (VLM の中間層の視覚トークンに DINOv3 の特徴を予測させる補助損失) を、starVLA の `QwenPI` に足して LIBERO で動かしたときのスクリプト。
講義課題の提出用。実行環境は Azure VM (A100 80GB ×1)。

## 構成 (3 アーム)

| ID | `Framework_name` | 補助損失の目標 | スクリプト |
|---|---|---|---|
| B | `QwenPI` | なし (ベースライン) | `run_B_main.sh` |
| A | `QwenPIAlign` | 未来フレーム t+4, 8, 12, 16 の DINOv3 特徴 | `run_A.sh` |
| C | `QwenPIAlign` | 現在フレーム t の DINOv3 特徴 (`current_target=true`) | `run_C.sh` |

3 アームとも 5,000 ステップ、実効バッチ 16 (per-device 4 × 勾配蓄積 4)、warmup 500。

## このブランチで足した・変えたコード

新規:
- `starVLA/model/framework/VLM4A/QwenPIAlign.py` — `QwenPI` を継承。18 層目の視覚トークンを `AlignHead` に通し、`aux_loss = λ · world_loss` を返す
- `starVLA/model/modules/align/align_head.py` — `AlignHead` (2 層 MLP、約 21M)、`gather_visual_tokens`、`align_loss` (1 − cos)
- `starVLA/model/modules/align/target_cache.py` — DINOv3 特徴のキャッシュ (memmap) から、未来 (または現在) の目標を引く

変更 (既存):
- `starVLA/dataloader/gr00t_lerobot/datasets.py` — サンプルに `wm_target` を追加
- `starVLA/dataloader/lerobot_datasets.py` — データのシードを設定から変えられるようにする
- `starVLA/training/train_starvla.py` — 総損失に `aux_loss` を足し、`world_loss` をログに出す

## 手順

```bash
# 1. DINOv3 特徴の事前計算 (Hugging Face のゲート付きモデル。利用同意とログインが要る)
python scripts_local/build_dino_cache.py --data_root /data/datasets/libero --cache_dir /data/cache/dinov3_b16
python scripts_local/check_dino_cache.py --data_root /data/datasets/libero --cache_dir /data/cache/dinov3_b16   # 再計算との一致を検査

# 2. 学習 (tmux の中で。B / A / C それぞれ約 5 時間)
bash scripts_local/run_B_main.sh
bash scripts_local/run_A.sh
bash scripts_local/run_C.sh

# 3. 評価: LIBERO-Goal、10 タスク × 50 試行
SUITES="libero_goal" bash scripts_local/run_eval_arm.sh B 6694 0
SUITES="libero_goal" bash scripts_local/run_eval_arm.sh A 6695 0
SUITES="libero_goal" bash scripts_local/run_eval_arm.sh C 6696 0

# 4. 集計
python scripts_local/collect_results.py --out /data/export --max_videos_per_arm 10   # VM で。results.json と成功動画を export.tar にまとめる
```

パス (`/data/...`) と GPU の数は VM の構成に合わせてある。違う環境では、各スクリプト冒頭の変数を直す。
評価で使う LIBERO の Python は、環境変数 `LIBERO_HOME` / `LIBERO_PYTHON` で上書きできる。

## 補助スクリプト

| ファイル | 内容 |
|---|---|
| `wm_copy_baseline.py` | 「現在の特徴をそのまま未来として出したとき」の損失 (world_loss の基準値) を測る |
| `collect_results.py` | 学習ログと評価結果を `results.json` にまとめ、成功動画を tar にする |
| `run_B.sh` | B の smoke run (300 ステップ) |
| `run_A_cont.sh` | A の追加学習 (**未実行**) |
| `deepspeed/` | DeepSpeed ZeRO-2、勾配蓄積 4 の設定 |

## 結果 (5,000 ステップ、LIBERO-Goal、500 エピソード / アーム)

| アーム | 成功数 |
|---|---|
| B | 3 / 500 |
| A | 0 / 500 |
| C | 3 / 500 |

学習が論文 (50,000 ステップ・バッチ 128) の約 1/10 以下と短く、どのアームも成功率は 1% 未満。アーム間の優劣は評価できていない。
