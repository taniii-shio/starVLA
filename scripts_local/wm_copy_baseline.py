"""WM-Align の world_loss の基準値 (何もしない予測の損失)。VM の starvla env で CPU だけ使って実行する。

  python scripts_local/wm_copy_baseline.py --cache_dir /data/cache/dinov3_b16 --n_samples 2000

「現在の特徴を、未来の特徴としてそのままコピーする」予測の損失を、オフセット k ごとに求める:
  L_copy(k) = mean_patch(1 - cos(特徴(t), 特徴(t+k)))
学習ログの world_loss (A: 終盤で約 0.05) と比べ、ヘッドが「現在の特徴をコピーする」より良い予測をしているかを見る基準。
サンプリング: エピソードを一様に選び、その中の step を一様に選ぶ (学習時の厳密な分布とは少し違う)。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from starVLA.model.modules.align.target_cache import DEFAULT_OFFSETS, future_frame_indices

DATASETS = [
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]
VIEWS = ["primary_image", "wrist_image"]


def one_minus_cos(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: [P, C] float32。パッチごとに 1 - cos を取って平均する (align_loss と同じ向き)。"""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return float((1.0 - num / den).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=2000, help="データセット・視点ごとのサンプル数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    by_k = {k: [] for k in DEFAULT_OFFSETS}
    for name in DATASETS:
        root = Path(args.cache_dir) / name
        meta = json.loads((root / "meta.json").read_text())
        lengths = np.asarray(meta["lengths"])
        starts = np.asarray(meta["starts"])
        bad = set(meta.get("bad_episodes", []))
        # 動画が壊れたエピソードは特徴が無い (ゼロ行) ので除く
        good = [i for i, e in enumerate(meta["episode_indices"]) if e not in bad]
        for view in VIEWS:
            mm = np.load(root / f"{view}.npy", mmap_mode="r")
            per_k = {k: [] for k in DEFAULT_OFFSETS}
            for _ in range(args.n_samples):
                pos = good[rng.integers(len(good))]
                step = int(rng.integers(lengths[pos]))
                x_cur = np.asarray(mm[int(starts[pos]) + step], dtype=np.float32)
                for k in DEFAULT_OFFSETS:
                    fut = future_frame_indices(int(starts[pos]), int(lengths[pos]), step, (k,))[0]  # 末尾はクランプ
                    per_k[k].append(one_minus_cos(x_cur, np.asarray(mm[fut], dtype=np.float32)))
            for k in DEFAULT_OFFSETS:
                v = np.asarray(per_k[k])
                by_k[k].append(v)
                print(f"{name} {view} k={k:2d}: {v.mean():.4f}", flush=True)

    print()
    for k in DEFAULT_OFFSETS:
        v = np.concatenate(by_k[k])
        print(f"ALL k={k:2d}: mean {v.mean():.4f}  se {v.std(ddof=1) / np.sqrt(len(v)):.4f}  n={len(v)}")
    all_v = np.concatenate([np.concatenate(by_k[k]) for k in DEFAULT_OFFSETS])
    print(f"ALL k 平均 (world_loss と同じ平均の取り方): {all_v.mean():.4f}")


if __name__ == "__main__":
    main()
