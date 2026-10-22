"""DINOv3 キャッシュの照合 (T1/T2)。VM の starvla env で実行する。

  python scripts_local/check_dino_cache.py --data_root /data/datasets/libero \
      --cache_dir /data/cache/dino_test --dataset libero_goal_no_noops_1.0.0_lerobot

T1: 動画から数フレームを取り出し、HF の AutoImageProcessor (build_dino_cache.py とは別の前処理経路) で
    特徴を再計算してキャッシュと比べる (コサイン類似度の最小値)。
T2: TargetCache の添字 (未来 t+4/8/12/16、末尾クランプ、アーム C) が、キャッシュを直接引いた値と一致する。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
from starVLA.model.modules.align.target_cache import TargetCache

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VIEWS = {"primary_image": "observation.images.image", "wrist_image": "observation.images.wrist_image"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dataset", default="libero_goal_no_noops_1.0.0_lerobot")
    ap.add_argument("--n_frames", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    root = Path(args.data_root) / args.dataset
    cache_root = Path(args.cache_dir) / args.dataset
    meta = json.loads((cache_root / "meta.json").read_text())
    lengths, starts, eps = meta["lengths"], meta["starts"], meta["episode_indices"]

    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)

    modality = json.loads((root / "meta" / "modality.json").read_text()).get("video", {})
    worst = 1.0
    for view, default_key in VIEWS.items():
        key = modality.get(view, {}).get("original_key") or default_key
        cache = np.load(cache_root / f"{view}.npy", mmap_mode="r")
        ep = eps[0]
        path = root / "videos" / f"chunk-{ep // 1000:03d}" / key / f"episode_{ep:06d}.mp4"
        frames = get_all_frames(str(path), "torchvision_av")
        picks = np.linspace(0, lengths[0] - 1, args.n_frames).astype(int)
        for t in picks:
            img = Image.fromarray(frames[t]).resize((224, 224))
            # 別経路: processor は resize/crop を勝手に足さないよう 224 固定・crop 無しで正規化だけ使う
            x = proc(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].cuda()
            with torch.no_grad():
                h = model(pixel_values=x).last_hidden_state[:, 1 + 4 :, :]
            grid = h.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
            ref = F.adaptive_avg_pool2d(grid, 8).flatten(2).transpose(1, 2)[0]  # [64,768]
            got = torch.from_numpy(np.asarray(cache[starts[0] + t], dtype=np.float32)).cuda()
            cos = F.cosine_similarity(ref, got, dim=-1).min().item()
            worst = min(worst, cos)
            print(f"T1 {view} frame {t}: min cos {cos:.5f}")
    print(f"T1 最小 cos = {worst:.5f} (fp16 保存の誤差なら 0.999 以上のはず)")
    assert worst > 0.995, "キャッシュと再計算が一致しない"

    # T2: 添字
    tc = TargetCache(args.cache_dir, args.dataset, list(VIEWS))
    pos, L = 0, lengths[0]
    for step in [0, L // 2, L - 5, L - 1]:
        got = tc.get(pos, step)  # [V,4,64,768]
        want_idx = [starts[pos] + min(step + o, L - 1) for o in (4, 8, 12, 16)]
        for vi, view in enumerate(VIEWS):
            mm = np.load(cache_root / f"{view}.npy", mmap_mode="r")
            assert np.array_equal(got[vi], np.stack([mm[i] for i in want_idx])), (view, step)
        cur = tc.get(pos, step, current_target=True)
        assert all(np.array_equal(cur[0, i], cur[0, 0]) for i in range(4))
    print("T2 添字 PASS")


if __name__ == "__main__":
    main()
