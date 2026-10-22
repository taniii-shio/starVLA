"""LIBERO 全フレームの DINOv3 特徴を事前計算して memmap に保存する (spec §4)。

使い方 (VM、starvla env、GPU が空いているとき):
  python scripts_local/build_dino_cache.py \
      --data_root /data/datasets/libero --cache_dir /data/cache/dinov3_b16 \
      [--datasets libero_goal_no_noops_1.0.0_lerobot ...] [--limit_episodes 3]  # 動作確認

出力: <cache_dir>/<dataset>/{primary_image,wrist_image}.npy  [N_frames, 64, 768] float16
      <cache_dir>/<dataset>/meta.json                         {"lengths", "starts", ...}
エピソードの並びは LeRobot の episodes.jsonl の順 (= 学習時の trajectory_ids の順)。
フレームは学習時と同じ経路 (動画をデコード → 224×224 に resize) で作る。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
from starVLA.model.modules.align.target_cache import episode_starts

DEFAULT_DATASETS = [
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]
VIEWS = {"primary_image": "observation.images.image", "wrist_image": "observation.images.wrist_image"}
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
N_REGISTER = 4
GRID = 8


def read_episodes(root: Path) -> list[dict]:
    with open(root / "meta" / "episodes.jsonl") as f:
        return [json.loads(line) for line in f]


def resolve_video_keys(root: Path) -> dict[str, str]:
    """modality.json から視点名 → 動画キーを引く。見つからなければ既定値。"""
    path = root / "meta" / "modality.json"
    keys = dict(VIEWS)
    if path.exists():
        video = json.loads(path.read_text()).get("video", {})
        for view in VIEWS:
            orig = video.get(view, {}).get("original_key")
            if orig:
                keys[view] = orig
    return keys


@torch.no_grad()
def encode(model, frames: np.ndarray, device: str, batch: int, mean, std) -> np.ndarray:
    """frames [N,224,224,3] uint8 → [N,64,768] float16。spec §4.2 の切り出し手順。"""
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i : i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        h = model(pixel_values=x).last_hidden_state  # [n, 1+4+196, 768]
        patch = h[:, 1 + N_REGISTER :, :]
        n = patch.shape[0]
        grid = patch.reshape(n, 14, 14, -1).permute(0, 3, 1, 2)
        z = F.adaptive_avg_pool2d(grid, GRID).flatten(2).transpose(1, 2)  # [n,64,768]
        out.append(z.half().cpu().numpy())
    return np.concatenate(out)


def to_224(frames: np.ndarray) -> np.ndarray:
    """学習時の _pack_sample と同じ: PIL の resize (既定の補間) で 224×224 にする。"""
    return np.stack([np.array(Image.fromarray(f).resize((224, 224))) for f in frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit_episodes", type=int, default=0, help="動作確認用。0 なら全エピソード")
    ap.add_argument("--video_backend", default="torchvision_av")
    args = ap.parse_args()

    from transformers import AutoModel

    device = "cuda"
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    cfg = model.config
    assert getattr(cfg, "num_register_tokens", N_REGISTER) == N_REGISTER, "register 数が想定と違う"
    # 前処理の平均・標準偏差。ImageNet の値を既定とし、モデルの設定に有ればそちらを使う【要確認: VM で表示して見る】
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    print("model", MODEL_ID, "hidden", cfg.hidden_size, "registers", getattr(cfg, "num_register_tokens", None))

    for name in args.datasets:
        root = Path(args.data_root) / name
        if (Path(args.cache_dir) / name / "meta.json").exists():
            print(f"skip {name}: meta.json あり (作り直すなら {Path(args.cache_dir) / name} を消す)")
            continue
        episodes = read_episodes(root)
        if args.limit_episodes:
            episodes = episodes[: args.limit_episodes]
        lengths = [int(e["length"]) for e in episodes]
        starts = episode_starts(lengths).tolist()
        total = int(sum(lengths))
        vkeys = resolve_video_keys(root)
        out_dir = Path(args.cache_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        mms = {
            v: np.lib.format.open_memmap(out_dir / f"{v}.npy", mode="w+", dtype=np.float16, shape=(total, GRID * GRID, 768))
            for v in VIEWS
        }
        bad_episodes = []
        for pos, ep in enumerate(episodes):
            chunk = ep["episode_index"] // 1000
            ok = True
            for v in VIEWS:
                path = root / "videos" / f"chunk-{chunk:03d}" / vkeys[v] / f"episode_{ep['episode_index']:06d}.mp4"
                try:
                    frames = get_all_frames(str(path), args.video_backend)  # torchvision_av / pyav / decord は配列を返す
                except Exception as e:
                    # 元データの動画が壊れている場合。記録して飛ばす (学習側はこのエピソードを引かない)
                    print(f"BAD VIDEO {path}: {e}", flush=True)
                    ok = False
                    break
                assert len(frames) == lengths[pos], f"{name} ep{ep['episode_index']} {v}: {len(frames)} != {lengths[pos]}"
                z = encode(model, to_224(frames), device, args.batch, mean, std)
                mms[v][starts[pos] : starts[pos] + lengths[pos]] = z
            if not ok:
                bad_episodes.append(int(ep["episode_index"]))
            if pos % 20 == 0:
                print(f"{name} {pos + 1}/{len(episodes)} episodes", flush=True)
        for mm in mms.values():
            mm.flush()
        meta = {
            "lengths": lengths,
            "starts": starts,
            "episode_indices": [int(e["episode_index"]) for e in episodes],
            "bad_episodes": bad_episodes,  # 動画が壊れていて特徴を作れなかったエピソード (行はゼロのまま)
            "model": MODEL_ID,
            "pool": f"adaptive_avg_pool2d {GRID}x{GRID}, row-major",
            "frame_resize": "PIL.Image.resize((224,224)) from 256x256",
            "normalize": {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()},
            "views": list(VIEWS),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"done {name}: {total} frames, bad episodes {bad_episodes}", flush=True)


if __name__ == "__main__":
    main()
