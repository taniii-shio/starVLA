"""学習時に実際に渡る sample["wm_target"] が、そのサンプルの画像・step に対応しているかを確かめる。VM の starvla env で実行する。

  cd ~/starVLA   # starVLA/ 直下
  python scripts_local/check_wm_target_batch.py --data_root /data/datasets/libero --cache_dir /data/cache/dinov3_b16
  python scripts_local/check_wm_target_batch.py ... --current_target   # アーム C の経路 (n を減らしてよい)

check_dino_cache.py (T1/T2) は「キャッシュ単体」と「TargetCache の添字」までを見た。ここでは学習の経路そのもの
(get_vla_dataset → LeRobotMixtureDataset.__getitem__) を通して、次の 4 つを調べる。

C1 目標の添字: sample["wm_target"] が、meta.json から自前で計算した通し番号でキャッシュを直接引いた値と一致する
   (オフセット 4/8/12/16、末尾クランプ、--current_target なら全て現在 step)。
C2 画像との対応: sample["image"] (VLM に入る 224×224 の 2 視点) から DINOv3 特徴を再計算し、キャッシュの
   「同じ step」の特徴と一致する (視点の順序 primary/wrist、step のずれ、エピソードのずれを検出する)。
   比較として step±8 のキャッシュとのコサインも出し、一致が偶然でない (差がある) ことを見る。
C3 エピソード末尾: step + offset がエピソード長を超えるサンプルで、最終フレームにクランプされている。
C4 壊れた動画のエピソード (meta.json の bad_episodes): __getitem__ が別のサンプルを返し、wm_target がゼロ行でない。

サンプルの (dataset, trajectory_id, step) は sample_step(index) で決まる。__getitem__ も同じ関数を使うので、
同じ index で再現できる (学習は set_epoch(epoch) と index でサンプルを決める。ここでは epoch=0)。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from build_dino_cache import MODEL_ID, encode  # noqa: E402

from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402

OFFSETS = (4, 8, 12, 16)
VIEWS = ["primary_image", "wrist_image"]


def load_meta(cache_dir, name):
    root = Path(cache_dir) / name
    meta = json.loads((root / "meta.json").read_text())
    lengths = np.asarray(meta["lengths"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])  # TargetCache を使わず自前で求める
    mms = {v: np.load(root / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    pos = {int(e): i for i, e in enumerate(meta["episode_indices"])}
    return lengths, starts, pos, set(int(e) for e in meta.get("bad_episodes", [])), mms


def expected_target(lengths, starts, pos, mms, tid, step, current):
    p = pos[tid]
    L, s = int(lengths[p]), int(starts[p])
    steps = [step] * 4 if current else [min(step + o, L - 1) for o in OFFSETS]
    return np.stack([np.stack([np.asarray(mms[v][s + t]) for t in steps]) for v in VIEWS])  # [V,4,64,768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", default="./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--data_mix", default="libero_all")
    ap.add_argument("--n_samples", type=int, default=64, help="C1/C2 を調べるランダムサンプル数")
    ap.add_argument("--n_tail", type=int, default=8, help="C3 で調べる末尾サンプル数")
    ap.add_argument("--current_target", action="store_true", help="アーム C の設定 (current_target=true) で調べる")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    cfg.datasets.vla_data.video_backend = "torchvision_av"
    cfg.datasets.vla_data.wm_align = {
        "enabled": True, "cache_dir": args.cache_dir, "current_target": bool(args.current_target),
    }
    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    ds.set_epoch(0)
    n = len(ds)
    print(f"dataset: {n} samples, {len(ds.datasets)} subsets, current_target={args.current_target}")
    metas = {d.dataset_name: load_meta(args.cache_dir, d.dataset_name) for d in ds.datasets}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel

    dino = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    rng = np.random.default_rng(args.seed)

    def check_one(index, label, c2=True):
        """index の sample を取り、C1 (と C2) を調べる。結果の辞書を返す。"""
        dataset, tid, step = ds.sample_step(index)
        lengths, starts, pos, bad, mms = metas[dataset.dataset_name]
        if tid in bad:
            return {"bad": True, "tid": tid, "step": step, "dataset": dataset.dataset_name}
        sample = ds[index]
        got = np.asarray(sample["wm_target"])
        want = expected_target(lengths, starts, pos, mms, tid, step, args.current_target)
        assert got.shape == want.shape == (2, 4, 64, 768), (got.shape, want.shape)
        assert np.array_equal(got, want), f"C1 不一致: {label} {dataset.dataset_name} ep{tid} step{step}"
        out = {"bad": False, "tid": tid, "step": step, "L": int(lengths[pos[tid]]), "dataset": dataset.dataset_name}
        if c2:
            imgs = np.stack([np.array(im) for im in sample["image"]])  # [V,224,224,3]。VLM に入る画像そのもの
            assert imgs.shape[0] == 2 and imgs.shape[1:] == (224, 224, 3), imgs.shape
            z = encode(dino, imgs, dev, 8, mean, std).astype(np.float32)  # [V,64,768]
            p, s = pos[tid], None
            L, st0 = int(lengths[p]), int(starts[p])

            def cos_to(t):
                t = min(max(t, 0), L - 1)
                ref = np.stack([np.asarray(mms[v][st0 + t], dtype=np.float32) for v in VIEWS])
                num = (z * ref).sum(-1)
                den = np.linalg.norm(z, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-8
                return float((num / den).mean(-1).min())  # 2 視点のうち悪い方の、パッチ平均コサイン

            out["cos_same"] = cos_to(step)
            out["cos_pm8"] = max(cos_to(step - 8), cos_to(step + 8))
            # 視点の入れ替わり検出: primary の画像を wrist のキャッシュと比べる
            ref_sw = np.stack([np.asarray(mms[v][st0 + step], dtype=np.float32) for v in VIEWS[::-1]])
            num = (z * ref_sw).sum(-1)
            den = np.linalg.norm(z, axis=-1) * np.linalg.norm(ref_sw, axis=-1) + 1e-8
            out["cos_swapped_views"] = float((num / den).mean(-1).max())
        return out

    # C1 + C2: ランダムサンプル
    print(f"\n[C1/C2] ランダム {args.n_samples} サンプル")
    idxs = rng.choice(n, size=args.n_samples, replace=False)
    res, n_bad = [], 0
    for i, index in enumerate(idxs):
        r = check_one(int(index), "random")
        if r["bad"]:
            n_bad += 1
            continue
        res.append(r)
    same = np.array([r["cos_same"] for r in res])
    pm8 = np.array([r["cos_pm8"] for r in res])
    sw = np.array([r["cos_swapped_views"] for r in res])
    print(f"  C1 PASS: {len(res)} サンプルで wm_target が直接引いた値と完全一致 (bad episode でスキップ: {n_bad})")
    print(f"  C2 同じ step のキャッシュとのコサイン: min {same.min():.4f} / mean {same.mean():.4f}  (fp16 の誤差なら 0.99 以上)")
    print(f"     step±8 のキャッシュ (外れ): mean {pm8.mean():.4f}   視点を入れ替えた場合 (外れ): mean {sw.mean():.4f}")
    gap = (same - pm8).min()
    assert same.min() > 0.99, "C2: 画像とキャッシュの同じ step が一致しない (step・エピソード・前処理のずれ)"
    assert gap > -1e-3, "C2: step±8 の方が近いサンプルがある (時間のずれの疑い)"
    assert (same > sw + 0.01).all(), "C2: 視点を入れ替えた方が近いサンプルがある (視点の順序の疑い)"
    print("  C2 PASS")

    # C3: エピソード末尾 (step + 16 > L - 1)
    print(f"\n[C3] 末尾サンプル {args.n_tail} 個")
    tail, scan = [], 0
    for index in range(n):
        scan += 1
        dataset, tid, step = ds.sample_step(index)
        lengths, starts, pos, bad, _ = metas[dataset.dataset_name]
        if tid not in bad and step + OFFSETS[-1] > int(lengths[pos[tid]]) - 1:
            tail.append(index)
            if len(tail) >= args.n_tail:
                break
    for index in tail:
        r = check_one(index, "tail", c2=False)
        print(f"  {r['dataset'][:20]} ep{r['tid']} step {r['step']}/{r['L']}: C1 一致 (clamp 後の最終フレーム={r['L'] - 1})")
    print(f"  C3 PASS ({len(tail)} サンプル、{scan} index を走査)" if tail else "  C3: 該当サンプルが見つからない")

    # C4: 壊れた動画のエピソード
    print("\n[C4] bad episode")
    found = 0
    for index in range(n):
        dataset, tid, step = ds.sample_step(index)
        if tid in metas[dataset.dataset_name][3]:
            sample = ds[index]  # 内部で別 index に引き直される
            t = np.asarray(sample["wm_target"])
            assert np.abs(t.astype(np.float32)).sum() > 0, "bad episode のゼロ行が学習に渡っている"
            print(f"  index {index} は {dataset.dataset_name} ep{tid} (bad) → 引き直されて wm_target はゼロ行でない")
            found += 1
            if found >= 3:
                break
    print(f"  C4 PASS ({found} 件)" if found else "  C4: 該当 index が見つからない (bad_episodes が空か、サンプルに当たらない)")

    print("\n全部 PASS: wm_target は画像・step・エピソードと対応している。")


if __name__ == "__main__":
    main()
