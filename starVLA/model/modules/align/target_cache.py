"""WM-Align の目標 (DINOv3 特徴) のキャッシュ。spec §4.3, §4.4。

1 サブセット (LIBERO の 1 データセット) × 1 視点につき 1 ファイル:
  <cache_dir>/<dataset_name>/<view>.npy   形状 [N_frames, 64, 768], float16 (memmap)
  <cache_dir>/<dataset_name>/meta.json    {"lengths": [...], "starts": [...], ...}
フレームの通し番号 = starts[episode_pos] + step。episode_pos は trajectory_ids の並び順の位置。
"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_OFFSETS = (4, 8, 12, 16)


def episode_starts(lengths: Sequence[int]) -> np.ndarray:
    """エピソード長の列から、通し番号の開始位置を求める。"""
    lengths = np.asarray(lengths, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(lengths)[:-1]])


def future_frame_indices(
    start: int, length: int, step: int, offsets: Sequence[int] = DEFAULT_OFFSETS, current_target: bool = False
) -> np.ndarray:
    """現在の step に対する、目標フレームの通し番号 (エピソード末尾は最終フレームにクランプ)。

    current_target=True のとき、全オフセットを 0 にする (アーム C: 現在フレームの特徴を 4 回)。
    """
    if not 0 <= step < length:
        raise ValueError(f"step {step} がエピソード長 {length} の範囲外")
    if current_target:
        steps = np.full(len(offsets), step, dtype=np.int64)
    else:
        steps = np.minimum(step + np.asarray(offsets, dtype=np.int64), length - 1)
    return start + steps


class TargetCache:
    """memmap で開いたキャッシュから、(エピソード位置, step) の目標を引く。"""

    def __init__(self, cache_dir: str | Path, dataset_name: str, views: Sequence[str]):
        root = Path(cache_dir) / dataset_name
        meta = json.loads((root / "meta.json").read_text())
        self.lengths = np.asarray(meta["lengths"], dtype=np.int64)
        self.starts = np.asarray(meta["starts"], dtype=np.int64)
        self.bad_episodes = {int(e) for e in meta.get("bad_episodes", [])}
        # episode_index → キャッシュ内の位置。学習側の trajectory_id (= episode_index) から引くのに使う
        self.episode_pos = {int(e): i for i, e in enumerate(meta.get("episode_indices", range(len(self.lengths))))}
        self.views = list(views)
        self._root = root
        self._mm = None  # worker ごとに遅延して開く (fork 後に memmap を共有しないため)

    def _open(self):
        if self._mm is None:
            self._mm = [np.load(self._root / f"{v}.npy", mmap_mode="r") for v in self.views]
        return self._mm

    def get_by_episode(
        self, episode_index: int, step: int, offsets: Sequence[int] = DEFAULT_OFFSETS, current_target: bool = False
    ) -> np.ndarray:
        """trajectory_id (= episode_index) で引く。動画が壊れたエピソードは ValueError (呼び出し側が引き直す)。"""
        if int(episode_index) in self.bad_episodes:
            raise ValueError(f"episode {episode_index} は動画が壊れていてキャッシュが無い")
        return self.get(self.episode_pos[int(episode_index)], step, offsets, current_target)

    def get(
        self, episode_pos: int, step: int, offsets: Sequence[int] = DEFAULT_OFFSETS, current_target: bool = False
    ) -> np.ndarray:
        """[V, T, 64, 768] float16 を返す。"""
        idx = future_frame_indices(
            int(self.starts[episode_pos]), int(self.lengths[episode_pos]), step, offsets, current_target
        )
        return np.stack([np.asarray(mm[idx]) for mm in self._open()])
