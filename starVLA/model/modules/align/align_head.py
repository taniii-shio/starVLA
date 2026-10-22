"""WM-Align (arXiv 2610.19825 §3.2, §A.3) の補助ヘッドと損失。

学習時だけ使う。推論では捨てる (spec §5.3)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignHead(nn.Module):
    """VLM の視覚トークン 1 つから、4 時刻ぶんの未来の特徴を予測する 2 層 MLP。

    入力 [B, V, P, D] (視点, パッチ, VLM hidden) → 出力 [B, V, P, T, C]
    論文の値: D=2048 (Qwen3-VL-2B), hidden=4096, T=4 (オフセット数), C=768 (DINOv3 ViT-B/16)。
    """

    def __init__(self, d_in: int = 2048, d_hidden: int = 4096, n_t: int = 4, d_feat: int = 768):
        super().__init__()
        self.n_t = n_t
        self.d_feat = d_feat
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_t * d_feat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unflatten(-1, (self.n_t, self.d_feat))


def gather_visual_tokens(
    hidden: torch.Tensor, input_ids: torch.Tensor, image_token_id: int, n_views: int, n_patch: int
) -> torch.Tensor:
    """input_ids の画像トークンの位置から、視点ごとの視覚トークンを取り出す。

    hidden [B, L, D] → [B, V, P, D]。想定と数が違えば止める (T3 に相当)。
    トークンはプロンプト内で視点ごとに連続して並ぶと仮定する (順序は T3 で確かめる)。
    """
    mask = input_ids == image_token_id  # [B, L]
    counts = mask.sum(dim=1)
    if not bool((counts == n_views * n_patch).all()):
        raise ValueError(f"視覚トークン数が想定と違う: 期待 {n_views * n_patch}, 実際 {counts.tolist()}")
    return hidden[mask].view(hidden.size(0), n_views, n_patch, -1)


def align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_world = mean(1 - cos(pred, target))。spec §5.4。

    pred   [B, V, P, T, C]
    target [B, V, T, P, C] (DINOv3 キャッシュの並び。時刻が 3 番目)
    """
    target = target.permute(0, 1, 3, 2, 4)
    return (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def align_head_param_count(d_in: int = 2048, d_hidden: int = 4096, n_t: int = 4, d_feat: int = 768) -> int:
    """AlignHead のパラメータ数。spec §5.3 の 20,978,688 と照合する (T4)。"""
    return sum(p.numel() for p in AlignHead(d_in, d_hidden, n_t, d_feat).parameters())
