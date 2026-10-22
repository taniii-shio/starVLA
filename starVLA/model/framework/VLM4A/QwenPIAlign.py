"""QwenPI + WM-Align (arXiv 2610.19825 §3.2)。

VLM の第 `layer` 層 (既定 18) の視覚トークンから、MLP ヘッドで未来 4 時刻の DINOv3 特徴を予測し、
L = L_action + λ · L_world (L_world = 1 - cos) で学習する。ヘッドは学習時だけ使い、推論は QwenPI と同じ。

- framework.align.enabled=false または lambda=0 のとき、損失は QwenPI と同じ (T6)。ただしヘッドの
  パラメータは存在する (チェックポイントの形が変わらないよう、常に作る)。
- 目標は dataset 側が sample["wm_target"] ([V,T,P,C] float16) に入れる (data_cfg.wm_align)。
  アーム C は dataset 側で current_target=true にするだけで、このクラスは A と同じ。
"""

from typing import List, Optional

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.modules.align.align_head import AlignHead, align_loss, gather_visual_tokens
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenPIAlign")
class Qwen_PI_Align(Qwen_PI):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        align_cfg = self.config.framework.get("align", None) or {}
        self.align_enabled = bool(align_cfg.get("enabled", True))
        self.align_lambda = float(align_cfg.get("lambda", 0.1))
        self.align_layer = int(align_cfg.get("layer", 18))  # 1 始まり。vl_embs_list[layer-1] がその層の出力
        self.align_n_views = int(align_cfg.get("n_views", 2))
        self.align_n_patch = int(align_cfg.get("n_patch", 64))
        n_t = int(align_cfg.get("n_t", 4))
        d_feat = int(align_cfg.get("d_feat", 768))
        d_hidden = int(align_cfg.get("d_hidden", 4096))
        d_in = int(self.config.framework.qwenvl.vl_hidden_dim)
        if not 1 <= self.align_layer <= int(self.config.framework.qwenvl.num_vl_layers):
            raise ValueError(f"align.layer={self.align_layer} は 1..num_vl_layers の範囲外")
        self.align_head = AlignHead(d_in=d_in, d_hidden=d_hidden, n_t=n_t, d_feat=d_feat)
        self.image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)
        self._last_input_ids = None
        self._align_hidden = None
        logger.info(
            f"[WM-Align] enabled={self.align_enabled} λ={self.align_lambda} layer={self.align_layer} "
            f"views={self.align_n_views} patch={self.align_n_patch} head {d_in}->{d_hidden}->{n_t}x{d_feat}"
        )

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> tuple:
        # 親のメソッドは input_ids を返さないので、画像トークンの位置を取るために控えておく
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        self._last_input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        # 学習時だけ控える (推論でテンソルを持ち続けない)。vl_embs_list[k-1] = 第 k 層の出力
        if self.training and self.align_enabled:
            self._align_hidden = vl_embs_list[self.align_layer - 1]
        return vl_embs_list, attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        self._last_input_ids = None
        self._align_hidden = None
        out = super().forward(examples, **kwargs)  # 内部で _encode_vl_hidden_states を呼ぶ
        if not self.align_enabled:
            return out
        if "wm_target" not in examples[0]:
            raise KeyError("align.enabled=true だが sample に wm_target が無い。data_cfg.wm_align.enabled=true にする")
        hidden = self._align_hidden
        input_ids = self._last_input_ids
        self._align_hidden = None
        self._last_input_ids = None

        tokens = gather_visual_tokens(hidden, input_ids, self.image_token_id, self.align_n_views, self.align_n_patch)
        pred = self.align_head(tokens)  # [B,V,P,T,C]
        target = torch.from_numpy(np.stack([e["wm_target"] for e in examples])).to(pred.device)  # [B,V,T,P,C]
        world_loss = align_loss(pred, target)
        out["world_loss"] = world_loss
        out["aux_loss"] = self.align_lambda * world_loss
        return out
