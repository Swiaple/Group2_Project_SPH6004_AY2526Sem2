from __future__ import annotations

from typing import Dict

import torch

from model.cma_surv import CmaSurvModel


class MultimodalFusionModel(CmaSurvModel):
    """
    Appendix-A multimodal model.
    Keeps CMA architecture and adds explicit modality toggles for ablation and robustness.
    """

    def __init__(
        self,
        *args,
        use_text: bool = True,
        use_time: bool = True,
        use_static: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_text = bool(use_text)
        self.use_time = bool(use_time)
        self.use_static = bool(use_static)

    def forward(
        self,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_time:
            time_x = torch.zeros_like(time_x)
        if not self.use_static:
            static_x = torch.zeros_like(static_x)
        if not self.use_text:
            input_ids = torch.zeros_like(input_ids)
            attention_mask = torch.zeros_like(attention_mask)
        return super().forward(
            time_x=time_x,
            static_x=static_x,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
