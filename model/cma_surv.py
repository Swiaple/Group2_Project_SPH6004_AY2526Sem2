from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class CmaSurvModel(nn.Module):
    def __init__(
        self,
        time_input_dim: int,
        static_input_dim: int,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        self.time_encoder = nn.LSTM(
            input_size=time_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.model_dim = lstm_hidden_dim * 2
        self.time_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.text_encoder = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            # We consume CLS from last_hidden_state, not pooler_output.
            # Keep pooler frozen to avoid DDP unused-parameter reduction errors.
            pooler = getattr(self.text_encoder, "pooler", None)
            if pooler is not None:
                for param in pooler.parameters():
                    param.requires_grad = False

        text_hidden_dim = int(self.text_encoder.config.hidden_size)
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.static_proj = nn.Sequential(
            nn.Linear(static_input_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )

        self.gate_net = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Survival head: [B, 3 bins, 3 classes(none/discharge/death)].
        self.survival_head = nn.Linear(self.model_dim, 3 * 3)
        # Task2 head keeps compatibility with existing task2 reporting.
        self.task2_head = nn.Linear(self.model_dim, 1)

    def _make_time_key_padding_mask(self, time_x: torch.Tensor) -> torch.Tensor:
        # Row is padding if all channels are zero (post-normalization missing window rows).
        mask = (time_x.abs().sum(dim=-1) < 1e-8)
        all_masked = mask.all(dim=1)
        if torch.any(all_masked):
            mask[all_masked, -1] = False
        return mask

    def forward(
        self,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        time_seq, _ = self.time_encoder(time_x)
        time_seq = self.time_proj(time_seq)
        time_pool = time_seq.mean(dim=1)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0, :]
        text_vec = self.text_proj(text_cls)
        static_vec = self.static_proj(static_x)

        query = text_vec.unsqueeze(1)
        key_padding_mask = self._make_time_key_padding_mask(time_x)
        attn_vec, attn_weights = self.cross_attn(
            query=query,
            key=time_seq,
            value=time_seq,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_vec = attn_vec.squeeze(1)
        attn_weights = attn_weights.squeeze(1)

        gate_in = torch.cat([text_vec, attn_vec, static_vec], dim=-1)
        gate = self.gate_net(gate_in)
        gated_cross = gate * attn_vec + (1.0 - gate) * text_vec

        fused_vec = self.fusion(torch.cat([gated_cross, static_vec, time_pool], dim=-1))
        survival_logits = self.survival_head(fused_vec).view(-1, 3, 3)
        task2_logits = self.task2_head(fused_vec).squeeze(-1)

        out = {
            "survival_logits": survival_logits,
            "task2_logits": task2_logits,
        }
        if return_attention:
            out["attention_weights"] = attn_weights
        return out
