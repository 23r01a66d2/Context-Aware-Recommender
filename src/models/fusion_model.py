"""
Module 8: Multi-Modal Fusion
Core Research Component.
Combines behavioral, content, and contextual embeddings using a
Neural Gated Attention Network that learns dynamic modality weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralGatedAttentionFusion(nn.Module):
    """
    Learns dynamic modality weights (alpha_beh, alpha_cont, alpha_ctx)
    and computes the unified fused user-context representation.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Input to gating is concatenation of all 3 modality embeddings (3 * 64 = 192)
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3) # 3 raw logits for [behavior, content, context]
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, e_beh: torch.Tensor, e_cont: torch.Tensor, e_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_beh:  [B, d] behavioral embedding
            e_cont: [B, d] candidate/content embedding
            e_ctx:  [B, d] context embedding
        Returns:
            fused_emb: [B, d] fused representation
            weights:   [B, 3] learned attention weights [alpha_beh, alpha_cont, alpha_ctx]
        """
        concat_emb = torch.cat([e_beh, e_cont, e_ctx], dim=-1) # [B, 3*d]
        gate_logits = self.gate_net(concat_emb) # [B, 3]

        # Softmax ensures weights sum to 1.0
        weights = F.softmax(gate_logits, dim=-1) # [B, 3]

        alpha_beh = weights[:, 0:1]   # [B, 1]
        alpha_cont = weights[:, 1:2]  # [B, 1]
        alpha_ctx = weights[:, 2:3]   # [B, 1]

        # Convex combination of modality representations
        weighted_sum = alpha_beh * e_beh + alpha_cont * e_cont + alpha_ctx * e_ctx
        fused_emb = self.layer_norm(weighted_sum)

        return fused_emb, weights


class SimpleConcatFusion(nn.Module):
    """Baseline Version 1: Simple Concatenation + Dense Layer"""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, e_beh: torch.Tensor, e_cont: torch.Tensor, e_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat_emb = torch.cat([e_beh, e_cont, e_ctx], dim=-1)
        fused = self.fc(concat_emb)
        # Fixed equal weights for logging
        b_size = e_beh.shape[0]
        equal_weights = torch.full((b_size, 3), 1.0 / 3.0, device=e_beh.device)
        return fused, equal_weights


if __name__ == "__main__":
    fusion = NeuralGatedAttentionFusion(64)
    b = torch.randn(4, 64)
    c = torch.randn(4, 64)
    x = torch.randn(4, 64)
    fused, w = fusion(b, c, x)
    print("Fused shape:", fused.shape)
    print("Learned weights shape:", w.shape)
    print("Sample weights:\n", w)
