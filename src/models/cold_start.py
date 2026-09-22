"""
Module 9: Cold-Start Handling & Adaptive Gating
Core Research Component.
Dynamically detects cold-start interactions (N_hist < K) and adjusts modality weights
to suppress noisy uninformative behavioral history while allowing content and context to dominate.
"""

import torch
import torch.nn as nn


class ColdStartAdaptiveGate(nn.Module):
    """
    Applies adaptive gating constraint based on historical interaction count N_hist.
    Configurable threshold K (evaluated for K in {1, 2, 3, 5}).
    """

    def __init__(self, primary_threshold: int = 3, behavior_penalty: float = 0.01, embedding_dim: int = 64):
        super().__init__()
        self.threshold = primary_threshold
        self.behavior_penalty = behavior_penalty
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def set_threshold(self, k: int):
        """Allows dynamic experimentation with different thresholds K."""
        self.threshold = k

    def forward(self, e_beh: torch.Tensor, e_cont: torch.Tensor, e_ctx: torch.Tensor,
                learned_weights: torch.Tensor, hist_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            e_beh:           [B, d] behavioral embedding
            e_cont:          [B, d] content embedding
            e_ctx:           [B, d] context embedding
            learned_weights: [B, 3] raw softmax gating weights [alpha_beh, alpha_cont, alpha_ctx]
            hist_counts:     [B] integer tensor of point-in-time prior interactions
        Returns:
            adjusted_fused_emb: [B, d]
            adaptive_weights:   [B, 3]
            cold_mask:          [B, 1] boolean mask (True = cold-start, False = warm-start)
        """
        device = learned_weights.device
        b_size = learned_weights.shape[0]

        # Cold start boolean mask: N_hist < threshold
        cold_mask = (hist_counts < self.threshold).view(b_size, 1).to(device) # [B, 1]

        alpha_beh = learned_weights[:, 0:1]
        alpha_cont = learned_weights[:, 1:2]
        alpha_ctx = learned_weights[:, 2:3]

        # Compute suppressed behavioral weight for cold cases
        suppressed_beh = alpha_beh * self.behavior_penalty
        removed_mass = alpha_beh - suppressed_beh

        # Re-allocate removed mass proportionally between content and context
        sum_cc = torch.clamp(alpha_cont + alpha_ctx, min=1e-7)
        boost_cont = removed_mass * (alpha_cont / sum_cc)
        boost_ctx = removed_mass * (alpha_ctx / sum_cc)

        cold_alpha_beh = suppressed_beh
        cold_alpha_cont = alpha_cont + boost_cont
        cold_alpha_ctx = alpha_ctx + boost_ctx

        # Select between warm (original learned) and cold (adjusted) weights
        final_alpha_beh = torch.where(cold_mask, cold_alpha_beh, alpha_beh)
        final_alpha_cont = torch.where(cold_mask, cold_alpha_cont, alpha_cont)
        final_alpha_ctx = torch.where(cold_mask, cold_alpha_ctx, alpha_ctx)

        adaptive_weights = torch.cat([final_alpha_beh, final_alpha_cont, final_alpha_ctx], dim=-1)

        # Compute adjusted fused embedding
        weighted_sum = (final_alpha_beh * e_beh +
                        final_alpha_cont * e_cont +
                        final_alpha_ctx * e_ctx)
        adjusted_fused_emb = self.layer_norm(weighted_sum)

        return adjusted_fused_emb, adaptive_weights, cold_mask


if __name__ == "__main__":
    gate = ColdStartAdaptiveGate(primary_threshold=3, behavior_penalty=0.01)
    b = torch.randn(4, 64)
    c = torch.randn(4, 64)
    x = torch.randn(4, 64)
    weights = torch.tensor([
        [0.60, 0.20, 0.20], # Warm
        [0.50, 0.25, 0.25], # Cold (N=0)
        [0.55, 0.25, 0.20], # Cold (N=1)
        [0.70, 0.15, 0.15]  # Warm (N=5)
    ])
    hist = torch.tensor([4, 0, 1, 5])
    fused, adj_w, cold_m = gate(b, c, x, weights, hist)
    print("Cold mask:\n", cold_m.view(-1))
    print("Adjusted weights:\n", adj_w)
