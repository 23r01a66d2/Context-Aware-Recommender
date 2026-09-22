"""
Module 5: Behavioral Encoder
Projects point-in-time user behavioral history (RFM, prior engagement, category preferences)
into a dense behavioral embedding e_beh in R^64.
"""

import torch
import torch.nn as nn


class BehavioralEncoder(nn.Module):
    """
    Neural encoder for customer historical behavior.
    Input: [batch_size, 16] (8 RFM & engagement metrics + 8 category preference shares)
    Output: [batch_size, embedding_dim] (e.g. 64)
    """

    def __init__(self, input_dim: int = 16, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x_beh: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_beh: [B, 16] float tensor of historical behavioral features.
            mask: [B, 1] optional binary mask (0 for cold users, 1 for warm users).
        Returns:
            e_beh: [B, embedding_dim] behavioral representation.
        """
        e_beh = self.net(x_beh)
        if mask is not None:
            # For cold users, zero out the behavioral embedding
            e_beh = e_beh * mask
        return e_beh


if __name__ == "__main__":
    encoder = BehavioralEncoder(16, 64)
    dummy_input = torch.randn(4, 16)
    out = encoder(dummy_input)
    print("BehavioralEncoder output shape:", out.shape)
