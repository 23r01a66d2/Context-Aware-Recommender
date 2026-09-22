"""
Module 7: Context Encoder
Encodes recommendation session context (device, channel, season, cyclical temporal, location)
into a dense context embedding e_ctx in R^64.
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Neural encoder for recommendation session context.
    Input: [batch_size, 21]
    Output: [batch_size, embedding_dim] (e.g. 64)
    """

    def __init__(self, input_dim: int = 21, embedding_dim: int = 64, dropout: float = 0.2):
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

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ctx: [B, 21] float tensor of session and temporal context.
        Returns:
            e_ctx: [B, embedding_dim] context representation.
        """
        return self.net(x_ctx)


if __name__ == "__main__":
    encoder = ContextEncoder(21, 64)
    dummy_input = torch.randn(4, 21)
    out = encoder(dummy_input)
    print("ContextEncoder output shape:", out.shape)
