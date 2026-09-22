"""
Module 6: Content Encoder (Product Tower)
Encodes candidate product features (product ID embedding, category, price, discount)
into a dense content embedding e_cont in R^64.
"""

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """
    Neural encoder for candidate product metadata.
    Input: [batch_size, 12] (col 0: product_id, cols 1-8: category one-hot, cols 9-11: scaled continuous)
    Output: [batch_size, embedding_dim] (e.g. 64)
    """

    def __init__(self, num_products: int = 1000, id_emb_dim: int = 32,
                 meta_input_dim: int = 11, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.id_embedding = nn.Embedding(num_products, id_emb_dim, padding_idx=0)
        
        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )

        self.proj = nn.Sequential(
            nn.Linear(id_emb_dim + 32, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cont: [B, 12] where col 0 is product_id (int), cols 1:12 are float metadata.
        Returns:
            e_cont: [B, embedding_dim] content representation.
        """
        prod_ids = x_cont[:, 0].long()
        meta_features = x_cont[:, 1:]

        id_emb = self.id_embedding(prod_ids)
        meta_emb = self.meta_net(meta_features)

        combined = torch.cat([id_emb, meta_emb], dim=-1)
        e_cont = self.proj(combined)
        return e_cont


if __name__ == "__main__":
    encoder = ContentEncoder(num_products=1000, embedding_dim=64)
    dummy_input = torch.zeros(4, 12)
    dummy_input[:, 0] = torch.tensor([10, 25, 100, 500])
    dummy_input[:, 1:] = torch.randn(4, 11)
    out = encoder(dummy_input)
    print("ContentEncoder output shape:", out.shape)
