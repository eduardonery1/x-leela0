"""
Q-Former: Querying Transformer Bridge for X-Leela

Bridges LC0 chess embeddings to ChessGPT soft prompts using
learnable query tokens and cross-attention, inspired by BLIP-2.

Architecture:
    LC0 embeddings (B, 64, 128)
        → project to query dim
        → cross-attention with learnable queries
        → soft prompts (B, 8, 2560)
        → ChessGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QFormerLayer(nn.Module):
    """
    Single Q-Former transformer layer with:
    1. Self-attention among query tokens
    2. Cross-attention from queries to LC0 embeddings
    3. Feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ffn_dim = ffn_dim or hidden_dim * 4

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention (queries attend to LC0 embeddings)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        lc0_features: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        lc0_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, hidden_dim) - learnable query tokens
            lc0_features: (B, 64, hidden_dim) - projected LC0 embeddings
            query_mask: Optional attention mask for queries
            lc0_mask: Optional attention mask for LC0 features

        Returns:
            Updated queries: (B, num_queries, hidden_dim)
        """
        # Self-attention among queries
        residual = queries
        queries = self.self_attn_norm(queries)
        queries_attn, _ = self.self_attn(
            queries, queries, queries,
            key_padding_mask=query_mask
        )
        queries = residual + queries_attn

        # Cross-attention: queries attend to LC0 features
        residual = queries
        queries = self.cross_attn_norm(queries)
        queries_cross, _ = self.cross_attn(
            queries, lc0_features, lc0_features,
            key_padding_mask=lc0_mask
        )
        queries = residual + queries_cross

        # Feed-forward
        residual = queries
        queries = self.ffn_norm(queries)
        queries = residual + self.ffn(queries)

        return queries


class QFormer(nn.Module):
    """
    Q-Former bridge module for X-Leela.

    Transforms LC0 chess position embeddings into soft prompts
    for ChessGPT using learnable query tokens and cross-attention.

    Uses a smaller internal dimension for efficiency, then projects
    to the target LLM dimension at the end.

    Input:  LC0 embeddings (B, 64, 128) - per-square features
    Output: Soft prompts (B, num_queries, output_dim) - for ChessGPT
    """

    def __init__(
        self,
        lc0_dim: int = 128,            # LC0 per-square embedding dimension
        hidden_dim: int = 768,          # Internal transformer dimension (smaller for efficiency)
        output_dim: int = 2560,         # ChessGPT hidden dimension
        num_queries: int = 8,           # Number of soft prompt tokens
        num_layers: int = 6,            # Number of transformer layers
        num_heads: int = 8,             # Attention heads
        ffn_dim: Optional[int] = None,  # FFN intermediate dim (default: 4x hidden)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lc0_dim = lc0_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Learnable query tokens (in internal dimension)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Project LC0 embeddings to hidden dimension
        self.lc0_proj = nn.Sequential(
            nn.Linear(lc0_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Optional: learnable positional embeddings for the 64 squares
        self.lc0_pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Project to output dimension (ChessGPT)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        lc0_embeddings: torch.Tensor,
        lc0_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transform LC0 embeddings into soft prompts for ChessGPT.

        Args:
            lc0_embeddings: (B, 64, 128) LC0 per-square embeddings
            lc0_mask: Optional (B, 64) mask for LC0 squares

        Returns:
            soft_prompts: (B, num_queries, 2560) ready for ChessGPT
        """
        batch_size = lc0_embeddings.shape[0]

        # Project LC0 to hidden dimension and add positional embeddings
        lc0_features = self.lc0_proj(lc0_embeddings)  # (B, 64, hidden_dim)
        lc0_features = lc0_features + self.lc0_pos_embed

        # Expand learnable queries for batch
        queries = self.queries.expand(batch_size, -1, -1)  # (B, num_queries, hidden_dim)

        # Pass through transformer layers
        for layer in self.layers:
            queries = layer(queries, lc0_features, lc0_mask=lc0_mask)

        # Final normalization and project to output dimension
        queries = self.final_norm(queries)
        soft_prompts = self.output_proj(queries)

        return soft_prompts

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class XLeelaModel(nn.Module):
    """
    Full X-Leela pipeline: LC0 → Q-Former → ChessGPT

    LC0 and ChessGPT are frozen; only Q-Former is trained.
    """

    def __init__(
        self,
        lc0_model: nn.Module,
        chessgpt_model: nn.Module,
        qformer: QFormer,
        freeze_endpoints: bool = True,
    ):
        super().__init__()
        self.lc0 = lc0_model
        self.chessgpt = chessgpt_model
        self.qformer = qformer

        if freeze_endpoints:
            self._freeze_endpoints()

    def _freeze_endpoints(self):
        """Freeze LC0 and ChessGPT parameters."""
        for param in self.lc0.parameters():
            param.requires_grad = False
        for param in self.chessgpt.parameters():
            param.requires_grad = False
        self.lc0.eval()
        self.chessgpt.eval()

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass.

        Args:
            positions: (B, 112, 8, 8) LC0 input planes
            input_ids: (B, seq_len) target token IDs
            attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) labels for loss computation

        Returns:
            CausalLMOutput with loss and logits
        """
        # Get LC0 embeddings (frozen)
        with torch.no_grad():
            lc0_emb = self.lc0.extract_spatial_embedding(positions)  # (B, 64, 128)

        # Q-Former produces soft prompts (trainable)
        soft_prompts = self.qformer(lc0_emb)  # (B, 8, 2560)

        # Get text embeddings from ChessGPT
        text_embeds = self.chessgpt.get_input_embeddings()(input_ids)  # (B, seq_len, 2560)

        # Concatenate: [soft_prompts, text_embeds]
        combined_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        # Adjust attention mask for soft prompts
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                attention_mask.shape[0], self.qformer.num_queries,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        # Adjust labels (shift for soft prompt tokens)
        if labels is not None:
            # Pad labels with -100 (ignore) for soft prompt positions
            label_padding = torch.full(
                (labels.shape[0], self.qformer.num_queries),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([label_padding, labels], dim=1)
        else:
            combined_labels = None

        # Forward through ChessGPT
        outputs = self.chessgpt(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return outputs

    def generate(
        self,
        positions: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Generate commentary for chess positions.

        Args:
            positions: (B, 112, 8, 8) LC0 input planes
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated token IDs
        """
        # Get LC0 embeddings
        with torch.no_grad():
            lc0_emb = self.lc0.extract_spatial_embedding(positions)

        # Q-Former produces soft prompts
        soft_prompts = self.qformer(lc0_emb)

        # Generate from soft prompts
        outputs = self.chessgpt.generate(
            inputs_embeds=soft_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.chessgpt.config.eos_token_id,
            **kwargs
        )

        return outputs


# Utility functions

def create_qformer(
    lc0_dim: int = 128,
    hidden_dim: int = 768,
    chessgpt_dim: int = 2560,
    num_queries: int = 8,
    num_layers: int = 6,
) -> QFormer:
    """Create a Q-Former with default settings (~60M params)."""
    return QFormer(
        lc0_dim=lc0_dim,
        hidden_dim=hidden_dim,
        output_dim=chessgpt_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=8,
        dropout=0.1,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Q-Former Test")
    print("=" * 60)

    # Create Q-Former
    qformer = create_qformer()

    # Count parameters
    num_params = qformer.get_num_params()
    print(f"\nQ-Former parameters: {num_params:,}")
    print(f"  (~{num_params / 1e6:.1f}M)")

    # Test forward pass
    batch_size = 4
    lc0_emb = torch.randn(batch_size, 64, 128)

    print(f"\nInput shape: {lc0_emb.shape}")

    soft_prompts = qformer(lc0_emb)
    print(f"Output shape: {soft_prompts.shape}")
    print(f"Expected: ({batch_size}, 8, 2560)")

    # Test gradient flow
    print("\nTesting gradient flow...")
    soft_prompts.sum().backward()

    grad_norm = sum(p.grad.norm().item() for p in qformer.parameters() if p.grad is not None)
    print(f"Total gradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 60)
    print("Q-Former ready!")
    print("=" * 60)
