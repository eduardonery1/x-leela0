"""
X-Leela Stage 1: Contrastive Pre-training

Aligns Q-Former position representations with text representations
using Image-Text Contrastive (ITC) loss from BLIP-2.

This stage teaches Q-Former to extract chess-relevant features
before connecting to ChessGPT.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import chess

# Add lc0 directory to path
sys.path.insert(0, str(Path(__file__).parent / "lc0"))

from qformer import QFormer, create_qformer
from lc0_pytorch import create_lc0_model, encode_position


@dataclass
class ContrastiveConfig:
    """Contrastive pre-training configuration."""
    # Data
    data_cache: str = "/home/nery/Projects/chess/data/cache.json"
    max_comment_length: int = 128

    # Model
    lc0_weights: str = "lc0/weights"
    qformer_hidden_dim: int = 768
    qformer_layers: int = 6
    num_queries: int = 8

    # Text encoder
    text_encoder_dim: int = 768  # Match Q-Former hidden dim

    # Contrastive learning
    temperature: float = 0.07

    # Training
    batch_size: int = 32  # Larger batch for contrastive learning
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 20000

    # Logging
    log_interval: int = 100
    save_interval: int = 2000
    checkpoint_dir: str = "checkpoints"


class TextEncoder(nn.Module):
    """
    Simple text encoder for contrastive learning.

    Uses a transformer encoder to produce text embeddings
    that can be aligned with Q-Former position embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 50432,  # ChessGPT vocab
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection to shared embedding space
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to embedding.

        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) mask

        Returns:
            text_emb: (B, hidden_dim) pooled text embedding
        """
        B, seq_len = input_ids.shape

        # Embed tokens + positions
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]

        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Mean pooling (masked)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Project
        x = self.proj(x)

        return x


class QFormerWithProjection(nn.Module):
    """Q-Former with projection head for contrastive learning."""

    def __init__(self, qformer: QFormer, proj_dim: int = 768):
        super().__init__()
        self.qformer = qformer

        # Projection from Q-Former output to shared space
        self.proj = nn.Sequential(
            nn.Linear(qformer.output_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, lc0_emb: torch.Tensor) -> torch.Tensor:
        """
        Get position embedding for contrastive learning.

        Args:
            lc0_emb: (B, 64, 128) LC0 per-square embeddings

        Returns:
            pos_emb: (B, proj_dim) pooled position embedding
        """
        # Get Q-Former output
        queries = self.qformer(lc0_emb)  # (B, num_queries, output_dim)

        # Mean pool over queries
        pooled = queries.mean(dim=1)  # (B, output_dim)

        # Project to shared space
        pos_emb = self.proj(pooled)  # (B, proj_dim)

        return pos_emb

    def get_soft_prompts(self, lc0_emb: torch.Tensor) -> torch.Tensor:
        """Get soft prompts for LLM (Stage 2)."""
        return self.qformer(lc0_emb)


class ContrastiveLoss(nn.Module):
    """
    Image-Text Contrastive (ITC) loss from BLIP-2.

    Aligns position and text embeddings in shared space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        # Learnable temperature (optional, from CLIP)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(
        self,
        pos_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.

        Args:
            pos_emb: (B, dim) position embeddings
            text_emb: (B, dim) text embeddings

        Returns:
            dict with loss, pos2text_acc, text2pos_acc
        """
        # Normalize embeddings
        pos_emb = F.normalize(pos_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * pos_emb @ text_emb.T  # (B, B)

        # Labels: diagonal is positive pairs
        labels = torch.arange(len(logits), device=logits.device)

        # Bidirectional contrastive loss
        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.T, labels)
        loss = (loss_p2t + loss_t2p) / 2

        # Accuracy for logging
        with torch.no_grad():
            p2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2p_acc = (logits.argmax(dim=0) == labels).float().mean()

        return {
            'loss': loss,
            'loss_p2t': loss_p2t,
            'loss_t2p': loss_t2p,
            'acc_p2t': p2t_acc,
            'acc_t2p': t2p_acc,
            'temperature': 1.0 / logit_scale.item(),
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive pre-training."""

    def __init__(
        self,
        cache_path: str,
        tokenizer,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        with open(cache_path, 'r') as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Encode position
        board = chess.Board(item['fen'])
        position = encode_position(board)

        # Tokenize text
        tokens = self.tokenizer(
            item['comment'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'position': torch.from_numpy(position),
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        'position': torch.stack([x['position'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
    }


class ContrastiveTrainer:
    """Trainer for contrastive pre-training."""

    def __init__(self, config: ContrastiveConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self._setup_models()
        self._setup_data()
        self._setup_training()

    def _setup_models(self):
        print("\n=== Setting up models ===")

        # LC0 (frozen)
        print("Loading LC0...")
        self.lc0 = create_lc0_model(self.config.lc0_weights)
        self.lc0.eval()
        for p in self.lc0.parameters():
            p.requires_grad = False
        self.lc0 = self.lc0.to(self.device)
        print(f"  LC0 params: {sum(p.numel() for p in self.lc0.parameters()):,}")

        # Q-Former with projection (trainable)
        print("Creating Q-Former...")
        qformer = create_qformer(
            lc0_dim=self.lc0.filters,
            hidden_dim=self.config.qformer_hidden_dim,
            chessgpt_dim=2560,  # Will be used in Stage 2
            num_queries=self.config.num_queries,
            num_layers=self.config.qformer_layers,
        )
        self.qformer = QFormerWithProjection(qformer, self.config.text_encoder_dim)
        self.qformer = self.qformer.to(self.device)
        qformer_params = sum(p.numel() for p in self.qformer.parameters())
        print(f"  Q-Former params: {qformer_params:,}")

        # Text encoder (trainable)
        print("Creating text encoder...")
        self.text_encoder = TextEncoder(
            hidden_dim=self.config.text_encoder_dim,
            num_layers=4,
            max_length=self.config.max_comment_length,
        )
        self.text_encoder = self.text_encoder.to(self.device)
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        print(f"  Text encoder params: {text_params:,}")

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(self.config.temperature)
        self.contrastive_loss = self.contrastive_loss.to(self.device)

        total_trainable = qformer_params + text_params
        print(f"\n  Total trainable: {total_trainable:,} ({total_trainable/1e6:.1f}M)")

    def _setup_data(self):
        print("\n=== Setting up data ===")

        # Simple tokenizer for text encoder
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Waterhorse/chessgpt-base-v1')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = ContrastiveDataset(
            self.config.data_cache,
            self.tokenizer,
            max_length=self.config.max_comment_length,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,  # Important for contrastive learning
        )

    def _setup_training(self):
        print("\n=== Setting up training ===")

        # Optimizer for both Q-Former and text encoder
        params = list(self.qformer.parameters()) + list(self.text_encoder.parameters())
        params.append(self.contrastive_loss.logit_scale)

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        positions = batch['position'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Get LC0 embeddings (frozen)
        with torch.no_grad():
            lc0_emb = self.lc0.extract_spatial_embedding(positions)

        # Get position embeddings from Q-Former
        pos_emb = self.qformer(lc0_emb)

        # Get text embeddings
        text_emb = self.text_encoder(input_ids, attention_mask)

        # Compute contrastive loss
        loss_dict = self.contrastive_loss(pos_emb, text_emb)

        return loss_dict

    def train(self):
        print("\n" + "=" * 60)
        print("Starting contrastive pre-training")
        print("=" * 60)

        self.qformer.train()
        self.text_encoder.train()

        step = 0
        accumulated = {'loss': 0, 'acc_p2t': 0, 'acc_t2p': 0}

        pbar = tqdm(total=self.config.max_steps, desc="Contrastive Pre-training", unit="step")

        while step < self.config.max_steps:
            for batch in self.dataloader:
                # Forward
                loss_dict = self.train_step(batch)
                loss = loss_dict['loss']

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.qformer.parameters()) + list(self.text_encoder.parameters()),
                    1.0
                )
                self.optimizer.step()
                self.scheduler.step()

                # Accumulate
                accumulated['loss'] += loss.item()
                accumulated['acc_p2t'] += loss_dict['acc_p2t'].item()
                accumulated['acc_t2p'] += loss_dict['acc_t2p'].item()

                step += 1
                pbar.update(1)

                # Log
                if step % self.config.log_interval == 0:
                    avg_loss = accumulated['loss'] / self.config.log_interval
                    avg_p2t = accumulated['acc_p2t'] / self.config.log_interval
                    avg_t2p = accumulated['acc_t2p'] / self.config.log_interval
                    lr = self.scheduler.get_last_lr()[0]
                    temp = loss_dict['temperature']

                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        p2t=f"{avg_p2t:.3f}",
                        t2p=f"{avg_t2p:.3f}",
                        lr=f"{lr:.2e}"
                    )

                    accumulated = {'loss': 0, 'acc_p2t': 0, 'acc_t2p': 0}

                # Save
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

                if step >= self.config.max_steps:
                    break

        pbar.close()

        self.save_checkpoint(step, final=True)
        print("\nContrastive pre-training complete!")

    def save_checkpoint(self, step: int, final: bool = False):
        filename = "pretrain_final.pt" if final else f"pretrain_step_{step}.pt"
        path = Path(self.config.checkpoint_dir) / filename

        torch.save({
            'step': step,
            'qformer_state_dict': self.qformer.qformer.state_dict(),
            'qformer_proj_state_dict': self.qformer.proj.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'logit_scale': self.contrastive_loss.logit_scale.data,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)

        print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='X-Leela Contrastive Pre-training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--temperature', type=float, default=0.07)
    args = parser.parse_args()

    config = ContrastiveConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )

    trainer = ContrastiveTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
