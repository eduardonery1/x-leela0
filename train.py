"""
X-Leela Training Script

Trains the Q-Former bridge between LC0 and ChessGPT.
Optimized for GTX 1650 (4GB VRAM) with gradient checkpointing and CPU offloading.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess

# Add lc0 directory to path
sys.path.insert(0, str(Path(__file__).parent / "lc0"))

from qformer import QFormer, create_qformer
from lc0_pytorch import create_lc0_model, encode_position
from pretrain_contrastive import ContrastiveConfig  # For loading pretrain checkpoints


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_cache: str = "/home/nery/Projects/chess/data/cache.json"
    max_comment_length: int = 128

    # Model
    lc0_weights: str = "lc0/weights"
    qformer_hidden_dim: int = 768
    qformer_layers: int = 6
    num_queries: int = 8

    # Pre-trained weights (from contrastive pre-training)
    pretrain_checkpoint: Optional[str] = None

    # Training
    batch_size: int = 4
    gradient_accumulation: int = 8  # Effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000

    # Memory optimization
    use_amp: bool = True  # Mixed precision
    gradient_checkpointing: bool = True
    offload_chessgpt: bool = True  # Keep ChessGPT on CPU, move per-batch

    # Logging
    log_interval: int = 100
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"


class CachedChessDataset(Dataset):
    """Dataset that loads from pre-cached JSON file."""

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

        print(f"Loaded {len(self.data)} samples from cache")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Encode chess position
        board = chess.Board(item['fen'])
        position = encode_position(board)

        # Tokenize comment
        tokens = self.tokenizer(
            item['comment'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'position': torch.from_numpy(position),  # (112, 8, 8)
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        'position': torch.stack([x['position'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
    }


class XLeelaTrainer:
    """Trainer for X-Leela Q-Former."""

    def __init__(self, config: TrainingConfig):
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
        """Initialize all models."""
        print("\n=== Setting up models ===")

        # LC0 (frozen, on CPU for memory)
        print("Loading LC0...")
        self.lc0 = create_lc0_model(self.config.lc0_weights)
        self.lc0.eval()
        for p in self.lc0.parameters():
            p.requires_grad = False
        print(f"  LC0 params: {sum(p.numel() for p in self.lc0.parameters()):,}")

        # ChessGPT (frozen)
        print("Loading ChessGPT...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained('Waterhorse/chessgpt-base-v1')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load in float16 to save memory
        self.chessgpt = AutoModelForCausalLM.from_pretrained(
            'Waterhorse/chessgpt-base-v1',
            torch_dtype=torch.float16,
        )
        self.chessgpt.eval()
        for p in self.chessgpt.parameters():
            p.requires_grad = False

        if self.config.gradient_checkpointing:
            self.chessgpt.gradient_checkpointing_enable()

        print(f"  ChessGPT params: {sum(p.numel() for p in self.chessgpt.parameters()):,}")

        # Q-Former (trainable)
        print("Creating Q-Former...")
        self.qformer = create_qformer(
            lc0_dim=self.lc0.filters,  # 128 for this network
            hidden_dim=self.config.qformer_hidden_dim,
            chessgpt_dim=self.chessgpt.config.hidden_size,  # 2560
            num_queries=self.config.num_queries,
            num_layers=self.config.qformer_layers,
        )
        print(f"  Q-Former params: {self.qformer.get_num_params():,}")

        # Load pre-trained weights if available
        if self.config.pretrain_checkpoint:
            print(f"Loading pre-trained weights from {self.config.pretrain_checkpoint}")
            ckpt = torch.load(self.config.pretrain_checkpoint, map_location='cpu', weights_only=False)
            self.qformer.load_state_dict(ckpt['qformer_state_dict'])
            print("  Pre-trained weights loaded!")

        # Move models to appropriate devices
        # LC0 (~4M params, ~16MB) and Q-Former (~60M, ~240MB) fit on GPU
        # ChessGPT (2.8B, ~5.5GB) stays on CPU
        self.lc0 = self.lc0.to(self.device)
        self.qformer = self.qformer.to(self.device)
        self.chessgpt = self.chessgpt.to('cpu')
        print(f"  LC0 + Q-Former on {self.device}, ChessGPT on CPU")

    def _setup_data(self):
        """Setup data loading."""
        print("\n=== Setting up data ===")

        self.dataset = CachedChessDataset(
            self.config.data_cache,
            self.tokenizer,
            max_length=self.config.max_comment_length,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def _setup_training(self):
        """Setup optimizer and scheduler."""
        print("\n=== Setting up training ===")

        self.optimizer = torch.optim.AdamW(
            self.qformer.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        import math
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single forward step returning loss.

        Strategy for GTX 1650 (4GB):
        - LC0: GPU (4M params, ~16MB)
        - Q-Former: GPU (60M params, ~240MB)
        - ChessGPT: CPU (2.8B params, won't fit on GPU)
        - Gradients flow through Q-Former only
        """
        positions = batch['position'].to(self.device)  # (B, 112, 8, 8)
        input_ids = batch['input_ids']  # Keep on CPU for ChessGPT
        attention_mask = batch['attention_mask']

        # Get LC0 embeddings (on GPU)
        with torch.no_grad():
            lc0_emb = self.lc0.extract_spatial_embedding(positions)  # (B, 64, 128)

        # Q-Former on GPU produces soft prompts
        soft_prompts = self.qformer(lc0_emb)  # (B, num_queries, 2560)

        # Move soft prompts to CPU for ChessGPT, keeping gradient connection
        soft_prompts_cpu = soft_prompts.to('cpu')

        # Get text embeddings (ChessGPT embedding layer on CPU)
        with torch.no_grad():
            text_embeds = self.chessgpt.get_input_embeddings()(input_ids)

        # Convert soft_prompts to match ChessGPT dtype
        soft_prompts_cpu = soft_prompts_cpu.to(text_embeds.dtype)

        # Concatenate soft prompts + text (on CPU)
        combined_embeds = torch.cat([soft_prompts_cpu, text_embeds], dim=1)

        # Create combined attention mask
        soft_mask = torch.ones(
            attention_mask.shape[0], self.config.num_queries,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # Create labels (shifted, with -100 for soft prompt positions)
        labels = input_ids.clone()
        label_padding = torch.full(
            (labels.shape[0], self.config.num_queries),
            -100, dtype=labels.dtype
        )
        combined_labels = torch.cat([label_padding, labels], dim=1)

        # Forward through ChessGPT (on CPU)
        outputs = self.chessgpt(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        # Loss is on CPU, gradients will flow back through soft_prompts to Q-Former on GPU
        return outputs.loss

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

        self.qformer.train()

        step = 0
        accumulated_loss = 0.0

        while step < self.config.max_steps:
            for batch in self.dataloader:
                # Forward pass (ChessGPT on CPU, Q-Former on GPU)
                # Note: AMP only applies to GPU operations (Q-Former)
                loss = self.forward_step(batch)
                loss = loss / self.config.gradient_accumulation

                # Backward pass - gradients flow from CPU loss through GPU Q-Former
                loss.backward()

                accumulated_loss += loss.item()

                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.qformer.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                step += 1

                # Logging
                if step % self.config.log_interval == 0:
                    avg_loss = accumulated_loss / self.config.log_interval
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {step}/{self.config.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    accumulated_loss = 0.0

                # Save checkpoint
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

                if step >= self.config.max_steps:
                    break

        # Save final checkpoint
        self.save_checkpoint(step, final=True)
        print("\nTraining complete!")

    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        filename = "final.pt" if final else f"step_{step}.pt"
        path = Path(self.config.checkpoint_dir) / filename

        torch.save({
            'step': step,
            'qformer_state_dict': self.qformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)

        print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train X-Leela Q-Former')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to contrastive pre-training checkpoint')
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        pretrain_checkpoint=args.pretrain,
    )

    trainer = XLeelaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
