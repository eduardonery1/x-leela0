"""
X-Leela Training Script

Trains the Q-Former bridge between LC0 and an LLM decoder.
Optimized for GTX 1650 (4GB VRAM) with gradient checkpointing and CPU offloading.
"""

import os
import sys
import json
import argparse
import time
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
    llm_model: str = "gpt2"  # Small LLM that fits on GPU (~500MB)

    # Pre-trained weights (from contrastive pre-training)
    pretrain_checkpoint: Optional[str] = None

    # Training
    batch_size: int = 4
    gradient_accumulation: int = 8  # Effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000

    # Validation
    val_split: float = 0.0001  # 5% for validation
    val_interval: int = 500  # Validate every N steps

    # Memory optimization
    use_amp: bool = True  # Mixed precision
    gradient_checkpointing: bool = True
    offload_llm: bool = False  # Not used with small LLMs

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

        # LLM decoder (frozen)
        print(f"Loading LLM: {self.config.llm_model}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=torch.float16,
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        if self.config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        print(f"  LLM params: {sum(p.numel() for p in self.llm.parameters()):,}")
        print(f"  LLM hidden_size: {self.llm.config.hidden_size}")

        # Q-Former (trainable)
        print("Creating Q-Former...")
        self.qformer = create_qformer(
            lc0_dim=self.lc0.filters,  # 128 for this network
            hidden_dim=self.config.qformer_hidden_dim,
            chessgpt_dim=self.llm.config.hidden_size,  # 768 for gpt2, 2560 for chessgpt
            num_queries=self.config.num_queries,
            num_layers=self.config.qformer_layers,
        )
        print(f"  Q-Former params: {self.qformer.get_num_params():,}")

        # Load pre-trained weights if available
        if self.config.pretrain_checkpoint:
            print(f"Loading pre-trained weights from {self.config.pretrain_checkpoint}")
            ckpt = torch.load(self.config.pretrain_checkpoint, map_location='cpu', weights_only=False)

            # Check for dimension mismatch (pretrained with different LLM)
            pretrain_state = ckpt['qformer_state_dict']
            current_state = self.qformer.state_dict()

            # Filter out incompatible weights (different output dimensions)
            compatible_state = {}
            skipped = []
            for k, v in pretrain_state.items():
                if k in current_state and v.shape == current_state[k].shape:
                    compatible_state[k] = v
                else:
                    skipped.append(k)

            if skipped:
                print(f"  Skipping incompatible weights (LLM dim mismatch): {skipped}")

            self.qformer.load_state_dict(compatible_state, strict=False)
            print(f"  Loaded {len(compatible_state)}/{len(pretrain_state)} pretrained weights")

        # Move all models to GPU
        self.lc0 = self.lc0.to(self.device)
        self.qformer = self.qformer.to(self.device)
        self.llm = self.llm.to(self.device)
        print(f"  All models on {self.device}")

    def _setup_data(self):
        """Setup data loading with train/val split."""
        print("\n=== Setting up data ===")

        full_dataset = CachedChessDataset(
            self.config.data_cache,
            self.tokenizer,
            max_length=self.config.max_comment_length,
        )

        # Split into train and validation
        val_size = int(len(full_dataset) * self.config.val_split)
        train_size = len(full_dataset) - val_size

        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
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

    def forward_step(self, batch: Dict[str, torch.Tensor], timing: bool = False) -> torch.Tensor:
        """Single forward step returning loss.

        All models on GPU. Gradients flow through Q-Former only.
        """
        t0 = time.time() if timing else None

        positions = batch['position'].to(self.device)  # (B, 112, 8, 8)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Get LC0 embeddings (on GPU)
        with torch.no_grad():
            lc0_emb = self.lc0.extract_spatial_embedding(positions)  # (B, 64, 128)

        if timing:
            torch.cuda.synchronize()
            print(f"  LC0: {time.time() - t0:.2f}s", end="")
            t0 = time.time()

        # Q-Former on GPU produces soft prompts
        soft_prompts = self.qformer(lc0_emb)  # (B, num_queries, 2560)

        if timing:
            torch.cuda.synchronize()
            print(f" | Q-Former: {time.time() - t0:.2f}s", end="")
            t0 = time.time()

        # Get text embeddings from LLM
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Convert soft_prompts to match LLM dtype
        soft_prompts = soft_prompts.to(text_embeds.dtype)

        # Concatenate soft prompts + text (on GPU)
        combined_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        # Create combined attention mask
        soft_mask = torch.ones(
            attention_mask.shape[0], self.config.num_queries,
            dtype=attention_mask.dtype, device=self.device
        )
        combined_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # Create labels (shifted, with -100 for soft prompt positions)
        labels = input_ids.clone()
        label_padding = torch.full(
            (labels.shape[0], self.config.num_queries),
            -100, dtype=labels.dtype, device=self.device
        )
        combined_labels = torch.cat([label_padding, labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        if timing:
            print(f" | LLM: {time.time() - t0:.2f}s")

        # Loss is on CPU, gradients will flow back through soft_prompts to Q-Former on GPU
        return outputs.loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.qformer.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for batch in pbar:
            loss = self.forward_step(batch)
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        self.qformer.train()
        return total_loss / max(num_batches, 1)

    def train(self):
        """Main training loop with epoch-based progress bars."""
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

        self.qformer.train()

        step = 0
        epoch = 0
        best_val_loss = float('inf')

        # Time first step to show breakdown
        print("\nTiming first step (this shows where time is spent):")
        for batch in self.train_loader:
            loss = self.forward_step(batch, timing=True)
            break
        print("")

        # Calculate epochs needed
        steps_per_epoch = self.config.val_interval
        total_epochs = (self.config.max_steps + steps_per_epoch - 1) // steps_per_epoch

        print(f"Training for {self.config.max_steps} steps ({total_epochs} epochs of {steps_per_epoch} steps)")
        print("")

        while step < self.config.max_steps:
            epoch += 1
            epoch_steps = min(steps_per_epoch, self.config.max_steps - step)
            epoch_loss = 0.0
            epoch_batches = 0

            # Training epoch
            pbar = tqdm(total=epoch_steps, desc=f"Epoch {epoch}/{total_epochs}", unit="step")
            train_iter = iter(self.train_loader)

            for _ in range(epoch_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # Forward pass
                loss = self.forward_step(batch)
                loss_value = loss.item()
                loss = loss / self.config.gradient_accumulation

                # Backward pass
                loss.backward()

                epoch_loss += loss_value
                epoch_batches += 1

                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.qformer.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                step += 1
                lr = self.scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / epoch_batches
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                # Save periodic checkpoint
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

            pbar.close()

            # Print epoch summary
            train_loss = epoch_loss / epoch_batches
            print(f"  Train loss: {train_loss:.4f}")

            # Validation
            val_loss = self.validate()
            print(f"  Val loss:   {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(step, best=True)
                print(f"  New best model! (val_loss: {val_loss:.4f})")

            print("")

        # Save final checkpoint
        self.save_checkpoint(step, final=True)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")

    def save_checkpoint(self, step: int, final: bool = False, best: bool = False):
        """Save model checkpoint."""
        if best:
            filename = "best.pt"
        elif final:
            filename = "final.pt"
        else:
            filename = f"step_{step}.pt"

        path = Path(self.config.checkpoint_dir) / filename

        torch.save({
            'step': step,
            'qformer_state_dict': self.qformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)

        tqdm.write(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train X-Leela Q-Former')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--val-interval', type=int, default=500,
                        help='Validate every N steps (default: 500)')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to contrastive pre-training checkpoint')
    parser.add_argument('--llm', type=str, default='gpt2',
                        help='LLM model to use (default: gpt2)')
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        val_interval=args.val_interval,
        pretrain_checkpoint=args.pretrain,
        llm_model=args.llm,
    )

    trainer = XLeelaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
