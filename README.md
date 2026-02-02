# X-Leela: Explainable Chess AI

**Bridging Strategic Mastery and Linguistic Interpretability via Latent Space Alignment**

## Project Goal

X-Leela transforms the "black-box" nature of superhuman chess engines into an interpretable system capable of providing faithful natural language explanations for positions and moves.

The system connects **LC0** (Leela Chess Zero) strategic representations to **ChessGPT** language generation through a trainable **Q-Former Bridge** (inspired by BLIP-2), ensuring explanations are grounded in the engine's actual decision-making process.

## Architecture

```
LC0 (frozen, 3.8M)     Q-Former (trainable, 59M)     ChessGPT (frozen, 2.78B)
      ↓                         ↓                            ↓
  (112,8,8)              (64,128) → (8,2560)           Soft prompts → Text
  Position             Cross-attention bridge         Commentary generation
```

Three-component pipeline with only the middle component trainable:

1. **LC0 128x10-SE** (frozen, 3.8M params): Produces (64, 128) per-square embeddings
2. **Q-Former Bridge** (trainable, 59M params): Learnable queries + cross-attention → soft prompts (8×2560)
3. **ChessGPT** (frozen, 2.78B params): GPT-NeoX decoder that generates chess commentary

## Components

| Component | Parameters | Status | Description |
|-----------|------------|--------|-------------|
| LC0 128x10-SE | 3.8M | Frozen | SE-ResNet for strategic position features |
| Q-Former Bridge | 59M | **Trainable** | Maps LC0 embeddings → ChessGPT soft prompts |
| ChessGPT | 2.78B | Frozen | GPT-NeoX 3B fine-tuned on chess data |

## Two-Stage Training (BLIP-2 Style)

### Stage 1: Contrastive Pre-training
Aligns Q-Former position representations with text representations using ITC loss.

```bash
python pretrain_contrastive.py --batch-size 32 --max-steps 20000
```

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| LC0 | 3.8M | No |
| Q-Former | 60.8M | Yes |
| Text Encoder | 67.8M | Yes |

Output: `checkpoints/pretrain_final.pt`

### Stage 2: LLM Fine-tuning
Connects pre-trained Q-Former to ChessGPT with language modeling loss.

```bash
python train.py --batch-size 4 --pretrain checkpoints/pretrain_final.pt --max-steps 50000
```

| Component | Parameters | Device |
|-----------|------------|--------|
| LC0 | 3.8M (frozen) | GPU |
| Q-Former | 59M | GPU |
| ChessGPT | 2.78B (frozen) | CPU |

## Key Dimensions

| Stage | Shape | Total |
|-------|-------|-------|
| LC0 Input | (112, 8, 8) | 7,168 |
| LC0 Output | (64, 128) | 8,192 |
| Q-Former Queries | (8, 768) | 6,144 |
| Soft Prompt | (8, 2560) | 20,480 |

## Training Objectives

### Stage 1: Contrastive (ITC)
```
logits = pos_emb @ text_emb.T / temperature
loss = (CrossEntropy(logits, labels) + CrossEntropy(logits.T, labels)) / 2
```

### Stage 2: Language Modeling
```
soft_prompts = qformer(lc0_embeddings)
loss = chessgpt(inputs_embeds=soft_prompts, labels=target).loss
```

## Quick Start

```bash
# Test LC0 embeddings
cd lc0 && python lc0_pytorch.py weights

# Test Q-Former
python qformer.py

# Test contrastive pre-training (1 step)
python -c "
from pretrain_contrastive import ContrastiveConfig, ContrastiveTrainer
trainer = ContrastiveTrainer(ContrastiveConfig(batch_size=8, max_steps=1))
"
```

## Project Structure

```
├── qformer.py                 # Q-Former bridge implementation
├── pretrain_contrastive.py    # Stage 1: Contrastive pre-training (ITC loss)
├── train.py                   # Stage 2: LLM fine-tuning
├── dataset.py                 # Chess commentary dataset loader
├── lc0/
│   ├── lc0_pytorch.py         # LC0 weight loading and inference
│   ├── weights                # LC0 128x10-SE weights (gzipped protobuf)
│   └── proto/net_pb2.py       # Generated protobuf
└── checkpoints/               # Model checkpoints
```

## Dataset

- **Pairs**: 354,423 (position, commentary) pairs
- **Format**: FEN + tokenized commentary

## Memory Optimization (GTX 1650 - 4GB VRAM)

- LC0 + Q-Former on GPU (~260MB)
- ChessGPT on CPU (2.78B doesn't fit on GPU)
- Gradients flow: CPU loss → GPU Q-Former
- Gradient accumulation for effective batch size

## Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| Li et al. - BLIP-2 | 2023 | Frozen decoder bridging pattern |
| Feng et al. - ChessGPT | 2023 | Chess language model |
| Silver et al. - AlphaZero | 2018 | Self-play RL architecture |

## License

Research use only. See individual component licenses (LC0: GPL, ChessGPT: Apache 2.0).
