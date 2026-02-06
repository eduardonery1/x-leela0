"""
X-Leela Inference Script

Generate chess commentary from a position using the trained Q-Former.
"""

import sys
import torch
from pathlib import Path

# Add lc0 directory to path
sys.path.insert(0, str(Path(__file__).parent / "lc0"))

from qformer import create_qformer
from lc0_pytorch import create_lc0_model, encode_position
from train import TrainingConfig  # Needed to unpickle checkpoint
import chess


def load_model(checkpoint_path: str, llm_model: str = "gpt2", device: str = "cuda"):
    """Load all models for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load LC0
    print("Loading LC0...")
    lc0 = create_lc0_model("lc0/weights")
    lc0.eval()
    lc0 = lc0.to(device)

    # Load LLM
    print(f"Loading LLM: {llm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16)
    llm.eval()
    llm = llm.to(device)

    # Create Q-Former with matching dimensions
    print("Loading Q-Former...")
    qformer = create_qformer(
        lc0_dim=lc0.filters,  # 128
        hidden_dim=768,
        chessgpt_dim=llm.config.hidden_size,  # 768 for gpt2
        num_queries=8,
        num_layers=6,
    )

    # Load weights
    qformer.load_state_dict(ckpt['qformer_state_dict'])
    qformer.eval()
    qformer = qformer.to(device)

    print(f"Loaded from step {ckpt['step']}")

    return lc0, qformer, llm, tokenizer, device


@torch.no_grad()
def generate_commentary(
    fen: str,
    lc0,
    qformer,
    llm,
    tokenizer,
    device: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 3,
):
    """Generate commentary for a chess position."""
    # Encode position
    board = chess.Board(fen)
    position = encode_position(board)
    position_tensor = torch.from_numpy(position).unsqueeze(0).to(device)

    # Get LC0 embeddings
    lc0_emb = lc0.extract_spatial_embedding(position_tensor)  # (1, 64, 128)

    # Get soft prompts from Q-Former
    soft_prompts = qformer(lc0_emb)  # (1, 8, 768)
    soft_prompts = soft_prompts.to(llm.dtype)

    # Generate multiple samples
    outputs = []
    for _ in range(num_samples):
        generated = llm.generate(
            inputs_embeds=soft_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode (skip the soft prompt positions)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append(text)

    return outputs


def print_board(fen: str):
    """Print a simple ASCII board."""
    board = chess.Board(fen)
    print(board)
    print(f"\nFEN: {fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate chess commentary')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to checkpoint')
    parser.add_argument('--llm', type=str, default='gpt2',
                        help='LLM model to use')
    parser.add_argument('--fen', type=str, default=None,
                        help='FEN string (default: example positions)')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of samples to generate')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lc0, qformer, llm, tokenizer, device = load_model(
        args.checkpoint, args.llm, device
    )

    # Test positions
    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        # Ruy Lopez
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        # Sicilian Dragon
        "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
        # Complex middlegame
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
    ]

    if args.fen:
        test_positions = [args.fen]

    print("=" * 60)
    print("X-Leela Inference")
    print("=" * 60)

    for fen in test_positions:
        print("\n" + "-" * 60)
        print_board(fen)
        print("\nGenerated commentaries:")

        commentaries = generate_commentary(
            fen, lc0, qformer, llm, tokenizer, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            num_samples=args.samples,
        )

        for i, text in enumerate(commentaries, 1):
            print(f"\n  [{i}] {text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
