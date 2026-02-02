"""
Chess Commentary Dataset Loader

Extracts (position, comment) pairs from annotated PGN files for X-Leela training.
Supports both raw PGN files and the ChessGPT JSONL format.
"""

import os
import json
import glob
import io
import re
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Optional, Union
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class PositionComment:
    """A single (position, comment) pair."""
    fen: str
    comment: str
    move_uci: str  # The move that led to this position
    move_san: str  # Human-readable move notation
    ply: int  # Half-move number (0-indexed)
    game_id: str  # Source identifier


class LC0Encoder:
    """
    Encodes chess positions into lc0's 112-channel format.

    Channel layout (8x8 each):
    - Planes 0-103: 8 history positions × 13 planes each
        - 6 planes for our pieces (P, N, B, R, Q, K)
        - 6 planes for their pieces
        - 1 plane for repetition count
    - Planes 104-111: Auxiliary planes
        - 104: Castling rights (us kingside)
        - 105: Castling rights (us queenside)
        - 106: Castling rights (them kingside)
        - 107: Castling rights (them queenside)
        - 108: Side to move (all 1s if black)
        - 109: Rule 50 counter (normalized)
        - 110: Reserved (zeros)
        - 111: All ones (bias plane)
    """

    PIECE_PLANES = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    def __init__(self, history_length: int = 8):
        self.history_length = history_length

    def encode(
        self,
        board: chess.Board,
        history: Optional[List[chess.Board]] = None
    ) -> np.ndarray:
        """
        Encode a position into 112 planes of 8x8.

        Args:
            board: Current board position
            history: List of previous board states (most recent first)
                    If None, only encodes current position

        Returns:
            np.ndarray of shape (112, 8, 8) with float32 values
        """
        planes = np.zeros((112, 8, 8), dtype=np.float32)

        # Determine perspective (always encode from side-to-move's view)
        flip = board.turn == chess.BLACK

        # Encode board history
        boards = [board]
        if history:
            boards.extend(history[:self.history_length - 1])

        for t, hist_board in enumerate(boards):
            if t >= self.history_length:
                break
            base_plane = t * 13
            self._encode_board(planes, base_plane, hist_board, flip)

        # Auxiliary planes (104-111)
        self._encode_auxiliary(planes, board, flip)

        return planes

    def _encode_board(
        self,
        planes: np.ndarray,
        base: int,
        board: chess.Board,
        flip: bool
    ) -> None:
        """Encode a single board position into 13 planes."""
        # Determine which color is "us" vs "them"
        us = chess.BLACK if flip else chess.WHITE
        them = chess.WHITE if flip else chess.BLACK

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Get coordinates (flip if playing as black)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if flip:
                rank = 7 - rank

            # Determine plane offset
            plane_offset = self.PIECE_PLANES[piece.piece_type]
            if piece.color == them:
                plane_offset += 6  # Their pieces are in planes 6-11

            planes[base + plane_offset, rank, file] = 1.0

        # Repetition plane (plane 12 relative to base)
        # Count how many times this position has occurred
        rep_count = 0
        if board.is_repetition(2):
            rep_count = 1
        if board.is_repetition(3):
            rep_count = 2
        if rep_count > 0:
            planes[base + 12, :, :] = rep_count / 2.0  # Normalize

    def _encode_auxiliary(
        self,
        planes: np.ndarray,
        board: chess.Board,
        flip: bool
    ) -> None:
        """Encode auxiliary planes 104-111."""
        us = chess.BLACK if flip else chess.WHITE
        them = chess.WHITE if flip else chess.BLACK

        # Castling rights
        if us == chess.WHITE:
            planes[104, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
            planes[105, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
            planes[106, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
            planes[107, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
        else:
            planes[104, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
            planes[105, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
            planes[106, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
            planes[107, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))

        # Side to move (1 if it's black's turn in original position)
        planes[108, :, :] = float(board.turn == chess.BLACK)

        # Rule 50 counter (normalized to 0-1)
        planes[109, :, :] = board.halfmove_clock / 100.0

        # Plane 110: Reserved (zeros)
        # Plane 111: All ones (bias)
        planes[111, :, :] = 1.0


def parse_pgn_with_comments(pgn_text: str, game_id: str = "") -> Iterator[PositionComment]:
    """
    Parse a PGN string and yield (position, comment) pairs.

    Args:
        pgn_text: PGN content as string
        game_id: Identifier for the source game

    Yields:
        PositionComment for each move that has a comment
    """
    pgn_io = io.StringIO(pgn_text)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        board = game.board()
        node = game
        ply = 0

        # Check for game-level comment
        if game.comment:
            yield PositionComment(
                fen=board.fen(),
                comment=clean_comment(game.comment),
                move_uci="",
                move_san="",
                ply=0,
                game_id=game_id
            )

        # Traverse the main line
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move

            # Get SAN before making the move
            move_san = board.san(move)
            move_uci = move.uci()

            # Make the move
            board.push(move)
            ply += 1

            # Check for comment on this move
            if next_node.comment:
                yield PositionComment(
                    fen=board.fen(),
                    comment=clean_comment(next_node.comment),
                    move_uci=move_uci,
                    move_san=move_san,
                    ply=ply,
                    game_id=game_id
                )

            node = next_node


def clean_comment(comment: str) -> str:
    """Clean up a PGN comment string."""
    # Remove excessive whitespace
    comment = re.sub(r'\s+', ' ', comment).strip()
    # Remove clock annotations like [%clk 0:05:00]
    comment = re.sub(r'\[%[^\]]+\]', '', comment).strip()
    return comment


class ChessCommentaryDataset(Dataset):
    """
    PyTorch Dataset for chess position-comment pairs.

    Loads and caches all data in memory for random access.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer=None,
        max_comment_length: int = 256,
        encode_positions: bool = True,
        cache_file: Optional[str] = None
    ):
        """
        Args:
            data_dir: Directory containing PGN files and/or JSONL files
            tokenizer: HuggingFace tokenizer for encoding comments
            max_comment_length: Maximum token length for comments
            encode_positions: Whether to encode positions as 112-channel tensors
            cache_file: Optional path to cache extracted pairs
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_comment_length = max_comment_length
        self.encode_positions = encode_positions
        self.encoder = LC0Encoder() if encode_positions else None

        # Load or extract data
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            self.pairs = self._load_cache(cache_file)
        else:
            print(f"Extracting position-comment pairs from {data_dir}")
            self.pairs = self._extract_all_pairs()
            if cache_file:
                self._save_cache(cache_file)

        print(f"Loaded {len(self.pairs)} position-comment pairs")

    def _extract_all_pairs(self) -> List[PositionComment]:
        """Extract all position-comment pairs from the data directory."""
        pairs = []

        # Process JSONL files (ChessGPT format)
        for jsonl_path in self.data_dir.rglob("*.jsonl*"):
            pairs.extend(self._process_jsonl(jsonl_path))

        # Process PGN files
        for pgn_path in self.data_dir.rglob("*.pgn"):
            pairs.extend(self._process_pgn(pgn_path))

        return pairs

    def _process_jsonl(self, path: Path) -> List[PositionComment]:
        """Process a JSONL file containing PGN text."""
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    pgn_text = record.get('text', '')
                    game_id = record.get('pipeline_key', f"{path.name}:{i}")
                    pairs.extend(parse_pgn_with_comments(pgn_text, game_id))
                except (json.JSONDecodeError, Exception) as e:
                    continue
        return pairs

    def _process_pgn(self, path: Path) -> List[PositionComment]:
        """Process a single PGN file."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                pgn_text = f.read()
            return list(parse_pgn_with_comments(pgn_text, str(path)))
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return []

    def _save_cache(self, cache_file: str) -> None:
        """Save extracted pairs to cache file."""
        data = [
            {
                'fen': p.fen,
                'comment': p.comment,
                'move_uci': p.move_uci,
                'move_san': p.move_san,
                'ply': p.ply,
                'game_id': p.game_id
            }
            for p in self.pairs
        ]
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        print(f"Saved cache to {cache_file}")

    def _load_cache(self, cache_file: str) -> List[PositionComment]:
        """Load pairs from cache file."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return [PositionComment(**d) for d in data]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns dict with:
            - position: (112, 8, 8) tensor if encode_positions=True, else FEN string
            - comment: Tokenized comment tensor if tokenizer provided, else string
            - move_san: The move in SAN notation
            - ply: Half-move number
        """
        pair = self.pairs[idx]

        result = {
            'move_san': pair.move_san,
            'ply': pair.ply,
            'game_id': pair.game_id,
        }

        # Encode position
        if self.encode_positions:
            board = chess.Board(pair.fen)
            position = self.encoder.encode(board)
            result['position'] = torch.from_numpy(position)
        else:
            result['fen'] = pair.fen

        # Tokenize comment
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                pair.comment,
                max_length=self.max_comment_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = tokens['input_ids'].squeeze(0)
            result['attention_mask'] = tokens['attention_mask'].squeeze(0)
        else:
            result['comment'] = pair.comment

        return result


class StreamingChessDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large PGN collections.

    Use this when the full dataset doesn't fit in memory.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer=None,
        max_comment_length: int = 256,
        shuffle_files: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_comment_length = max_comment_length
        self.shuffle_files = shuffle_files
        self.encoder = LC0Encoder()

        # Collect all file paths
        self.files = list(self.data_dir.rglob("*.pgn"))
        self.files.extend(self.data_dir.rglob("*.jsonl*"))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self.files.copy()
        if self.shuffle_files:
            import random
            random.shuffle(files)

        for file_path in files:
            if file_path.suffix == '.pgn':
                yield from self._iter_pgn(file_path)
            else:
                yield from self._iter_jsonl(file_path)

    def _iter_pgn(self, path: Path) -> Iterator[Dict]:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                pgn_text = f.read()
            for pair in parse_pgn_with_comments(pgn_text, str(path)):
                yield self._pair_to_dict(pair)
        except Exception:
            pass

    def _iter_jsonl(self, path: Path) -> Iterator[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        record = json.loads(line)
                        pgn_text = record.get('text', '')
                        game_id = record.get('pipeline_key', f"{path.name}:{i}")
                        for pair in parse_pgn_with_comments(pgn_text, game_id):
                            yield self._pair_to_dict(pair)
                    except Exception:
                        continue
        except Exception:
            pass

    def _pair_to_dict(self, pair: PositionComment) -> Dict:
        board = chess.Board(pair.fen)
        position = self.encoder.encode(board)

        result = {
            'position': torch.from_numpy(position),
            'move_san': pair.move_san,
            'ply': pair.ply,
        }

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                pair.comment,
                max_length=self.max_comment_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = tokens['input_ids'].squeeze(0)
            result['attention_mask'] = tokens['attention_mask'].squeeze(0)
        else:
            result['comment'] = pair.comment

        return result


def get_dataset_stats(data_dir: str) -> Dict:
    """
    Get statistics about the dataset without loading it fully.
    """
    data_dir = Path(data_dir)

    pgn_files = list(data_dir.rglob("*.pgn"))
    jsonl_files = list(data_dir.rglob("*.jsonl*"))

    # Sample to estimate
    sample_pairs = 0
    sample_files = min(10, len(pgn_files))

    for pgn_path in pgn_files[:sample_files]:
        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
                pgn_text = f.read()
            sample_pairs += len(list(parse_pgn_with_comments(pgn_text, "")))
        except Exception:
            pass

    avg_pairs_per_file = sample_pairs / sample_files if sample_files > 0 else 0
    estimated_total = int(avg_pairs_per_file * len(pgn_files))

    # Count JSONL records
    jsonl_records = 0
    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, 'r') as f:
                jsonl_records += sum(1 for _ in f)
        except Exception:
            pass

    return {
        'pgn_files': len(pgn_files),
        'jsonl_files': len(jsonl_files),
        'jsonl_records': jsonl_records,
        'estimated_pairs_from_pgn': estimated_total,
        'sample_avg_pairs_per_pgn': avg_pairs_per_file,
    }


# Example usage and testing
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/nery/Projects/chess/data"

    print("=" * 60)
    print("Chess Commentary Dataset Loader")
    print("=" * 60)

    # Get stats
    print("\nDataset Statistics:")
    stats = get_dataset_stats(data_dir)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test extraction
    print("\n" + "-" * 60)
    print("Sample position-comment pairs:")
    print("-" * 60)

    dataset = ChessCommentaryDataset(
        data_dir,
        tokenizer=None,
        encode_positions=True,
        cache_file=None
    )

    # Show a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n[{i}] Move: {sample['move_san']} (ply {sample['ply']})")
        print(f"    Position shape: {sample['position'].shape}")
        print(f"    Comment: {sample['comment'][:100]}...")

    print(f"\nTotal pairs: {len(dataset)}")
