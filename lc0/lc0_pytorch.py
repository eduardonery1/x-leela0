"""
LC0 128x10-SE Network in PyTorch

Loads LC0 weights and provides embedding extraction for chess positions.
"""

import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

# Import generated protobuf (run: protoc --python_out=proto --proto_path=proto net.proto)
import importlib.util
from pathlib import Path
_proto_path = Path(__file__).parent / "proto" / "net_pb2.py"
_spec = importlib.util.spec_from_file_location("net_pb2", _proto_path)
net_pb2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(net_pb2)


# =============================================================================
# Weight Loading Utilities
# =============================================================================

def decode_layer(layer: net_pb2.Weights.Layer) -> np.ndarray:
    """Decode a protobuf weight layer to float32 numpy array."""
    if not layer.params:
        return np.array([], dtype=np.float32)

    params = layer.params
    encoding = layer.encoding if layer.HasField('encoding') else net_pb2.Weights.Layer.LINEAR16

    if encoding == net_pb2.Weights.Layer.LINEAR16:
        # Min-max quantized uint16
        data = np.frombuffer(params, dtype=np.uint16)
        theta = data.astype(np.float32) / 65535.0
        min_val = layer.min_val if layer.HasField('min_val') else 0.0
        max_val = layer.max_val if layer.HasField('max_val') else 1.0
        weights = min_val * (1.0 - theta) + max_val * theta

    elif encoding == net_pb2.Weights.Layer.FLOAT16:
        weights = np.frombuffer(params, dtype=np.float16).astype(np.float32)

    elif encoding == net_pb2.Weights.Layer.BFLOAT16:
        # BF16: upper 16 bits of float32
        u16 = np.frombuffer(params, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        weights = u32.view(np.float32)

    elif encoding == net_pb2.Weights.Layer.FLOAT32:
        weights = np.frombuffer(params, dtype=np.float32).copy()

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return weights


def load_lc0_weights(filepath: str) -> net_pb2.Net:
    """Load and parse LC0 weight file."""
    path = Path(filepath)

    # Handle both .pb.gz and raw gzip files
    with gzip.open(path, 'rb') as f:
        data = f.read()

    net = net_pb2.Net()
    net.ParseFromString(data)

    # Validate magic number
    if net.magic != 0x1c0:
        raise ValueError(f"Invalid LC0 magic number: 0x{net.magic:x}")

    return net


# =============================================================================
# PyTorch Network Components
# =============================================================================

class SEUnit(nn.Module):
    """Squeeze-and-Excitation unit."""

    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.channels = channels
        self.se_channels = se_channels
        self.fc1 = nn.Linear(channels, se_channels, bias=True)
        self.fc2 = nn.Linear(se_channels, 2 * channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape

        # Global average pooling
        y = x.mean(dim=[2, 3])  # (B, C)

        # FC1 + ReLU
        y = F.relu(self.fc1(y))

        # FC2 -> split into W (scale) and B (bias)
        y = self.fc2(y)  # (B, 2C)

        # First half is multiplicative (sigmoid), second half is additive
        w = torch.sigmoid(y[:, :channels]).view(batch, channels, 1, 1)
        b = y[:, channels:].view(batch, channels, 1, 1)

        return x * w + b


class LC0BatchNorm(nn.Module):
    """LC0-style BatchNorm.

    LC0 stores bn_stddivs as VARIANCE (not 1/sqrt(var+eps)).
    Formula: y = gamma * (x - mean) / sqrt(variance + eps) + beta
    """
    EPS = 1e-5

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer('bn_means', torch.zeros(channels))
        self.register_buffer('bn_variances', torch.ones(channels))  # This is variance, not stddiv
        self.register_buffer('bn_gammas', torch.ones(channels))
        self.register_buffer('bn_betas', torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, h, w)
        # Apply: y = gamma * (x - mean) / sqrt(variance + eps) + beta
        mean = self.bn_means.view(1, -1, 1, 1)
        var = self.bn_variances.view(1, -1, 1, 1)
        gamma = self.bn_gammas.view(1, -1, 1, 1)
        beta = self.bn_betas.view(1, -1, 1, 1)

        return gamma * (x - mean) / torch.sqrt(var + self.EPS) + beta


class ConvBlock(nn.Module):
    """Convolution + BatchNorm block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.bn = LC0BatchNorm(out_channels)

    def forward(self, x: torch.Tensor, relu: bool = True) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if relu:
            x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional SE unit."""

    def __init__(self, channels: int, se_channels: Optional[int] = None):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3)
        self.se = SEUnit(channels, se_channels) if se_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x, relu=True)
        out = self.conv2(out, relu=False)
        if self.se:
            out = self.se(out)
        return F.relu(out + residual)


class PolicyHead(nn.Module):
    """Convolutional policy head."""

    def __init__(self, channels: int, policy_channels: int = 80):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3)
        # Policy uses 3x3 conv with bias (no batchnorm)
        self.conv2 = nn.Conv2d(channels, policy_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, relu=True)
        x = self.conv2(x)
        # Flatten to (batch, 80*64) = (batch, 5120), then map to 1858 moves
        return x.view(x.size(0), -1)


class ValueHead(nn.Module):
    """WDL value head."""

    def __init__(self, channels: int, value_channels: int = 32, fc_size: int = 128):
        super().__init__()
        self.conv = ConvBlock(channels, value_channels, kernel_size=1)
        self.fc1 = nn.Linear(value_channels * 64, fc_size)
        self.fc2 = nn.Linear(fc_size, 3)  # WDL output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, relu=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Raw logits, apply softmax for probabilities


class MovesLeftHead(nn.Module):
    """Moves left head."""

    def __init__(self, channels: int, mlh_channels: int = 8, fc_size: int = 128):
        super().__init__()
        self.conv = ConvBlock(channels, mlh_channels, kernel_size=1)
        self.fc1 = nn.Linear(mlh_channels * 64, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, relu=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Main LC0 Network
# =============================================================================

class LC0Network(nn.Module):
    """
    LC0 SE (Squeeze-Excitation) Network.

    This network extracts chess position embeddings from 112-plane input.
    """

    def __init__(self, filters: int = 128, blocks: int = 10, se_ratio: int = 4):
        super().__init__()
        self.filters = filters
        self.blocks = blocks
        self.se_channels = filters // se_ratio

        # Input convolution
        self.input_conv = ConvBlock(112, filters, kernel_size=3)

        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(filters, self.se_channels) for _ in range(blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)
        self.mlh_head = MovesLeftHead(filters)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass with all outputs.

        Args:
            x: (batch, 112, 8, 8) input planes

        Returns:
            dict with 'policy', 'value', 'mlh', 'embedding'
        """
        # Input conv
        x = self.input_conv(x, relu=True)

        # Residual tower
        for block in self.residual_tower:
            x = block(x)

        # x is now (batch, filters, 8, 8) - the embedding
        embedding = x

        return {
            'embedding': embedding,
            'embedding_flat': embedding.view(embedding.size(0), -1),
            'policy': self.policy_head(x),
            'value': self.value_head(x),
            'mlh': self.mlh_head(x),
        }

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract only the embedding (faster, no head computation).

        Args:
            x: (batch, 112, 8, 8) input planes

        Returns:
            embedding: (batch, filters * 64) = (batch, 8192) for 128 filters
        """
        x = self.input_conv(x, relu=True)
        for block in self.residual_tower:
            x = block(x)
        return x.view(x.size(0), -1)

    def extract_spatial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-square embeddings.

        Args:
            x: (batch, 112, 8, 8) input planes

        Returns:
            embedding: (batch, 64, filters) = (batch, 64, 128)
        """
        x = self.input_conv(x, relu=True)
        for block in self.residual_tower:
            x = block(x)
        # (batch, C, 8, 8) -> (batch, 8, 8, C) -> (batch, 64, C)
        return x.permute(0, 2, 3, 1).reshape(x.size(0), 64, -1)


# =============================================================================
# Weight Loading
# =============================================================================

def load_conv_block(conv_block: ConvBlock, pb_block, filters_in: int, filters_out: int):
    """Load weights into a ConvBlock from protobuf.

    LC0 stores BatchNorm as: y = gamma * (x - mean) * stddiv + beta
    where stddiv = 1/sqrt(var + eps).
    """
    # Convolution weights
    conv_weights = decode_layer(pb_block.weights)
    kernel_size = 3 if len(conv_weights) == filters_out * filters_in * 9 else 1
    conv_weights = conv_weights.reshape(filters_out, filters_in, kernel_size, kernel_size)
    conv_block.conv.weight.data = torch.from_numpy(conv_weights)

    # BatchNorm parameters - load directly into LC0BatchNorm
    bn_gammas = decode_layer(pb_block.bn_gammas)
    bn_betas = decode_layer(pb_block.bn_betas)
    bn_means = decode_layer(pb_block.bn_means)
    bn_stddivs = decode_layer(pb_block.bn_stddivs)

    if len(bn_gammas) > 0:
        conv_block.bn.bn_gammas.copy_(torch.from_numpy(bn_gammas))
    if len(bn_betas) > 0:
        conv_block.bn.bn_betas.copy_(torch.from_numpy(bn_betas))
    if len(bn_means) > 0:
        conv_block.bn.bn_means.copy_(torch.from_numpy(bn_means))
    if len(bn_stddivs) > 0:
        # bn_stddivs in LC0 is actually variance, not 1/sqrt(var+eps)
        conv_block.bn.bn_variances.copy_(torch.from_numpy(bn_stddivs))


def load_se_unit(se: SEUnit, pb_se, channels: int):
    """Load weights into an SEUnit from protobuf."""
    se_channels = se.se_channels

    # FC1: channels -> se_channels
    w1 = decode_layer(pb_se.w1).reshape(se_channels, channels)
    b1 = decode_layer(pb_se.b1)
    se.fc1.weight.data = torch.from_numpy(w1)
    if len(b1) > 0:
        se.fc1.bias.data = torch.from_numpy(b1)

    # FC2: se_channels -> 2*channels (for W and B)
    w2 = decode_layer(pb_se.w2).reshape(2 * channels, se_channels)
    b2 = decode_layer(pb_se.b2)
    se.fc2.weight.data = torch.from_numpy(w2)
    if len(b2) > 0:
        se.fc2.bias.data = torch.from_numpy(b2)


def load_weights_into_model(model: LC0Network, net: net_pb2.Net):
    """Load all weights from protobuf into PyTorch model."""
    w = net.weights
    filters = model.filters

    # Input convolution (112 -> filters)
    load_conv_block(model.input_conv, w.input, 112, filters)

    # Residual blocks
    for i, block in enumerate(model.residual_tower):
        pb_block = w.residual[i]
        load_conv_block(block.conv1, pb_block.conv1, filters, filters)
        load_conv_block(block.conv2, pb_block.conv2, filters, filters)
        if block.se:
            load_se_unit(block.se, pb_block.se, filters)

    # Policy head
    load_conv_block(model.policy_head.conv1, w.policy1, filters, filters)

    # Policy conv2 (3x3 with bias, no batchnorm)
    pol_weights = decode_layer(w.policy.weights)
    pol_bias = decode_layer(w.policy.biases)
    pol_channels = len(pol_bias)  # 80 channels
    pol_weights = pol_weights.reshape(pol_channels, filters, 3, 3)
    model.policy_head.conv2.weight.data = torch.from_numpy(pol_weights)
    model.policy_head.conv2.bias.data = torch.from_numpy(pol_bias)

    # Value head - infer channels from FC layer
    ip1_val_w = decode_layer(w.ip1_val_w)
    ip1_val_b = decode_layer(w.ip1_val_b)
    fc_size = len(ip1_val_b)  # 128
    val_fc_input = len(ip1_val_w) // fc_size  # 2048
    val_channels = val_fc_input // 64  # 32

    load_conv_block(model.value_head.conv, w.value, filters, val_channels)

    model.value_head.fc1.weight.data = torch.from_numpy(
        ip1_val_w.reshape(fc_size, val_channels * 64))
    model.value_head.fc1.bias.data = torch.from_numpy(ip1_val_b)

    ip2_val_w = decode_layer(w.ip2_val_w)
    ip2_val_b = decode_layer(w.ip2_val_b)
    model.value_head.fc2.weight.data = torch.from_numpy(ip2_val_w.reshape(3, fc_size))
    if len(ip2_val_b) > 0:
        model.value_head.fc2.bias.data = torch.from_numpy(ip2_val_b)

    # Moves left head - infer channels from FC layer
    ip1_mov_w = decode_layer(w.ip1_mov_w)
    ip1_mov_b = decode_layer(w.ip1_mov_b)
    mlh_fc_size = len(ip1_mov_b)  # 128
    mlh_fc_input = len(ip1_mov_w) // mlh_fc_size  # 512
    mlh_channels = mlh_fc_input // 64  # 8

    load_conv_block(model.mlh_head.conv, w.moves_left, filters, mlh_channels)

    model.mlh_head.fc1.weight.data = torch.from_numpy(
        ip1_mov_w.reshape(mlh_fc_size, mlh_channels * 64))
    model.mlh_head.fc1.bias.data = torch.from_numpy(ip1_mov_b)

    ip2_mov_w = decode_layer(w.ip2_mov_w)
    ip2_mov_b = decode_layer(w.ip2_mov_b)
    model.mlh_head.fc2.weight.data = torch.from_numpy(ip2_mov_w.reshape(1, mlh_fc_size))
    model.mlh_head.fc2.bias.data = torch.from_numpy(ip2_mov_b)

    print(f"Loaded {filters}x{len(model.residual_tower)} SE network")


def create_lc0_model(weights_path: str) -> LC0Network:
    """
    Create LC0 model and load weights from file.

    Args:
        weights_path: Path to LC0 .pb.gz weights file

    Returns:
        LC0Network with loaded weights in eval mode
    """
    # Load protobuf
    net = load_lc0_weights(weights_path)
    w = net.weights

    # Infer network dimensions
    input_conv_params = len(decode_layer(w.input.weights))
    filters = input_conv_params // (112 * 3 * 3)
    blocks = len(w.residual)

    # Infer SE ratio from first block
    se_w1_params = len(decode_layer(w.residual[0].se.w1))
    se_channels = se_w1_params // filters
    se_ratio = filters // se_channels

    print(f"Detected network: {filters}x{blocks}-SE (SE ratio={se_ratio})")

    # Create model
    model = LC0Network(filters=filters, blocks=blocks, se_ratio=se_ratio)

    # Load weights
    load_weights_into_model(model, net)

    # Set to eval mode
    model.eval()

    return model


# =============================================================================
# Position Encoding
# =============================================================================

def encode_board(board, flip: bool = False) -> np.ndarray:
    """
    Encode a python-chess Board to a single position's 13 planes.

    Args:
        board: chess.Board object
        flip: Whether to flip perspective (True if black to move)

    Returns:
        np.ndarray of shape (13, 8, 8)
    """
    import chess

    planes = np.zeros((13, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }

    our_color = chess.BLACK if flip else chess.WHITE

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if flip:
            rank = 7 - rank

        plane_idx = piece_map[piece.piece_type]
        if piece.color != our_color:
            plane_idx += 6

        planes[plane_idx, rank, file] = 1.0

    return planes


def encode_position(board, history: list = None) -> np.ndarray:
    """
    Encode a chess position to 112 input planes.

    Args:
        board: chess.Board - current position
        history: Optional list of previous chess.Board positions

    Returns:
        np.ndarray of shape (112, 8, 8)
    """
    import chess

    planes = np.zeros((112, 8, 8), dtype=np.float32)

    if history is None:
        history = []
    positions = history + [board]

    flip = board.turn == chess.BLACK

    # Encode up to 8 history positions
    for i in range(min(8, len(positions))):
        hist_idx = len(positions) - 1 - i
        pos = positions[hist_idx]
        pos_flip = flip if (i % 2 == 0) else not flip

        base = i * 13
        planes[base:base+13] = encode_board(pos, pos_flip)

    # Auxiliary planes (104-111)
    # Castling rights
    if board.has_queenside_castling_rights(chess.WHITE):
        rank = 0 if not flip else 7
        planes[104, rank, 0] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        rank = 7 if not flip else 0
        planes[104, rank, 0] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        rank = 0 if not flip else 7
        planes[105, rank, 7] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        rank = 7 if not flip else 0
        planes[105, rank, 7] = 1.0

    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        if flip:
            rank = 7 - rank
        planes[108, rank, file] = 1.0

    # Rule50
    planes[109, :, :] = board.halfmove_clock / 100.0

    # All ones plane
    planes[111, :, :] = 1.0

    return planes


def encode_fen(fen: str) -> np.ndarray:
    """Convenience function to encode a FEN string."""
    import chess
    board = chess.Board(fen)
    return encode_position(board)


# =============================================================================
# Main / Test
# =============================================================================

if __name__ == "__main__":
    import sys

    weights_path = sys.argv[1] if len(sys.argv) > 1 else "weights"

    print("=" * 60)
    print("Loading LC0 Network")
    print("=" * 60)

    # Create and load model
    model = create_lc0_model(weights_path)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test with starting position
    print("\n" + "=" * 60)
    print("Testing with starting position")
    print("=" * 60)

    try:
        import chess
        board = chess.Board()
        input_planes = encode_position(board)
        input_tensor = torch.from_numpy(input_planes).unsqueeze(0)  # (1, 112, 8, 8)

        with torch.no_grad():
            outputs = model(input_tensor)

        print(f"\nInput shape: {input_tensor.shape}")
        print(f"Embedding shape: {outputs['embedding'].shape}")
        print(f"Embedding flat shape: {outputs['embedding_flat'].shape}")
        print(f"Policy shape: {outputs['policy'].shape}")
        print(f"Value shape: {outputs['value'].shape}")
        print(f"MLH shape: {outputs['mlh'].shape}")

        # WDL probabilities
        wdl = F.softmax(outputs['value'], dim=1)
        print(f"\nWDL probabilities: W={wdl[0,0]:.3f}, D={wdl[0,1]:.3f}, L={wdl[0,2]:.3f}")
        print(f"Moves left prediction: {outputs['mlh'][0,0]:.1f}")

        # Embedding stats
        emb = outputs['embedding_flat']
        print(f"\nEmbedding stats:")
        print(f"  Mean: {emb.mean():.4f}")
        print(f"  Std: {emb.std():.4f}")
        print(f"  Min: {emb.min():.4f}")
        print(f"  Max: {emb.max():.4f}")

    except ImportError:
        print("python-chess not installed. Skipping position test.")
        print("Install with: pip install chess")

        # Test with random input
        random_input = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            outputs = model(random_input)
        print(f"\nRandom input test:")
        print(f"Embedding shape: {outputs['embedding_flat'].shape}")

    print("\n" + "=" * 60)
    print("Model ready for use!")
    print("=" * 60)
