import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from . import config


@dataclass
class GPTConfig:
    """Configuration for the GPT model.

    Attributes:
        vocab_size: Size of the vocabulary (default: 0, must be set before use).
        num_heads: Number of attention heads (default: from config).
        num_layers: Number of transformer layers (default: from config).
        block_size: Maximum sequence length (default: from config).
        emb_dim: Embedding dimension (default: from config).
        p: Dropout probability (default: from config).
    """

    vocab_size: int = 0
    num_heads: int = config.NUM_HEADS
    num_layers: int = config.NUM_LAYERS
    block_size: int = config.BLOCK_SIZE
    emb_dim: int = config.EMB_DIM
    p: float = config.DROPOUT_P


class GPTModel(nn.Module):
    """GPT model for sequence processing.

    Combines token and positional embeddings, followed by a stack of transformer blocks,
    and a final language modeling head.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config: GPTConfig = config

        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.emb_dim)

        self.blocks = nn.Sequential(
            *[
                Transformer(
                    config.emb_dim, config.num_heads, config.block_size, p=config.p
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_emb(x)

        seq_len = x.shape[1]

        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"The sequence length is {seq_len}, but the positional embedding has {self.pos_emb.num_embeddings} embeddings"
            )

        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


class Head(nn.Module):
    """Single self-attention head for transformer models.

    Computes scaled dot-product attention with causal masking. This mechanism allows the model
    to dynamically focus on different parts of the input sequence for each token, capturing
    long-range dependencies and contextual relationships.

    The attention is calculated as:
        1. Project the input into query (Q), key (K), and value (V) spaces using learned linear transformations.
        2. Compute attention scores as the dot product of queries and keys, scaled by the square root
           of the head dimension to avoid large gradient magnitudes.
        3. Apply a causal mask to ensure each token only attends to previous tokens (no future information).
        4. Normalize scores with softmax to obtain attention weights.
        5. Weight the values (V) using the attention weights and sum to produce the output.

    The causal mask is implemented via a triangular matrix (`self.tril`), where future positions
    are set to -inf, resulting in zero attention weight after softmax.

    Attributes:
        query: Linear layer to project input to query space.
        key: Linear layer to project input to key space.
        value: Linear layer to project input to value space.
        tril: Lower triangular mask to enforce causality.
        dropout: Dropout layer for regularization.
    """

    def __init__(self, emb_dim: int, head_size: int, block_size: int, p: float = 0.2):
        super().__init__()

        self.querie = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # We use a mask to ensure that the attention weights are only computed for the past tokens instead of the future tokens.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        q = self.querie(x)
        v = self.value(x)

        _, t, d = x.shape

        # sa = V * softmax((K^T * Q) / sqrt(D))
        wei = (q @ k.mT) * (d**-0.5)

        assert not isinstance(self.tril, nn.Module)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))

        out = F.softmax(wei, dim=-1) @ v

        return out


class MHSelfAttention(nn.Module):
    """Multi-head self-attention module.

    Combines multiple attention heads to allow the model to jointly attend to information
    from different representation subspaces. Each head learns a distinct set of attention weights,
    enabling the model to capture diverse patterns (e.g., syntactic, semantic) in parallel.

    The outputs of all heads are concatenated and linearly projected to the original embedding dimension.

    Attributes:
        heads: List of `Head` instances.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        p: float = 0.2,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, p) for _ in range(num_heads)]
        )
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for transformer blocks.

    Applies a two-layer feed-forward network with GELU activation and dropout,
    as used in the transformer architecture. This MLP is responsible for the
    non-linear transformation of the input embeddings after self-attention,
    allowing the model to learn complex feature representations.

    The network architecture is:
        Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim) -> Dropout

    Attributes:
        mlp: Sequential module containing the linear layers, activation, and dropout.
    """

    def __init__(self, emb_dim: int, p: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Transformer(nn.Module):
    """Single decoder transformer block.

    Implements the standard transformer architecture with:

        1. Pre-layer normalization (LayerNorm before attention/MLP).
        2. Residual connections around both the attention and MLP sub-layers.
        3. Multi-head self-attention for capturing contextual relationships.
        4. Position-wise feed-forward network (MLP) for non-linear transformation.

    The block follows the structure:
        x -> LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual

    Attributes:
        mhsa: Multi-head self-attention module.
        ln1: Layer normalization before self-attention.
        ffn: Position-wise feed-forward network (MLP).
        ln2: Layer normalization before MLP.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        p: float = 0.2,
    ):
        super().__init__()

        head_size = emb_dim // num_heads

        self.mhsa = MHSelfAttention(emb_dim, num_heads, head_size, block_size, p=p)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = MLP(emb_dim, p=p)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
