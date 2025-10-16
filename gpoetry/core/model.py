import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from . import config


@dataclass
class GPTConfig:
    """GPT model configuration."""

    vocab_size: int = 0
    num_heads: int = config.NUM_HEADS
    num_layers: int = config.NUM_LAYERS
    block_size: int = config.BLOCK_SIZE
    emb_dim: int = config.EMB_DIM
    p: float = config.DROPOUT_P


class GPTModel(nn.Module):
    """A GPT model."""

    def __init__(self, config: GPTConfig):
        """Initializes the model.

        Args:
            config (GPTConfig): The model configuration.
        """
        super().__init__()
        self.config = config

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
        """Performs a forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Raises:
            ValueError: If the sequence length is greater than the positional embedding size.

        Returns:
            torch.Tensor: The output logits.
        """
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
    """A single self-attention head."""

    def __init__(self, emb_dim: int, head_size: int, block_size: int, p: float = 0.2):
        """Initializes the head.

        Args:
            emb_dim (int): The embedding dimension.
            head_size (int): The head size.
            block_size (int): The block size.
            p (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()

        self.querie = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the head.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        k = self.key(x)
        q = self.querie(x)
        v = self.value(x)

        _, t, d = x.shape

        # sa = softmax(qk^T * sqrt(d))
        wei = (q @ k.mT) * (d**-0.5)

        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # type: ignore[reportIndexIssue]

        out = F.softmax(wei, dim=-1) @ v

        return out


class MHSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        p: float = 0.2,
    ):
        """Initializes the multi-head self-attention module.

        Args:
            emb_dim (int): The embedding dimension.
            num_heads (int): The number of heads.
            head_size (int): The head size.
            block_size (int): The block size.
            p (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, p) for _ in range(num_heads)]
        )
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the multi-head self-attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    """A multi-layer perceptron."""

    def __init__(self, emb_dim: int, p: float = 0.2):
        """Initializes the MLP.

        Args:
            emb_dim (int): The embedding dimension.
            p (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.mlp(x)


class Transformer(nn.Module):
    """A single transformer block."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        p: float = 0.2,
    ):
        """Initializes the transformer block.

        Args:
            emb_dim (int): The embedding dimension.
            num_heads (int): The number of heads.
            block_size (int): The block size.
            p (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()

        head_size = emb_dim // num_heads

        self.mhsa = MHSelfAttention(emb_dim, num_heads, head_size, block_size, p=p)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = MLP(emb_dim, p=p)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the transformer block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
