import torch
from torch import nn
from torch.nn import functional as F


class GPoeTry(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        num_layers: int = 4,
        block_size: int = 256,
        emb_dim: int = 256,
        p: float = 0.2,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)

        layers = [
            TransformerLayer(
                emb_dim,
                num_heads,
                block_size,
                p=p,
            )
            for _ in range(num_layers)
        ]

        self.blocks = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1]))

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


class Head(nn.Module):
    def __init__(self, emb_dim: int, head_size: int, block_size: int, p: float = 0.2):
        super().__init__()

        self.querie = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class FeedForward(nn.Module):
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


class TransformerLayer(nn.Module):
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
        self.ffn = FeedForward(emb_dim, p=p)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
