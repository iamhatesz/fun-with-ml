import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention = query @ key.transpose(2, 1)
    d_k = key.shape[-1]
    scaled = attention / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.masked_fill(mask == 0, -float("inf"))
    normalized = F.softmax(scaled, dim=1)
    return normalized @ value


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, head_dim)
        self.k = nn.Linear(hidden_dim, head_dim)
        self.v = nn.Linear(hidden_dim, head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attention = scaled_dot_product_attention(
            self.q(hidden), self.k(hidden), self.v(hidden)
        )
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        scaled_head_dim = head_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(hidden_dim, scaled_head_dim) for _ in range(num_heads)]
        )
        self.fc = nn.Linear(head_dim, head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attention = torch.cat([head(hidden) for head in self.heads], dim=-1)
        attention = self.fc(attention)
        return attention


class Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        fc_hidden_dim: int,
        prob: float,
    ):
        super().__init__()
        self.heads = MultiHeadAttention(hidden_dim, head_dim, num_heads)
        self.attention_drop = nn.Dropout(prob)
        self.attention_norm = nn.LayerNorm(head_dim)
        self.fc1 = nn.Linear(head_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, head_dim)
        self.fc_drop = nn.Dropout(prob)
        self.fc_norm = nn.LayerNorm(head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attention = self.heads(hidden)
        attention = self.attention_drop(attention)
        attention = attention + hidden
        attention = self.attention_norm(attention)

        fc = self.fc2(F.gelu(self.fc1(attention)))
        fc = self.fc_drop(fc)
        fc = fc + attention
        fc = self.fc_norm(fc)

        return fc


class Embeddings(nn.Module):
    def __init__(
        self, vocab_size: int, max_context_size: int, emb_dim: int, drop: float
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_context_size, emb_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=tokens.device
        ).unsqueeze(0)

        token_emb = self.tok_emb(tokens)
        pos_emb = self.pos_emb(position_ids)
        emb = token_emb + pos_emb
        emb_drop = self.drop(emb)
        return emb_drop


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        max_context_size: int,
        num_blocks: int,
        num_heads: int,
        fc_hidden_dim: int,
        drop: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = Embeddings(vocab_size, max_context_size, emb_dim, drop)
        hidden_dim = head_dim = emb_dim
        self.stacks = nn.Sequential(
            *[
                Block(hidden_dim, head_dim, num_heads, fc_hidden_dim, drop)
                for _ in range(num_blocks)
            ]
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        summary(self)

    def forward(self, tokens: torch.Tensor):
        embeddings = self.embeddings(tokens)
        hidden = self.stacks(embeddings)
        logits = self.fc(hidden)
        return logits
