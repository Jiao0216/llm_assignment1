"""A compact decoder-only Transformer language model."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.attention = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(1)
        output, _ = self.attention(
            inputs,
            inputs,
            inputs,
            attn_mask=self.causal_mask[:length, :length],
            need_weights=False,
        )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError(f"input length {length} exceeds block_size {self.config.block_size}")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        logits = self.output_head(self.final_norm(self.blocks(hidden)))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)