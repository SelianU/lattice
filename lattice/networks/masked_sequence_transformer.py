import torch
import torch.nn as nn


class MaskedSequenceTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        activation="relu",
        causal=False,
        norm_first=True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.causal = bool(causal)
        self.num_heads = num_heads
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, inputs, valid_mask=None):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, sequence, embedding)")
        batch_size, sequence_length, _ = inputs.shape
        key_padding_mask = None
        if valid_mask is not None:
            if tuple(valid_mask.shape) != (batch_size, sequence_length):
                raise ValueError("valid_mask must match the first two input dimensions")
            valid_mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
            if not torch.all(valid_mask.any(dim=1)):
                raise ValueError("each batch row must contain at least one valid token")
            key_padding_mask = ~valid_mask
        attention_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=inputs.device,
                ),
                diagonal=1,
            )
            attention_mask = causal_mask
            if valid_mask is not None:
                invalid = ~valid_mask
                attention_mask = causal_mask.unsqueeze(0) | invalid.unsqueeze(1)
                safe_diagonal = invalid.unsqueeze(-1) & torch.eye(
                    sequence_length,
                    dtype=torch.bool,
                    device=inputs.device,
                ).unsqueeze(0)
                attention_mask = attention_mask & ~safe_diagonal
                attention_mask = (
                    attention_mask.unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(
                        batch_size * self.num_heads,
                        sequence_length,
                        sequence_length,
                    )
                )
                key_padding_mask = None
        outputs = self.encoder(
            inputs,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        if valid_mask is not None:
            outputs = outputs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return outputs
