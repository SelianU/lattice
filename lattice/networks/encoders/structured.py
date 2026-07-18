from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class StructuredEncoder(nn.Module):
    def __init__(
        self,
        fields: Sequence[str],
        encoders: Mapping[str, nn.Module],
    ):
        super().__init__()
        declared = tuple(fields)
        if (
            not declared
            or any(not isinstance(field, str) or not field for field in declared)
            or len(set(declared)) != len(declared)
            or set(declared) != set(encoders)
        ):
            raise ValueError(
                "fields must be unique, non-empty, and match encoder fields"
            )
        self.fields = declared
        self.encoders = nn.ModuleDict(
            {field: encoders[field] for field in self.fields}
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(inputs, Mapping) or set(inputs) != set(self.fields):
            raise ValueError("input fields must exactly match declared fields")

        outputs: list[torch.Tensor] = []
        expected_prefix: tuple[int, ...] | None = None
        for field in self.fields:
            output = self.encoders[field](inputs[field])
            if not isinstance(output, torch.Tensor) or output.ndim == 0:
                raise TypeError("field encoders must return non-scalar tensors")
            prefix = tuple(output.shape[:-1])
            if expected_prefix is None:
                expected_prefix = prefix
            elif prefix != expected_prefix:
                raise ValueError(
                    "encoded fields must have matching batch/time prefixes"
                )
            outputs.append(output)
        return torch.cat(outputs, dim=-1)
