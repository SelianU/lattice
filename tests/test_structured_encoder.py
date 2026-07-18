from collections import OrderedDict

import pytest
import torch
from torch import nn

from lattice.networks.encoders import StructuredEncoder


def _linear(in_features, weights):
    layer = nn.Linear(in_features, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([weights], dtype=torch.float32))
    return layer


def test_structured_encoder_concatenates_in_declared_field_order():
    encoder = StructuredEncoder(
        fields=("goal", "state"),
        encoders=OrderedDict(
            (
                ("state", _linear(2, [1, 1])),
                ("goal", _linear(1, [2])),
            )
        ),
    )
    inputs = OrderedDict(
        (
            ("state", torch.tensor([[3.0, 4.0]])),
            ("goal", torch.tensor([[5.0]])),
        )
    )

    output = encoder(inputs)

    torch.testing.assert_close(output, torch.tensor([[10.0, 7.0]]))


def test_structured_encoder_preserves_matching_batch_and_time_prefixes():
    encoder = StructuredEncoder(
        fields=("goal", "state"),
        encoders={"goal": nn.Identity(), "state": nn.Identity()},
    )

    output = encoder(
        {
            "goal": torch.ones(2, 3, 1),
            "state": torch.ones(2, 3, 2),
        }
    )

    assert output.shape == (2, 3, 3)


@pytest.mark.parametrize(
    "inputs",
    [
        {"goal": torch.ones(1, 1)},
        {
            "goal": torch.ones(1, 1),
            "state": torch.ones(1, 1),
            "extra": torch.ones(1, 1),
        },
    ],
)
def test_structured_encoder_requires_exact_runtime_fields(inputs):
    encoder = StructuredEncoder(
        fields=("goal", "state"),
        encoders={"goal": nn.Identity(), "state": nn.Identity()},
    )

    with pytest.raises(ValueError, match="input fields"):
        encoder(inputs)


def test_structured_encoder_rejects_unequal_output_prefixes():
    encoder = StructuredEncoder(
        fields=("goal", "state"),
        encoders={"goal": nn.Identity(), "state": nn.Identity()},
    )

    with pytest.raises(ValueError, match="batch/time prefixes"):
        encoder(
            {
                "goal": torch.ones(2, 3, 1),
                "state": torch.ones(2, 4, 1),
            }
        )


@pytest.mark.parametrize(
    ("fields", "encoders"),
    [
        (("goal", "goal"), {"goal": nn.Identity()}),
        (("goal", "state"), {"goal": nn.Identity()}),
        (("goal",), {"goal": nn.Identity(), "extra": nn.Identity()}),
        ((), {}),
    ],
)
def test_structured_encoder_validates_constructor_fields(fields, encoders):
    with pytest.raises(ValueError, match="fields"):
        StructuredEncoder(fields=fields, encoders=encoders)
