import pytest
import torch

from lattice import SequenceTransformer


def make_stack(causal):
    torch.manual_seed(3)
    model = SequenceTransformer(
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        causal=causal,
        norm_first=True,
    )
    return model.eval()


def test_causal_prefix_ignores_future_tokens():
    model = make_stack(causal=True)
    first = torch.randn(1, 4, 4)
    second = first.clone()
    second[:, 2:, 0] += 100.0
    assert torch.allclose(model(first)[:, :2], model(second)[:, :2], atol=1e-6)


def test_bidirectional_prefix_uses_future_tokens():
    model = make_stack(causal=False)
    first = torch.randn(1, 4, 4)
    second = first.clone()
    second[:, 2:, 0] += 100.0
    assert not torch.allclose(model(first)[:, :2], model(second)[:, :2])


def test_padding_values_do_not_change_valid_outputs():
    model = make_stack(causal=False)
    valid_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    first = torch.randn(1, 4, 4)
    second = first.clone()
    second[:, 2:, 0] += 100.0
    assert torch.allclose(
        model(first, valid_mask)[:, :2],
        model(second, valid_mask)[:, :2],
        atol=1e-6,
    )


def test_rejects_fully_padded_rows():
    with pytest.raises(ValueError, match="at least one valid token"):
        make_stack(causal=False)(torch.zeros(1, 2, 4), torch.zeros(1, 2))
