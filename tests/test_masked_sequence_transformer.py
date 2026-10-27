import pytest
import torch

from lattice import MaskedSequenceTransformer


def make_stack(causal):
    torch.manual_seed(3)
    model = MaskedSequenceTransformer(
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


def test_causal_left_padding_is_finite_and_invariant_in_eval_fast_path():
    model = make_stack(causal=True)
    valid_mask = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 1]],
        dtype=torch.bool,
    )
    inputs = torch.randn(2, 4, 4)
    changed_padding = inputs.clone()
    changed_padding[~valid_mask] += 100.0

    with torch.no_grad():
        outputs = model(inputs, valid_mask)
        changed_outputs = model(changed_padding, valid_mask)

    assert torch.isfinite(outputs).all()
    assert torch.equal(outputs[~valid_mask], torch.zeros_like(outputs[~valid_mask]))
    assert torch.allclose(
        outputs[valid_mask],
        changed_outputs[valid_mask],
        atol=1e-6,
    )


def test_causal_left_padding_keeps_valid_loss_gradients_and_step_finite():
    model = make_stack(causal=True).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    valid_mask = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 1]],
        dtype=torch.bool,
    )
    outputs = model(torch.randn(2, 4, 4), valid_mask)

    assert torch.isfinite(outputs).all()
    assert torch.equal(outputs[~valid_mask], torch.zeros_like(outputs[~valid_mask]))
    loss = outputs[valid_mask].square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
