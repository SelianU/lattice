"""
Interoperability contract with plain PyTorch.

See docs/torch-interoperability.md. Every test here checks one of two rules:

1. Objects lattice returns are ordinary torch objects.
2. Places where lattice accepts a shortcut also accept the torch object itself.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lattice import (
    make_cnn_mlp,
    make_dqn_agent,
    make_mlp,
    make_mlp_classifier,
    make_rnn,
    make_transformer,
    train_supervised_model,
)
from lattice.training import make_loss, make_optimizer
from lattice.training.supervised_training import evaluate_supervised_model


def _classification_loader():
    torch.manual_seed(0)
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=8)


def _train_one_epoch(model, **kwargs):
    return train_supervised_model(
        model=model,
        train_loader=_classification_loader(),
        epochs=1,
        log_interval=0,
        **kwargs,
    )


# Rule 1: outputs are plain torch objects.


@pytest.mark.parametrize(
    "build_model",
    [
        lambda: make_mlp(input_dim=4, output_dim=2, hidden_dims=[8]),
        lambda: make_mlp_classifier(input_shape=(1, 4, 4), output_dim=2, hidden_dims=[8]),
        lambda: make_cnn_mlp(input_shape=(1, 8, 8), output_dim=2),
        lambda: make_rnn(input_dim=4, output_dim=2),
        lambda: make_transformer(input_dim=4, output_dim=2),
    ],
)
def test_model_builders_return_nn_modules(build_model):
    model = build_model()
    assert isinstance(model, nn.Module)

    copy = build_model()
    copy.load_state_dict(model.state_dict())


def test_model_builders_compose_with_torch_layers():
    model = nn.Sequential(make_mlp(input_dim=4, output_dim=8, hidden_dims=[8]), nn.ReLU(), nn.Linear(8, 2))
    assert model(torch.randn(3, 4)).shape == (3, 2)


def test_make_optimizer_returns_torch_optimizer():
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])
    optimizer = make_optimizer("adam", model.parameters())
    assert isinstance(optimizer, torch.optim.Optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


# Rule 2: shortcuts also accept the torch object.


def test_training_accepts_optimizer_instance():
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    _train_one_epoch(model, optimizer=optimizer)

    after = list(model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_training_accepts_optimizer_class():
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])
    _train_one_epoch(model, optimizer=torch.optim.SGD, learning_rate=0.1, momentum=0.9)


@pytest.mark.parametrize(
    "loss",
    [
        nn.CrossEntropyLoss(label_smoothing=0.1),
        F.cross_entropy,
        lambda outputs, targets: F.cross_entropy(outputs, targets),
    ],
    ids=["module", "functional", "lambda"],
)
def test_training_and_evaluation_accept_torch_losses(loss):
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])
    history = _train_one_epoch(model, loss=loss)
    assert history.last("train_loss") is not None

    metrics = evaluate_supervised_model(model, _classification_loader(), loss=loss)
    assert metrics.num_samples == 16


def test_make_loss_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        make_loss(3)


def test_training_accepts_torch_device():
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])
    _train_one_epoch(model, device=torch.device("cpu"))


def test_training_accepts_plain_torch_model():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    _train_one_epoch(model)


def test_training_accepts_any_iterable_of_batches():
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    model = make_mlp(input_dim=4, output_dim=2, hidden_dims=[8])

    train_supervised_model(model, [(inputs, targets)], epochs=1, log_interval=0)


def test_dqn_agent_accepts_plain_torch_q_network():
    q_network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    agent = make_dqn_agent(action_dim=2, model=q_network, total_timesteps=2000)
    assert agent is not None
