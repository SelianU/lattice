import pytest
import torch

from lattice.training import make_optimizer


def test_make_optimizer_builds_adamw():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = make_optimizer("adamw", [parameter], learning_rate=1e-4, weight_decay=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_unknown_optimizer_lists_adamw():
    with pytest.raises(ValueError, match="adamw"):
        make_optimizer("not_real", [], learning_rate=1e-3)
