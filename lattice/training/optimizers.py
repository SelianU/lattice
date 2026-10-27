from typing import Callable, Iterable, Union

import torch
import torch.nn as nn


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossLike = Union[str, LossFunction]
OptimizerLike = Union[str, type, torch.optim.Optimizer]


def make_loss(loss: LossLike) -> LossFunction:
    """
    Build a loss module from a beginner-friendly name or return an existing loss.

    Any callable `(outputs, targets) -> loss` is returned unchanged, including
    `nn.Module` losses and functions such as `torch.nn.functional.cross_entropy`.

    Supported names:
    - "cross_entropy": classification with class-index targets.
    - "mse": regression or value fitting.
    - "smooth_l1": robust regression, commonly used by DQN-style updates.
    """
    if not isinstance(loss, str):
        if callable(loss):
            return loss
        raise TypeError(f"loss must be a name or a callable, got {type(loss).__name__}")

    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss == "mse":
        return nn.MSELoss()
    if loss == "smooth_l1":
        return nn.SmoothL1Loss()

    raise ValueError(
        f"Unknown loss '{loss}'. Available: ['cross_entropy', 'mse', 'smooth_l1']"
    )


def make_optimizer(
    optimizer: OptimizerLike,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float = 1e-3,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """
    Build an optimizer from a name/class or return an existing optimizer.

    Supported names:
    - "adam"
    - "adamw"
    - "sgd"
    - "rmsprop"
    """
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer

    optimizer_cls = optimizer
    if isinstance(optimizer, str):
        optimizer_classes = _optimizer_classes()
        optimizer_cls = optimizer_classes.get(optimizer)
        if optimizer_cls is None:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Available: {sorted(optimizer_classes)}"
            )

    return optimizer_cls(parameters, lr=learning_rate, **optimizer_kwargs)


def _optimizer_classes() -> dict:
    return {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
