# Torch Interoperability

`lattice` aims to be a strict superset of PyTorch: anything you can do with
plain `torch` still works inside `lattice`, and common tasks take less code.

`lattice` does not re-export or wrap the PyTorch API. Use `torch.nn`,
`torch.optim`, and `torch.utils.data` directly; `lattice` adds shortcuts on top.

## The Two Rules

### 1. Outputs are plain torch objects

Everything `lattice` builds is an ordinary PyTorch object, so you can leave
`lattice` at any point and keep working in plain `torch`.

| Helper | Returns |
|---|---|
| `make_mlp`, `make_mlp_classifier`, `make_cnn_mlp` | `nn.Sequential` |
| `make_rnn`, `make_transformer`, `build_model` | `nn.Module` |
| `make_optimizer` | `torch.optim.Optimizer` |
| `make_loss` | the loss callable |

This means `state_dict`, `torch.save`, `torch.optim.lr_scheduler`, and
composition with `nn.Sequential` all work without `lattice` knowing about them.

```python
import torch.nn as nn

from lattice import make_mlp

model = nn.Sequential(make_mlp(input_dim=4, output_dim=8), nn.ReLU(), nn.Linear(8, 2))
```

### 2. Shortcuts also accept the torch object

Wherever `lattice` accepts a beginner-friendly shortcut, it also accepts the
torch object that the shortcut stands for.

| Argument | Shortcut | Also accepts |
|---|---|---|
| `model` | `make_mlp(...)` | any `nn.Module` |
| `loss` | `"cross_entropy"`, `"mse"`, `"smooth_l1"` | `nn.Module` losses, functions such as `F.cross_entropy`, any `(outputs, targets) -> loss` callable |
| `optimizer` | `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"` | an optimizer class, or a built `torch.optim.Optimizer` instance |
| `device` | `"cpu"`, `"cuda"` | `torch.device` |
| `train_loader` | — | any `DataLoader`, or any iterable of `(inputs, targets)` |
| RL `model` | `backbone="mlp"` | any `nn.Module` with the expected input and output shapes |

```python
import torch
import torch.nn.functional as F

from lattice import train_supervised_model

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
train_supervised_model(model, train_loader, loss=F.cross_entropy, optimizer=optimizer)
```

## Checking New Features

Every new public helper should satisfy both rules. When adding one:

1. Return a torch object, not a `lattice`-only wrapper.
2. If an argument takes a string shortcut, also accept the torch object.
3. Add a case to `tests/test_torch_interoperability.py`.

## Known Gaps

These are plain PyTorch workflows that `train_supervised_model` does not
support yet. Until they are added, write the loop yourself or use
`training_step_function`.

- Learning-rate schedulers (`torch.optim.lr_scheduler`)
- Mixed precision (`torch.autocast`, `torch.amp.GradScaler`)
- Gradient accumulation
- Saving and resuming supervised training checkpoints
- Batches that are not `(inputs, targets)` tuples, such as dictionaries
- `torch.compile`
- Multi-GPU training (`DistributedDataParallel`)
