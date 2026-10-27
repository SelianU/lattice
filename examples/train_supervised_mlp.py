"""
Train an MLP classifier on a synthetic two-spiral dataset.

Run from the repository root:

    python examples/train_supervised_mlp.py
"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from lattice import SupervisedTrainingConfig, make_mlp, train_supervised_model
from lattice.training.supervised_training import evaluate_supervised_model


def make_spiral_dataset(points_per_class: int = 500, noise: float = 0.1) -> TensorDataset:
    """Two interleaved spirals: a classic dataset a linear model cannot solve."""
    radius = torch.linspace(0.5, 4.0, points_per_class)
    angle = radius * math.pi

    spiral_points = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    inputs = torch.cat([spiral_points, -spiral_points])
    inputs = inputs + noise * torch.randn_like(inputs)
    targets = torch.cat([
        torch.zeros(points_per_class, dtype=torch.long),
        torch.ones(points_per_class, dtype=torch.long),
    ])
    return TensorDataset(inputs, targets)


def stop_when_accurate(metrics, history, model):
    """Callbacks return True to stop training early."""
    return metrics.validation_accuracy is not None and metrics.validation_accuracy >= 0.98


def train():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = make_spiral_dataset()
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256)

    # make_mlp returns a plain nn.Sequential: input -> hidden layers -> class logits.
    model = make_mlp(input_dim=2, output_dim=2, hidden_dims=[64, 64, 64])

    training_config = SupervisedTrainingConfig(
        loss="cross_entropy",
        optimizer="adam",
        learning_rate=3e-3,
        epochs=100,
        device=device,
        log_interval=10,
    )
    history = train_supervised_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=training_config,
        callbacks=[stop_when_accurate],
    )

    validation_metrics = evaluate_supervised_model(model, validation_loader, device=device)
    print(f"Trained for {len(history.epochs)} epochs")
    print(f"Final validation accuracy: {validation_metrics.accuracy:.2%}")

    # The trained model is a regular PyTorch module, so inference is plain torch.
    model.eval()
    # (2, 0) lies on the class-0 spiral and (-2, 0) on the class-1 spiral.
    new_points = torch.tensor([[2.0, 0.0], [-2.0, 0.0]], device=device)
    with torch.no_grad():
        predicted_classes = model(new_points).argmax(dim=-1)
    print(f"Predicted classes for {new_points.tolist()}: {predicted_classes.tolist()}")


if __name__ == "__main__":
    train()
