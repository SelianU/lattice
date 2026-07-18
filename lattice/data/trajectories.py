from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np


def _owned_array(value, *, dtype) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True)


def _freeze(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclasses.dataclass(frozen=True)
class EncodedTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __post_init__(self) -> None:
        observations = _owned_array(self.observations, dtype=np.float32)
        actions = _owned_array(self.actions, dtype=np.float32)
        rewards = _owned_array(self.rewards, dtype=np.float32)
        terminated = _owned_array(self.terminated, dtype=np.bool_)
        truncated = _owned_array(self.truncated, dtype=np.bool_)

        if observations.ndim != 2:
            raise ValueError("observations must be two-dimensional")
        if actions.ndim != 2:
            raise ValueError("actions must be two-dimensional")
        for name, array in (
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")

        lengths = {
            observations.shape[0],
            actions.shape[0],
            rewards.shape[0],
            terminated.shape[0],
            truncated.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("trajectory fields must have matching time lengths")
        if observations.shape[0] == 0:
            raise ValueError("trajectory must not be empty")
        if observations.shape[1] == 0 or actions.shape[1] == 0:
            raise ValueError("trajectory feature dimensions must be positive")
        if not (
            np.isfinite(observations).all()
            and np.isfinite(actions).all()
            and np.isfinite(rewards).all()
        ):
            raise ValueError("trajectory numeric values must be finite")

        for name, array in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            object.__setattr__(self, name, _freeze(array))

    @property
    def length(self) -> int:
        return self.observations.shape[0]


@dataclasses.dataclass(frozen=True)
class TrajectoryBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        observations = _owned_array(self.observations, dtype=np.float32)
        actions = _owned_array(self.actions, dtype=np.float32)
        rewards = _owned_array(self.rewards, dtype=np.float32)
        terminated = _owned_array(self.terminated, dtype=np.bool_)
        truncated = _owned_array(self.truncated, dtype=np.bool_)
        valid_mask = _owned_array(self.valid_mask, dtype=np.bool_)

        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("batched observations and actions must be three-dimensional")
        prefix = observations.shape[:2]
        if actions.shape[:2] != prefix:
            raise ValueError("batched fields must share batch and time dimensions")
        for array in (rewards, terminated, truncated, valid_mask):
            if array.shape != prefix:
                raise ValueError("batched fields must share batch and time dimensions")
        if not (
            np.isfinite(observations).all()
            and np.isfinite(actions).all()
            and np.isfinite(rewards).all()
        ):
            raise ValueError("batch numeric values must be finite")

        for name, array in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
            ("valid_mask", valid_mask),
        ):
            object.__setattr__(self, name, _freeze(array))


def collate_trajectories(
    trajectories: Iterable[EncodedTrajectory], *, padding_side: str = "left"
) -> TrajectoryBatch:
    items = tuple(trajectories)
    if not items:
        raise ValueError("collate_trajectories requires at least one trajectory")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    if not all(isinstance(item, EncodedTrajectory) for item in items):
        raise TypeError("all items must be EncodedTrajectory instances")

    observation_dims = {item.observations.shape[1] for item in items}
    if len(observation_dims) != 1:
        raise ValueError("observation feature dimensions must match")
    action_dims = {item.actions.shape[1] for item in items}
    if len(action_dims) != 1:
        raise ValueError("action feature dimensions must match")

    batch_size = len(items)
    max_length = max(item.length for item in items)
    observation_dim = observation_dims.pop()
    action_dim = action_dims.pop()
    observations = np.zeros(
        (batch_size, max_length, observation_dim), dtype=np.float32
    )
    actions = np.zeros((batch_size, max_length, action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size, max_length), dtype=np.float32)
    terminated = np.zeros((batch_size, max_length), dtype=np.bool_)
    truncated = np.zeros((batch_size, max_length), dtype=np.bool_)
    valid_mask = np.zeros((batch_size, max_length), dtype=np.bool_)

    for row, trajectory in enumerate(items):
        start = max_length - trajectory.length if padding_side == "left" else 0
        target = slice(start, start + trajectory.length)
        observations[row, target] = trajectory.observations
        actions[row, target] = trajectory.actions
        rewards[row, target] = trajectory.rewards
        terminated[row, target] = trajectory.terminated
        truncated[row, target] = trajectory.truncated
        valid_mask[row, target] = True

    return TrajectoryBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        valid_mask=valid_mask,
    )
