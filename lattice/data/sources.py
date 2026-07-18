from __future__ import annotations

from collections.abc import Iterator, Mapping
import importlib
from typing import Protocol, runtime_checkable

import numpy as np

from lattice.spaces import ActionAdapter, ObservationAdapter

from .trajectories import EncodedTrajectory


@runtime_checkable
class OfflineTrajectorySource(Protocol):
    def iter_trajectories(self) -> Iterator[EncodedTrajectory]: ...


def _time_length(value: object) -> int:
    if isinstance(value, Mapping):
        lengths = {_time_length(child) for child in value.values()}
        if len(lengths) != 1:
            raise ValueError("episode fields must align in time")
        return lengths.pop() if lengths else 0
    if isinstance(value, tuple):
        lengths = {_time_length(child) for child in value}
        if len(lengths) != 1:
            raise ValueError("episode fields must align in time")
        return lengths.pop() if lengths else 0
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("episode fields must align in time")
    return int(array.shape[0])


def _time_item(value: object, index: int) -> object:
    if isinstance(value, Mapping):
        return {name: _time_item(child, index) for name, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_time_item(child, index) for child in value)
    return np.asarray(value)[index]


class MinariTrajectorySource:
    def __init__(
        self,
        dataset,
        observation_adapter: ObservationAdapter,
        action_adapter: ActionAdapter,
    ):
        self.dataset = dataset
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter

    @classmethod
    def from_id(
        cls,
        dataset_id: str,
        observation_adapter: ObservationAdapter,
        action_adapter: ActionAdapter,
    ) -> "MinariTrajectorySource":
        try:
            minari = importlib.import_module("minari")
        except ModuleNotFoundError as error:
            raise ImportError(
                "Minari support requires installing lattice[offline]"
            ) from error
        dataset = minari.load_dataset(dataset_id)
        return cls(dataset, observation_adapter, action_adapter)

    def iter_trajectories(self) -> Iterator[EncodedTrajectory]:
        for episode in self.dataset.iterate_episodes():
            transition_count = _time_length(episode.actions)
            observation_count = _time_length(episode.observations)
            field_lengths = (
                _time_length(episode.rewards),
                _time_length(episode.terminations),
                _time_length(episode.truncations),
            )
            if (
                any(length != transition_count for length in field_lengths)
                or observation_count not in {transition_count, transition_count + 1}
            ):
                raise ValueError("episode fields must align in time")

            observations = np.stack(
                [
                    self.observation_adapter.encode(
                        _time_item(episode.observations, index)
                    )
                    for index in range(transition_count)
                ]
            )
            actions = np.stack(
                [
                    self.action_adapter.encode(_time_item(episode.actions, index))
                    for index in range(transition_count)
                ]
            )
            yield EncodedTrajectory(
                observations=observations,
                actions=actions,
                rewards=episode.rewards,
                terminated=episode.terminations,
                truncated=episode.truncations,
            )
