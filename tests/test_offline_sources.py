from collections import OrderedDict
import importlib

import gymnasium as gym
import numpy as np
import pytest

from lattice.data import MinariTrajectorySource, OfflineTrajectorySource
from lattice.spaces import ActionAdapter, ObservationAdapter


class FakeEpisode:
    def __init__(self):
        self.observations = {
            "state": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            "goal": np.array([[7], [8], [9]], dtype=np.float32),
        }
        self.actions = np.array([[0.25], [-0.5]], dtype=np.float32)
        self.rewards = np.array([1, 2], dtype=np.float32)
        self.terminations = np.array([False, True])
        self.truncations = np.array([False, False])


class FakeDataset:
    def iterate_episodes(self):
        yield FakeEpisode()


def _adapters():
    observation_space = gym.spaces.Dict(
        OrderedDict(
            (
                (
                    "goal",
                    gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
                ),
                (
                    "state",
                    gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32),
                ),
            )
        )
    )
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    return ObservationAdapter(observation_space), ActionAdapter(action_space)


def test_minari_source_encodes_t_plus_one_observations_as_transitions():
    observation_adapter, action_adapter = _adapters()
    source = MinariTrajectorySource(
        FakeDataset(), observation_adapter, action_adapter
    )

    trajectories = list(source.iter_trajectories())

    assert isinstance(source, OfflineTrajectorySource)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory.observations.shape == (2, 3)
    assert trajectory.actions.shape == (2, 1)
    np.testing.assert_array_equal(
        trajectory.observations,
        np.array([[7, 1, 2], [8, 3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(trajectory.rewards, np.array([1, 2]))
    assert trajectory.terminated.tolist() == [False, True]


def test_minari_source_rejects_unaligned_episode_fields():
    observation_adapter, action_adapter = _adapters()
    episode = FakeEpisode()
    episode.rewards = np.array([1], dtype=np.float32)

    class BadDataset:
        def iterate_episodes(self):
            yield episode

    source = MinariTrajectorySource(
        BadDataset(), observation_adapter, action_adapter
    )

    with pytest.raises(ValueError, match="episode fields must align"):
        list(source.iter_trajectories())


def test_from_id_reports_the_offline_extra_when_minari_is_missing(monkeypatch):
    observation_adapter, action_adapter = _adapters()
    real_import = importlib.import_module

    def missing_minari(name, package=None):
        if name == "minari":
            raise ModuleNotFoundError("minari")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_minari)

    with pytest.raises(
        ImportError, match=r"Minari support requires installing lattice\[offline\]"
    ):
        MinariTrajectorySource.from_id(
            "example-dataset-v0", observation_adapter, action_adapter
        )


def test_from_id_loads_dataset_lazily(monkeypatch):
    observation_adapter, action_adapter = _adapters()
    loaded = []

    class FakeMinariModule:
        @staticmethod
        def load_dataset(dataset_id):
            loaded.append(dataset_id)
            return FakeDataset()

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: FakeMinariModule if name == "minari" else None,
    )

    source = MinariTrajectorySource.from_id(
        "example-dataset-v0", observation_adapter, action_adapter
    )

    assert loaded == ["example-dataset-v0"]
    assert len(list(source.iter_trajectories())) == 1
