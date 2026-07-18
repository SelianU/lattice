from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest

from lattice.training import GymnasiumEnvAdapter


class DictEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
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
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(1,), dtype=np.float32
        )
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {
            "state": np.array([1, 2], dtype=np.float32),
            "goal": np.array([3], dtype=np.float32),
        }, {"seed": seed, "options": options}

    def step(self, action):
        self.last_action = action
        return (
            {
                "state": np.array([4, 5], dtype=np.float32),
                "goal": np.array([6], dtype=np.float32),
            },
            1.0,
            False,
            True,
            {"native": True},
        )


def test_adapter_encodes_dict_observations_and_decodes_native_actions():
    env = DictEnvironment()
    adapter = GymnasiumEnvAdapter(env)

    observation, info = adapter.reset(seed=7, options={"mode": "test"})
    transition = adapter.step(np.array([3.0], dtype=np.float32))

    np.testing.assert_array_equal(observation, np.array([3, 1, 2], np.float32))
    assert info == {"seed": 7, "options": {"mode": "test"}}
    np.testing.assert_array_equal(
        transition.observation, np.array([6, 4, 5], np.float32)
    )
    assert transition.reward == 1.0
    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.info == {"native": True}
    np.testing.assert_array_equal(env.last_action, np.array([1.0], np.float32))


@pytest.mark.parametrize("reward", [True, np.nan, np.inf, [1.0]])
def test_adapter_rejects_non_finite_or_non_scalar_rewards(reward):
    class BadRewardEnvironment(DictEnvironment):
        def step(self, action):
            observation, _, terminated, truncated, info = super().step(action)
            return observation, reward, terminated, truncated, info

    adapter = GymnasiumEnvAdapter(BadRewardEnvironment())
    adapter.reset()

    with pytest.raises(ValueError, match="finite scalar"):
        adapter.step(np.array([0.0], dtype=np.float32))


def test_adapter_requires_boolean_flags_and_mapping_info():
    class BadStepEnvironment(DictEnvironment):
        def step(self, action):
            observation, reward, _, _, _ = super().step(action)
            return observation, reward, 0, False, []

    adapter = GymnasiumEnvAdapter(BadStepEnvironment())
    adapter.reset()

    with pytest.raises(TypeError, match="terminated must be boolean"):
        adapter.step(np.array([0.0], dtype=np.float32))


def test_adapter_requires_mapping_reset_info_and_closes_environment():
    class BadResetEnvironment(DictEnvironment):
        def reset(self, *, seed=None, options=None):
            observation, _ = super().reset(seed=seed, options=options)
            return observation, []

        def close(self):
            self.closed = True

    env = BadResetEnvironment()
    adapter = GymnasiumEnvAdapter(env)

    with pytest.raises(TypeError, match="reset info must be a mapping"):
        adapter.reset()
    adapter.close()
    assert env.closed is True
