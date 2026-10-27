import gymnasium as gym
import numpy as np
import pytest

from lattice.training import (
    EnvTransition,
    GymnasiumEnvAdapter,
    LearnedRewardWrapper,
    RewardNormalizationWrapper,
    build_trainer,
    run_interaction_step,
)


class ImageEnvironment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(3, 8, 8), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3, start=1)

    def __init__(self):
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((3, 8, 8), dtype=np.uint8), {}

    def step(self, action):
        self.last_action = action
        return np.ones((3, 8, 8), dtype=np.uint8), 2.0, False, False, {}


class DiscreteObservationEnvironment(gym.Env):
    observation_space = gym.spaces.Discrete(4)
    action_space = gym.spaces.Box(-2, 2, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 2, {}

    def step(self, action):
        return 3, 0.0, True, False, {}


def test_box_observations_keep_native_shape_and_dtype():
    adapter = GymnasiumEnvAdapter(ImageEnvironment())

    observation, _ = adapter.reset()

    assert observation.shape == (3, 8, 8)
    assert observation.dtype == np.uint8
    assert adapter.observation_shape == (3, 8, 8)


def test_discrete_actions_accept_integer_index_and_respect_start():
    environment = ImageEnvironment()
    adapter = GymnasiumEnvAdapter(environment)
    adapter.reset()

    adapter.step(np.int64(2))

    assert environment.last_action == 3
    assert adapter.action_dim == 3
    assert adapter.is_continuous is False
    assert adapter.action_low is None


@pytest.mark.parametrize("action", [3, -1, 1.0, True])
def test_discrete_actions_reject_invalid_indices(action):
    adapter = GymnasiumEnvAdapter(ImageEnvironment())
    adapter.reset()

    with pytest.raises(ValueError, match="Discrete action index"):
        adapter.step(action)


def test_non_box_observations_are_encoded_as_flat_vectors():
    adapter = GymnasiumEnvAdapter(DiscreteObservationEnvironment())

    observation, _ = adapter.reset()

    np.testing.assert_array_equal(observation, np.array([0, 0, 1, 0], np.float32))
    assert adapter.observation_shape == (4,)
    assert adapter.is_continuous is True
    assert adapter.action_dim == 1
    assert (adapter.action_low, adapter.action_high) == (-2.0, 2.0)


def test_run_interaction_step_accepts_gymnasium_tuples_and_env_transitions():
    class FixedAgent:
        config = None

        def __init__(self):
            self.observed = []

        def select_action(self, state, evaluation=False):
            return 0

        def observe(self, *transition):
            self.observed.append(transition)

        def update(self):
            return {}

    agent = FixedAgent()
    state = np.zeros(2, dtype=np.float32)
    next_state = np.ones(2, dtype=np.float32)

    from_tuple = run_interaction_step(
        agent, state, lambda action: (next_state, 1.0, False, True, {})
    )
    from_transition = run_interaction_step(
        agent,
        state,
        lambda action: EnvTransition(next_state, 1.0, True, False, {}),
    )

    assert (from_tuple.transition.terminated, from_tuple.transition.truncated) == (False, True)
    assert (from_transition.transition.terminated, from_transition.transition.truncated) == (True, False)
    assert from_tuple.transition.done and from_transition.transition.done
    assert len(agent.observed) == 2


def test_run_interaction_step_rejects_legacy_four_value_results():
    class FixedAgent:
        config = None

        def select_action(self, state, evaluation=False):
            return 0

    with pytest.raises(ValueError, match="terminated, truncated"):
        run_interaction_step(
            FixedAgent(),
            np.zeros(2),
            lambda action: (np.zeros(2), 0.0, True, {}),
        )


def test_reward_normalization_wrapper_is_a_gymnasium_wrapper():
    environment = RewardNormalizationWrapper(gym.make("Pendulum-v1"), clip=1.0)
    environment.reset(seed=0)

    rewards = [environment.step(environment.action_space.sample())[1] for _ in range(20)]

    assert isinstance(environment, gym.Wrapper)
    assert all(-1.0 <= reward <= 1.0 for reward in rewards)


def test_learned_reward_wrapper_scores_the_observation_the_action_was_taken_in():
    import torch

    class FirstFeatureReward(torch.nn.Module):
        uses_action = False

        def forward(self, states):
            return states[:, 0]

    environment = LearnedRewardWrapper(ImageObservationFreeEnvironment(), FirstFeatureReward())
    environment.reset()

    _, reward, _, _, _ = environment.step(0)
    _, next_reward, _, _, _ = environment.step(0)

    assert reward == 0.0
    assert next_reward == 1.0


class ImageObservationFreeEnvironment(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0.0
        return np.array([self.position], dtype=np.float32), {}

    def step(self, action):
        self.position += 1.0
        return np.array([self.position], dtype=np.float32), -5.0, False, False, {}


@pytest.mark.parametrize(
    "algorithm, environment_name",
    [("dqn", "CartPole-v1"), ("ppo", "CartPole-v1"), ("sac", "Pendulum-v1"), ("td3", "Pendulum-v1")],
)
def test_build_trainer_runs_on_gymnasium_environments(algorithm, environment_name):
    config = {"total_timesteps": 64, "device": "cpu", "log_interval": 0}
    if algorithm == "ppo":
        config.update(num_steps=32, batch_size=16)
    else:
        config.update(learning_starts=16, batch_size=16)

    trainer = build_trainer({
        "algorithm": algorithm,
        "env": {"type": "gym", "name": environment_name},
        "model": {"backbone": {"type": "mlp", "hidden_dims": [8]}},
        "config": config,
    })

    history = trainer.train()

    assert isinstance(trainer.env, GymnasiumEnvAdapter)
    assert trainer.evaluate(num_episodes=1)["mean_reward"] <= 500
    assert history is not None
