"""
Termination vs truncation.

A terminated transition has no future value. A truncated one (for example a
time limit) does, so targets must still bootstrap from the next state.
"""

import numpy as np
import pytest
import torch

from lattice import make_ppo_agent
from lattice.algorithms import ReplayBuffer, SequenceReplayBuffer
from lattice.training import EnvTransition, RLTrainer


def test_replay_buffer_bootstraps_truncated_but_not_terminated_transitions():
    buffer = ReplayBuffer(capacity=2)
    buffer.push(np.zeros(2), 0, 1.0, np.ones(2), terminated=False, truncated=True)
    buffer.push(np.zeros(2), 0, 1.0, np.ones(2), terminated=True, truncated=False)

    *_, terminated = buffer.sample(2)

    assert sorted(terminated.squeeze(1).tolist()) == [0.0, 1.0]


def test_sequence_buffer_ends_episodes_on_truncation_without_marking_them_terminal():
    buffer = SequenceReplayBuffer(capacity_episodes=4)
    buffer.push(np.zeros(2), 0, 1.0, np.ones(2), terminated=False, truncated=False)
    buffer.push(np.zeros(2), 0, 1.0, np.ones(2), terminated=False, truncated=True)

    assert len(buffer) == 1
    _, _, _, _, terminated, masks = buffer.sample(batch_size=1, seq_len=2)
    assert terminated.squeeze(-1).tolist() == [[0.0, 0.0]]
    assert masks.squeeze(-1).tolist() == [[1.0, 1.0]]


def test_ppo_folds_next_state_value_into_reward_only_when_truncated():
    torch.manual_seed(0)
    agent = make_ppo_agent(state_dim=2, action_dim=2, num_steps=8, batch_size=4, device="cpu")
    state = np.zeros(2, dtype=np.float32)
    next_state = np.ones(2, dtype=np.float32)
    next_value = agent._state_value(next_state)

    agent.select_action(state)
    agent.observe(state, 0, 1.0, next_state, terminated=False, truncated=True)
    agent.select_action(state)
    agent.observe(state, 0, 1.0, next_state, terminated=True, truncated=False)

    rewards = agent.rollout_buffer._rewards
    assert rewards[0] == pytest.approx(1.0 + agent.config.gamma * next_value)
    assert rewards[1] == 1.0
    assert agent.rollout_buffer._dones == [1.0, 1.0]


def test_trainer_passes_terminated_and_truncated_to_the_agent():
    class RecordingAgent:
        config = None

        def __init__(self):
            self.observed = []

        def select_action(self, state, evaluation=False):
            return 0

        def observe(self, state, action, reward, next_state, terminated, truncated):
            self.observed.append((terminated, truncated))

        def update(self):
            return {}

    class TimeLimitEnvironment:
        def reset(self):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return EnvTransition(np.zeros(1, dtype=np.float32), 1.0, False, True, {})

        def close(self):
            return None

    class Config:
        total_timesteps = 2
        log_interval = 0
        eval_interval = 0

    agent = RecordingAgent()
    RLTrainer(agent, TimeLimitEnvironment(), Config(), logger=_SilentLogger()).train()

    assert agent.observed == [(False, True), (False, True)]


class _SilentLogger:
    def log_episode(self, episode, step, metrics):
        return None

    def close(self):
        return None
