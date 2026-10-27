"""
Gymnasium reward wrappers.

Both wrappers wrap a raw Gymnasium environment, so they stack with any other
`gymnasium.Wrapper`. Wrap the result in `GymnasiumEnvAdapter` to train on it:

    environment = GymnasiumEnvAdapter(
        RewardNormalizationWrapper(gym.make("Pendulum-v1"), clip=10.0)
    )
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch


class RewardNormalizationWrapper(gym.RewardWrapper):
    """
    Normalises rewards online using Welford's running mean/variance.

    At each step the raw reward is divided by the running standard deviation
    (clipped to ±clip). Useful for continuous-control tasks (SAC on Pendulum,
    etc.) where reward scales vary widely across environments.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        self._clip = clip
        self._epsilon = epsilon
        # Welford's online algorithm state
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0     # accumulated sum of squared deviations from mean

    def reward(self, reward: float) -> float:
        reward = float(reward)
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (reward - self._mean)
        if self._count < 2:
            return reward
        std = (self._M2 / (self._count - 1)) ** 0.5
        return float(np.clip(reward / (std + self._epsilon), -self._clip, self._clip))


class LearnedRewardWrapper(gym.Wrapper):
    """
    Replaces the environment's reward signal with a learned RewardModel.

    Used in D-REX, CLARE, and any reward-learning pipeline where the true
    environment reward is unavailable at training time. The reward for a step
    is computed from the observation the action was taken in.

    Parameters
    ----------
    env          : Any Gymnasium environment.
    reward_model : Trained RewardModel (R(s) or R(s,a)).
    device       : Torch device for reward inference.
    """

    def __init__(self, env: gym.Env, reward_model, device: str = "cpu"):
        super().__init__(env)
        self._reward_model = reward_model
        self._device = torch.device(device)
        self._reward_model.to(self._device)
        self._reward_model.eval()
        self._last_observation: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        with torch.no_grad():
            state_t = torch.from_numpy(
                np.array(self._last_observation, dtype=np.float32)
            ).unsqueeze(0).to(self._device)

            if self._reward_model.uses_action:
                action_t = torch.from_numpy(
                    np.array(action, dtype=np.float32)
                ).unsqueeze(0).to(self._device)
                reward = self._reward_model(state_t, action_t).item()
            else:
                reward = self._reward_model(state_t).item()

        self._last_observation = observation
        return observation, reward, terminated, truncated, info
