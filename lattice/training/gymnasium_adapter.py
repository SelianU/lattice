from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math

import gymnasium as gym
import numpy as np

from lattice.spaces import ActionAdapter, ObservationAdapter


@dataclasses.dataclass(frozen=True)
class EnvTransition:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, object]


class GymnasiumEnvAdapter:
    """
    Connect a Gymnasium-style environment to lattice agents.

    Without explicit adapters, encoding is chosen from the spaces:

    - Box observations keep their native shape and dtype, so image models
      receive `(channels, height, width)` arrays.
    - Other observations (Discrete, MultiDiscrete, Dict, Tuple) are encoded
      into one flat float32 vector by `ObservationAdapter`.
    - Discrete actions accept an integer index or a score vector.
    - Other actions are decoded from a vector by `ActionAdapter`, which clips
      Box actions to the space bounds.
    """

    def __init__(
        self,
        env,
        observation_adapter: ObservationAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
    ):
        self.env = env
        if observation_adapter is None and not isinstance(
            env.observation_space, gym.spaces.Box
        ):
            observation_adapter = ObservationAdapter(env.observation_space)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter or ActionAdapter(env.action_space)

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observations this adapter returns."""
        if self.observation_adapter is None:
            return tuple(self.observation_space.shape)
        return (self.observation_adapter.model_dim,)

    @property
    def is_continuous(self) -> bool:
        return isinstance(self.action_space, gym.spaces.Box)

    @property
    def action_dim(self) -> int:
        """Number of discrete actions, or the size of a continuous action."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(self.action_space.n)
        return self.action_adapter.model_dim

    @property
    def action_low(self) -> float | None:
        if not self.is_continuous:
            return None
        return float(np.asarray(self.action_space.low).reshape(-1)[0])

    @property
    def action_high(self) -> float | None:
        if not self.is_continuous:
            return None
        return float(np.asarray(self.action_space.high).reshape(-1)[0])

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if not isinstance(info, Mapping):
            raise TypeError("environment reset info must be a mapping")
        return self._encode_observation(observation), info

    def step(self, model_action) -> EnvTransition:
        native_action = self._decode_action(model_action)
        observation, reward, terminated, truncated, info = self.env.step(
            native_action
        )
        if isinstance(reward, (bool, np.bool_)) or not np.isscalar(reward):
            raise ValueError("environment reward must be a finite scalar")
        try:
            reward = float(reward)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "environment reward must be a finite scalar"
            ) from error
        if not math.isfinite(reward):
            raise ValueError("environment reward must be a finite scalar")
        if type(terminated) not in (bool, np.bool_):
            raise TypeError("terminated must be boolean")
        if type(truncated) not in (bool, np.bool_):
            raise TypeError("truncated must be boolean")
        if not isinstance(info, Mapping):
            raise TypeError("environment step info must be a mapping")
        return EnvTransition(
            observation=self._encode_observation(observation),
            reward=reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self.env.close()

    def _encode_observation(self, observation) -> np.ndarray:
        if self.observation_adapter is not None:
            return self.observation_adapter.encode(observation)
        array = np.asarray(observation)
        if array.shape != self.observation_space.shape:
            raise ValueError(
                f"Box observation shape must be {self.observation_space.shape}, "
                f"got {array.shape}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError("Box observation must contain only finite values")
        return array

    def _decode_action(self, model_action):
        if isinstance(self.action_space, gym.spaces.Discrete) and np.ndim(model_action) == 0:
            if isinstance(model_action, (bool, np.bool_)) or not np.issubdtype(
                np.asarray(model_action).dtype, np.integer
            ):
                raise ValueError("Discrete action index must be an integer")
            index = int(model_action)
            if not 0 <= index < int(self.action_space.n):
                raise ValueError(
                    f"Discrete action index must be in [0, {int(self.action_space.n)}), "
                    f"got {index}"
                )
            return index + int(self.action_space.start)
        return self.action_adapter.decode(np.asarray(model_action).reshape(-1))


def make_gymnasium_environment(environment_id: str, **make_kwargs) -> GymnasiumEnvAdapter:
    """Create a registered Gymnasium environment wrapped for lattice agents."""
    return GymnasiumEnvAdapter(gym.make(environment_id, **make_kwargs))
