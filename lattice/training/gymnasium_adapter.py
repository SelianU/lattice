from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math

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
    def __init__(
        self,
        env,
        observation_adapter: ObservationAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
    ):
        self.env = env
        self.observation_adapter = observation_adapter or ObservationAdapter(
            env.observation_space
        )
        self.action_adapter = action_adapter or ActionAdapter(env.action_space)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if not isinstance(info, Mapping):
            raise TypeError("environment reset info must be a mapping")
        return self.observation_adapter.encode(observation), info

    def step(self, model_action) -> EnvTransition:
        native_action = self.action_adapter.decode(model_action)
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
            observation=self.observation_adapter.encode(observation),
            reward=reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self.env.close()
