from abc import ABC, abstractmethod
import numpy as np
from typing import Union

class BaseAgent(ABC):
    """
    Abstract Base Class for Reinforcement Learning Agents.
    """

    def observe(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """
        Store a transition. Default implementation delegates to self.replay_buffer.
        Override for on-policy agents (PPO, etc.) that don't use a replay buffer.

        `terminated` means no future reward follows `next_state`. `truncated`
        means the episode was cut off (for example by a time limit) and the
        value of `next_state` should still be bootstrapped.
        """
        self.replay_buffer.push(state, action, reward, next_state, terminated, truncated)

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluation: bool = False) -> Union[int, np.ndarray]:
        """
        Select an action given an observation/state.
        Args:
            state: The current state observation.
            evaluation: If True, select action greedily (no exploration).
        """
        pass
        
    @abstractmethod
    def update(self) -> dict:
        """
        Run a single optimization step.
        Returns:
            A dictionary containing losses or training metrics.
        """
        pass
        
    @abstractmethod
    def save(self, filepath: str):
        """
        Save agent checkpoints.
        """
        pass
        
    @abstractmethod
    def load(self, filepath: str):
        """
        Load agent checkpoints.
        """
        pass
