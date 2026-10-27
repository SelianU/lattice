import os
from typing import Dict, List, Optional
import numpy as np

from .gymnasium_adapter import GymnasiumEnvAdapter
from .history import RLTrainingHistory
from .hooks import HookManager, HookSpec, RLHookContext, RL_HOOK_NAMES
from .interaction import _run_interaction
from .logger import BaseLogger, ConsoleLogger


class RLTrainer:
    """
    Unified reinforcement learning loop for DQN, SAC, PPO, and TD3 agents.

    Handles:
    - Episode management and step counting
    - Per-episode metric aggregation and logging
    - Hidden-state reset for recurrent agents (DRQN)
    - Model checkpointing
    - Evaluation rollouts

    The environment must follow the GymnasiumEnvAdapter interface:
        reset() -> (state, info)
        step(action) -> EnvTransition

    The agent must implement:
        observe(state, action, reward, next_state, done)
        select_action(state, evaluation) -> action
        update() -> dict of metrics (may be empty)
        save(filepath)
    Recurrent agents optionally implement reset_hidden_state().
    Set config.eval_interval > 0 to run periodic deterministic evaluation
    during training (logged with an "eval_reward" key).
    """

    def __init__(
        self,
        agent,
        env: GymnasiumEnvAdapter,
        config,
        logger: Optional[BaseLogger] = None,
        save_path: Optional[str] = None,
        hooks: HookSpec = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or ConsoleLogger(
            log_interval=getattr(config, "log_interval", 10)
        )
        self.save_path = save_path
        self.hooks = HookManager(hooks, allowed_names=RL_HOOK_NAMES)

    def add_hook(self, name: str, hook) -> None:
        self.hooks.add(name, hook)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self) -> RLTrainingHistory:
        """
        Run the full training loop.
        Returns a dict of episode-level metric histories, e.g.:
        {"reward": [...], "loss": [...], "epsilon": [...]}
        """
        history = RLTrainingHistory()
        episode_reward = 0.0
        step_buf: Dict[str, List[float]] = {}
        curr_episode = 0

        hook_context = RLHookContext(
            agent=self.agent,
            environment=self.env,
            config=self.config,
        )
        state = self._start_episode(hook_context)

        for global_step in range(1, self.config.total_timesteps + 1):
            hook_context.global_step = global_step
            hook_context.episode = curr_episode + 1

            step = _run_interaction(
                agent=self.agent,
                state=state,
                transition_function=self.env.step,
                hooks=self.hooks,
                context=hook_context,
                evaluation=False,
                update=True,
            )
            transition = step.transition
            episode_reward += transition.reward
            state = transition.next_state

            for k, v in step.metrics.items():
                step_buf.setdefault(k, []).append(v)

            if transition.done:
                curr_episode += 1
                log_metrics = {"reward": episode_reward}
                for k, vs in step_buf.items():
                    log_metrics[k] = float(np.mean(vs))

                self.logger.log_episode(curr_episode, global_step, log_metrics)
                history.append(log_metrics)

                hook_context.metrics = log_metrics
                self.hooks.run("on_episode_end", metrics=log_metrics, context=hook_context)

                eval_interval = getattr(self.config, "eval_interval", 0)
                if eval_interval > 0 and curr_episode % eval_interval == 0:
                    eval_result = self.evaluate(num_episodes=5)
                    eval_metrics = {"eval_reward": eval_result["mean_reward"]}
                    self.logger.log_episode(curr_episode, global_step, eval_metrics)
                    history.append(eval_metrics)

                hook_context.episode = curr_episode + 1
                state = self._start_episode(hook_context)
                episode_reward = 0.0
                step_buf = {}

        self.env.close()

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.agent.save(self.save_path)
            print(f"Model saved to {self.save_path}")

        self.logger.close()
        return history

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Run deterministic evaluation episodes.
        Returns mean and std of episode rewards.
        """
        rewards = []
        hook_context = RLHookContext(
            agent=self.agent,
            environment=self.env,
            config=self.config,
            evaluation=True,
        )
        for episode_index in range(1, num_episodes + 1):
            hook_context.episode = episode_index
            state = self._start_episode(hook_context)
            ep_reward = 0.0
            done = False
            while not done:
                step = _run_interaction(
                    agent=self.agent,
                    state=state,
                    transition_function=self.env.step,
                    hooks=self.hooks,
                    context=hook_context,
                    evaluation=True,
                    update=False,
                )
                state = step.transition.next_state
                ep_reward += step.transition.reward
                done = step.transition.done
            hook_context.metrics = {"eval_reward": ep_reward}
            self.hooks.run("on_episode_end", metrics=hook_context.metrics, context=hook_context)
            rewards.append(ep_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _start_episode(self, hook_context: RLHookContext):
        state, _ = self.env.reset()
        self._maybe_reset_hidden()
        self.hooks.run("on_episode_start", state=state, context=hook_context)
        return state

    def _maybe_reset_hidden(self):
        if hasattr(self.agent, "reset_hidden_state"):
            self.agent.reset_hidden_state()
