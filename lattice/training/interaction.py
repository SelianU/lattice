from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .gymnasium_adapter import EnvTransition
from .hooks import HookManager, HookSpec, RLHookContext, RLTransition, RL_HOOK_NAMES


TransitionResult = Union[
    EnvTransition,
    Tuple[np.ndarray, float, bool, bool, Dict[str, Any]],
]
TransitionFunction = Callable[[Any], TransitionResult]


@dataclass(frozen=True)
class InteractionStep:
    """Result from one external interaction plus optional agent update."""

    transition: RLTransition
    metrics: Dict[str, float]


def run_interaction_step(
    agent,
    state: np.ndarray,
    transition_function: TransitionFunction,
    evaluation: bool = False,
    update: bool = True,
    hooks: HookSpec = None,
    context: Optional[RLHookContext] = None,
) -> InteractionStep:
    """
    Run one state -> action -> external transition -> observe/update step.

    The transition_function owns the outside world. It receives the action and
    returns an `EnvTransition` or a Gymnasium-style
    `(next_state, reward, terminated, truncated, info)` tuple.
    """
    hook_manager = HookManager(hooks, allowed_names=RL_HOOK_NAMES)
    hook_context = context or RLHookContext(
        agent=agent,
        environment=None,
        config=getattr(agent, "config", None),
        evaluation=evaluation,
    )
    return _run_interaction(
        agent=agent,
        state=state,
        transition_function=transition_function,
        hooks=hook_manager,
        context=hook_context,
        evaluation=evaluation,
        update=update,
    )


def _run_interaction(
    agent,
    state: np.ndarray,
    transition_function: TransitionFunction,
    hooks: HookManager,
    context: RLHookContext,
    evaluation: bool,
    update: bool,
) -> InteractionStep:
    processed_state = hooks.transform("process_state", state, context=context)
    custom_action = hooks.first(
        "select_action",
        agent=agent,
        state=processed_state,
        context=context,
    )
    action = custom_action if custom_action is not None else agent.select_action(
        processed_state,
        evaluation=evaluation,
    )
    action = hooks.transform("process_action", action, context=context)
    environment_transition = _to_env_transition(transition_function(action))
    next_state = hooks.transform(
        "process_next_state",
        environment_transition.observation,
        state=processed_state,
        action=action,
        context=context,
    )
    transition = RLTransition(
        state=processed_state,
        action=action,
        reward=environment_transition.reward,
        next_state=next_state,
        terminated=environment_transition.terminated,
        truncated=environment_transition.truncated,
        info=dict(environment_transition.info),
    )
    transition.reward = hooks.transform(
        "process_reward",
        transition.reward,
        transition=transition,
        context=context,
    )

    metrics = {}
    if not evaluation:
        hooks.run("on_transition", transition=transition, context=context)
        agent.observe(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        )
        if update:
            hooks.run("before_update", agent=agent, context=context)
            metrics = agent.update()
            context.metrics = metrics
            hooks.run("after_update", metrics=metrics, context=context)

    return InteractionStep(transition=transition, metrics=metrics)


def _to_env_transition(result: TransitionResult) -> EnvTransition:
    if isinstance(result, EnvTransition):
        return result
    if isinstance(result, tuple) and len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise TypeError("transition info must be a mapping")
        return EnvTransition(
            observation=next_state,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
    raise ValueError(
        "transition_function must return an EnvTransition or "
        "(next_state, reward, terminated, truncated, info)."
    )
