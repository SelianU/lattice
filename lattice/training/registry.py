import copy
from typing import Any, Dict, Type
import torch
import torch.nn as nn

from lattice.networks.encoders import MLPBackbone, CNNBackbone, RNNBackbone, TransformerBackbone
from lattice.networks import (
    QHead, DuelingQHead,
    GaussianPolicyHead, DiagonalGaussianHead, DeterministicPolicyHead,
    CategoricalPolicyHead,
    DoubleQCriticHead, ValueHead,
)
from lattice.algorithms.config import DQNConfig, SACConfig, PPOConfig, TD3Config
from lattice.algorithms.buffers import ReplayBuffer, SequenceReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
from lattice.algorithms.models import QNetwork, SACActor, SACCritic
from lattice.algorithms.agents import DQNAgent, SACAgent, PPOAgent, TD3Agent
from lattice.training.gymnasium_adapter import GymnasiumEnvAdapter, make_gymnasium_environment
from lattice.training.logger import ConsoleLogger, MatplotlibLogger, CompositeLogger


class Registry:
    """
    Central string-keyed registry for backbones, heads, and agents.

    Built-in keys
    -------------
    Backbones : "mlp", "cnn", "rnn", "transformer"
    Heads     : "q_head", "dueling_q_head", "gaussian_policy", ...
    Agents    : "dqn", "sac", "ppo", "td3"

    Extend at runtime
    -----------------
    Registry.register_backbone("my_net", MyNet)
    Registry.register_agent("my_algo", MyAgent)

    Dict-driven construction
    ------------------------
    q_network = Registry.build_q_network(
        backbone_config={"type": "mlp", "input_dim": 4, "hidden_dims": [128, 128]},
        head_config={"action_dim": 2},
    )
    """

    _backbones: Dict[str, Type[nn.Module]] = {
        "mlp": MLPBackbone,
        "cnn": CNNBackbone,
        "rnn": RNNBackbone,
        "transformer": TransformerBackbone,
    }

    _heads: Dict[str, Type[nn.Module]] = {
        "q_head": QHead,
        "dueling_q_head": DuelingQHead,
        "gaussian_policy": GaussianPolicyHead,
        "diagonal_gaussian": DiagonalGaussianHead,
        "deterministic_policy": DeterministicPolicyHead,
        "categorical_policy": CategoricalPolicyHead,
        "double_q_critic": DoubleQCriticHead,
        "value": ValueHead,
    }

    _agents: Dict[str, type] = {
        "dqn": DQNAgent,
        "sac": SACAgent,
        "ppo": PPOAgent,
        "td3": TD3Agent,
    }

    _agent_builders: Dict[str, type] = {}

    _configs: Dict[str, type] = {
        "dqn": DQNConfig,
        "sac": SACConfig,
        "ppo": PPOConfig,
        "td3": TD3Config,
    }

    _buffers: Dict[str, type] = {
        "replay": ReplayBuffer,
        "prioritized_replay": PrioritizedReplayBuffer,
        "sequence_replay": SequenceReplayBuffer,
        "rollout": RolloutBuffer,
    }

    _environments: Dict[str, type] = {
        "gym": make_gymnasium_environment,
        "custom": GymnasiumEnvAdapter,
    }

    _loggers: Dict[str, type] = {
        "console": ConsoleLogger,
        "matplotlib": MatplotlibLogger,
        "composite": CompositeLogger,
    }

    _optimizers: Dict[str, type] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def register_backbone(cls, name: str, backbone_cls: Type[nn.Module]) -> None:
        cls._backbones[name] = backbone_cls

    @classmethod
    def register_head(cls, name: str, head_cls: Type[nn.Module]) -> None:
        cls._heads[name] = head_cls

    @classmethod
    def register_agent(cls, name: str, agent_cls: type) -> None:
        cls._agents[name] = agent_cls

    @classmethod
    def register_agent_builder(cls, name: str, builder_cls: type) -> None:
        cls._agent_builders[name] = builder_cls

    @classmethod
    def register_config(cls, name: str, config_cls: type) -> None:
        cls._configs[name] = config_cls

    @classmethod
    def register_buffer(cls, name: str, buffer_cls: type) -> None:
        cls._buffers[name] = buffer_cls

    @classmethod
    def register_environment(cls, name: str, environment_factory: type) -> None:
        cls._environments[name] = environment_factory

    @classmethod
    def register_logger(cls, name: str, logger_cls: type) -> None:
        cls._loggers[name] = logger_cls

    @classmethod
    def register_optimizer(cls, name: str, optimizer_cls: type) -> None:
        cls._optimizers[name] = optimizer_cls

    # ------------------------------------------------------------------ #
    # Instantiation helpers                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_backbone(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._backbones:
            raise KeyError(f"Backbone '{name}' not found. Available: {list(cls._backbones)}")
        return cls._backbones[name](**kwargs)

    @classmethod
    def build_head(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._heads:
            raise KeyError(f"Head '{name}' not found. Available: {list(cls._heads)}")
        return cls._heads[name](**kwargs)

    @classmethod
    def build_agent(cls, name: str, **kwargs):
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(cls._agents)}")
        return cls._agents[name](**kwargs)

    @classmethod
    def build_agent_builder(cls, name: str, **kwargs):
        if name not in cls._agent_builders:
            raise KeyError(
                f"Agent builder '{name}' not found. Available: {list(cls._agent_builders)}"
            )
        return cls._agent_builders[name](**kwargs)

    @classmethod
    def get_agent_builder_class(cls, name: str):
        if name not in cls._agent_builders:
            raise KeyError(
                f"Agent builder '{name}' not found. Available: {list(cls._agent_builders)}"
            )
        return cls._agent_builders[name]

    @classmethod
    def build_config(cls, name: str, **kwargs):
        if name not in cls._configs:
            raise KeyError(f"Config '{name}' not found. Available: {list(cls._configs)}")
        return cls._configs[name](**kwargs)

    @classmethod
    def build_buffer(cls, name: str, **kwargs):
        if name not in cls._buffers:
            raise KeyError(f"Buffer '{name}' not found. Available: {list(cls._buffers)}")
        return cls._buffers[name](**kwargs)

    @classmethod
    def build_environment(cls, name: str, **kwargs):
        if name not in cls._environments:
            raise KeyError(f"Environment '{name}' not found. Available: {list(cls._environments)}")
        return cls._environments[name](**kwargs)

    @classmethod
    def build_logger(cls, name: str, **kwargs):
        if name not in cls._loggers:
            raise KeyError(f"Logger '{name}' not found. Available: {list(cls._loggers)}")
        return cls._loggers[name](**kwargs)

    @classmethod
    def build_optimizer(cls, name: str, params, **kwargs):
        if name not in cls._optimizers:
            raise KeyError(f"Optimizer '{name}' not found. Available: {list(cls._optimizers)}")
        return cls._optimizers[name](params, **kwargs)

    # ------------------------------------------------------------------ #
    # Dict-driven network builders                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_q_network(
        cls,
        backbone_config: Dict[str, Any],
        head_config: Dict[str, Any],
    ) -> QNetwork:
        """
        Build a QNetwork from plain dicts (e.g. loaded from YAML).

        backbone_config: {"type": "mlp", "input_dim": 4, "hidden_dims": [128, 128]}
        head_config:     {"action_dim": 2}
        """
        backbone_kwargs = copy.deepcopy(backbone_config)
        backbone = cls.build_backbone(backbone_kwargs.pop("type"), **backbone_kwargs)
        return QNetwork(backbone, QHead(backbone.output_dim, **head_config))

    @classmethod
    def build_sac_actor(
        cls,
        backbone_config: Dict[str, Any],
        head_config: Dict[str, Any],
    ) -> SACActor:
        backbone_kwargs = copy.deepcopy(backbone_config)
        backbone = cls.build_backbone(backbone_kwargs.pop("type"), **backbone_kwargs)
        return SACActor(backbone, GaussianPolicyHead(backbone.output_dim, **head_config))

    @classmethod
    def build_sac_critic(
        cls,
        backbone_config: Dict[str, Any],
        head_config: Dict[str, Any],
    ) -> SACCritic:
        backbone_kwargs = copy.deepcopy(backbone_config)
        backbone = cls.build_backbone(backbone_kwargs.pop("type"), **backbone_kwargs)
        return SACCritic(backbone, DoubleQCriticHead(backbone.output_dim, **head_config))

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def list_backbones(cls) -> list:
        return list(cls._backbones)

    @classmethod
    def list_heads(cls) -> list:
        return list(cls._heads)

    @classmethod
    def list_agents(cls) -> list:
        return list(cls._agents)

    @classmethod
    def list_agent_builders(cls) -> list:
        return list(cls._agent_builders)

    @classmethod
    def list_configs(cls) -> list:
        return list(cls._configs)

    @classmethod
    def list_buffers(cls) -> list:
        return list(cls._buffers)

    @classmethod
    def list_environments(cls) -> list:
        return list(cls._environments)

    @classmethod
    def list_loggers(cls) -> list:
        return list(cls._loggers)

    @classmethod
    def list_optimizers(cls) -> list:
        return list(cls._optimizers)
