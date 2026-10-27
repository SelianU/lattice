"""
Lattice: PyTorch tools for models, training loops, and RL agents.
"""

__version__ = "0.1.0"

from .networks import (
    MaskedSequenceTransformer,
    build_model,
    make_cnn_mlp,
    make_mlp,
    make_mlp_classifier,
    make_rnn,
    make_transformer,
)
from .training import (
    EnvTransition,
    GymnasiumEnvAdapter,
    SupervisedTrainingConfig,
    evaluate_supervised_model,
    quick_dqn,
    quick_ppo,
    quick_sac,
    quick_td3,
    train_supervised_model,
)
from .data import (
    EncodedTrajectory,
    MinariTrajectorySource,
    OfflineTrajectorySource,
    TrajectoryBatch,
    collate_trajectories,
)
from .networks.encoders import StructuredEncoder
from .spaces import (
    ActionAdapter,
    ObservationAdapter,
    SpaceSpec,
    infer_space_spec,
)
from .algorithms import (
    make_dqn_agent,
    make_ppo_agent,
    make_sac_agent,
    make_td3_agent,
)

__all__ = [
    "__version__",
    "MaskedSequenceTransformer",
    "ActionAdapter",
    "EncodedTrajectory",
    "EnvTransition",
    "GymnasiumEnvAdapter",
    "MinariTrajectorySource",
    "ObservationAdapter",
    "OfflineTrajectorySource",
    "SpaceSpec",
    "StructuredEncoder",
    "TrajectoryBatch",
    "collate_trajectories",
    "infer_space_spec",
    "build_model",
    "make_cnn_mlp",
    "make_mlp",
    "make_mlp_classifier",
    "make_rnn",
    "make_transformer",
    "SupervisedTrainingConfig",
    "evaluate_supervised_model",
    "train_supervised_model",
    "quick_dqn",
    "quick_ppo",
    "quick_sac",
    "quick_td3",
    "make_dqn_agent",
    "make_ppo_agent",
    "make_sac_agent",
    "make_td3_agent",
]
