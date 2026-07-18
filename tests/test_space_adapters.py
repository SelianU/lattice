import dataclasses

import gymnasium as gym
import numpy as np
import pytest

from lattice.spaces import infer_space_spec


def test_dict_space_uses_declared_child_order_and_flat_dim():
    space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        }
    )

    spec = infer_space_spec(space)

    assert spec.kind == "dict"
    assert tuple(name for name, _ in spec.children) == tuple(space.spaces)
    assert spec.flat_dim == 5


def test_discrete_and_multidiscrete_flat_dims_are_one_hot_sizes():
    discrete = infer_space_spec(gym.spaces.Discrete(4))
    multidiscrete = infer_space_spec(gym.spaces.MultiDiscrete([2, 3]))

    assert discrete.cardinalities == (4,)
    assert discrete.flat_dim == 4
    assert multidiscrete.cardinalities == (2, 3)
    assert multidiscrete.flat_dim == 5


def test_box_and_tuple_specs_capture_shape_dtype_and_index_names():
    space = gym.spaces.Tuple(
        (
            gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float64),
            gym.spaces.Discrete(3),
        )
    )

    spec = infer_space_spec(space)

    assert spec.kind == "tuple"
    assert tuple(name for name, _ in spec.children) == ("0", "1")
    assert spec.children[0][1].shape == (2, 2)
    assert spec.children[0][1].dtype == "float64"
    assert spec.flat_dim == 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "box"


def test_unsupported_space_reports_the_class_name():
    with pytest.raises(TypeError, match="unsupported Gymnasium space: Text"):
        infer_space_spec(gym.spaces.Text(max_length=8))
