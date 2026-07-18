import dataclasses
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest

from lattice.spaces import ActionAdapter, ObservationAdapter, infer_space_spec


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


def test_dict_observation_encoding_uses_space_order_and_float32():
    space = gym.spaces.Dict(
        OrderedDict(
            (
                (
                    "desired_goal",
                    gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
                ),
                (
                    "observation",
                    gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32),
                ),
            )
        )
    )
    adapter = ObservationAdapter(space)

    encoded = adapter.encode(
        {
            "observation": np.array([1, 2], dtype=np.float32),
            "desired_goal": np.array([3], dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(encoded, np.array([3, 1, 2], np.float32))
    assert encoded.dtype == np.float32
    assert adapter.model_dim == 3


def test_observation_encoding_rejects_wrong_shape_keys_and_non_finite_values():
    box = ObservationAdapter(
        gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="shape"):
        box.encode(np.array([1], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        box.encode(np.array([0, np.nan], dtype=np.float32))

    mapping = ObservationAdapter(
        gym.spaces.Dict(
            {"value": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)}
        )
    )
    with pytest.raises(ValueError, match="keys"):
        mapping.encode({"other": np.array([0], dtype=np.float32)})


def test_box_action_decode_reshapes_clips_and_uses_native_dtype():
    adapter = ActionAdapter(
        gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    )

    decoded = adapter.decode(np.array([2.0, -2.0], dtype=np.float32))

    np.testing.assert_array_equal(decoded, np.array([1.0, -1.0], np.float32))
    assert decoded.shape == (2,)
    assert decoded.dtype == np.float32


def test_discrete_action_round_trip_respects_nonzero_start():
    adapter = ActionAdapter(gym.spaces.Discrete(3, start=5))

    np.testing.assert_array_equal(
        adapter.encode(7), np.array([0, 0, 1], dtype=np.float32)
    )
    assert adapter.decode(np.array([0.1, 0.7, 0.2], dtype=np.float32)) == 6


def test_multidiscrete_and_tuple_actions_round_trip():
    space = gym.spaces.Tuple(
        (
            gym.spaces.MultiDiscrete([2, 3], start=[1, 4]),
            gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        )
    )
    adapter = ActionAdapter(space)

    encoded = adapter.encode((np.array([2, 6]), np.array([0.5], np.float32)))
    decoded = adapter.decode(encoded)

    np.testing.assert_array_equal(decoded[0], np.array([2, 6]))
    np.testing.assert_array_equal(decoded[1], np.array([0.5], np.float32))


def test_action_decode_rejects_wrong_length_and_non_finite_vectors():
    adapter = ActionAdapter(gym.spaces.Discrete(3))
    with pytest.raises(ValueError, match="length"):
        adapter.decode(np.array([1, 0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        adapter.decode(np.array([0, np.inf, 1], dtype=np.float32))
