import dataclasses

import numpy as np
import pytest

from lattice.data import EncodedTrajectory, collate_trajectories


def _trajectory(length=2, observation_dim=3, action_dim=1):
    return EncodedTrajectory(
        observations=np.ones((length, observation_dim), dtype=np.float32),
        actions=np.ones((length, action_dim), dtype=np.float32),
        rewards=np.arange(1, length + 1, dtype=np.float32),
        terminated=np.array(
            [False] * max(length - 1, 0) + ([True] if length else [])
        ),
        truncated=np.zeros(length, dtype=bool),
    )


def test_left_padding_marks_only_real_timesteps_valid():
    first = _trajectory(length=2)
    second = dataclasses.replace(
        first,
        observations=np.full((1, 3), 4, dtype=np.float32),
        actions=np.full((1, 1), 5, dtype=np.float32),
        rewards=np.array([6], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
    )

    batch = collate_trajectories([first, second], padding_side="left")

    assert batch.observations.shape == (2, 2, 3)
    assert batch.actions.shape == (2, 2, 1)
    assert batch.valid_mask.tolist() == [[True, True], [False, True]]
    np.testing.assert_array_equal(batch.observations[1, 0], np.zeros(3))
    np.testing.assert_array_equal(batch.observations[1, 1], np.full(3, 4))


def test_right_padding_places_short_trajectory_at_the_start():
    batch = collate_trajectories(
        [_trajectory(length=1), _trajectory(length=3)], padding_side="right"
    )

    assert batch.valid_mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]
    assert batch.terminated[0].tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observations", np.ones((1, 3), np.float32), "time lengths"),
        ("actions", np.ones((1, 1), np.float32), "time lengths"),
        ("rewards", np.ones(1, np.float32), "time lengths"),
        ("terminated", np.ones(1, bool), "time lengths"),
        ("truncated", np.ones(1, bool), "time lengths"),
    ],
)
def test_trajectory_rejects_mismatched_time_lengths(field, value, message):
    values = {
        "observations": np.ones((2, 3), np.float32),
        "actions": np.ones((2, 1), np.float32),
        "rewards": np.ones(2, np.float32),
        "terminated": np.zeros(2, bool),
        "truncated": np.zeros(2, bool),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        EncodedTrajectory(**values)


def test_trajectory_rejects_empty_non_finite_or_wrong_rank_values():
    with pytest.raises(ValueError, match="must not be empty"):
        _trajectory(length=0)

    values = dataclasses.asdict(_trajectory())
    values["rewards"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        EncodedTrajectory(**values)

    values = dataclasses.asdict(_trajectory())
    values["observations"] = np.ones(2, np.float32)
    with pytest.raises(ValueError, match="observations must be two-dimensional"):
        EncodedTrajectory(**values)


def test_collation_rejects_feature_mismatch_empty_input_and_padding_name():
    with pytest.raises(ValueError, match="at least one"):
        collate_trajectories([])
    with pytest.raises(ValueError, match="observation feature dimensions"):
        collate_trajectories([_trajectory(), _trajectory(observation_dim=4)])
    with pytest.raises(ValueError, match="action feature dimensions"):
        collate_trajectories([_trajectory(), _trajectory(action_dim=2)])
    with pytest.raises(ValueError, match="padding_side"):
        collate_trajectories([_trajectory()], padding_side="middle")


def test_records_own_read_only_float32_and_boolean_arrays():
    trajectory = EncodedTrajectory(
        observations=np.ones((1, 2), dtype=np.float64),
        actions=np.ones((1, 1), dtype=np.float64),
        rewards=np.ones(1, dtype=np.float64),
        terminated=np.ones(1, dtype=np.int8),
        truncated=np.zeros(1, dtype=np.int8),
    )
    batch = collate_trajectories([trajectory])

    assert trajectory.observations.dtype == np.float32
    assert trajectory.terminated.dtype == np.bool_
    assert trajectory.observations.flags.writeable is False
    assert batch.valid_mask.flags.writeable is False
