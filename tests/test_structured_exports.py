def test_structured_task_primitives_are_public_lattice_exports():
    from lattice import (
        ActionAdapter,
        EncodedTrajectory,
        EnvTransition,
        GymnasiumEnvAdapter,
        MinariTrajectorySource,
        ObservationAdapter,
        OfflineTrajectorySource,
        SpaceSpec,
        StructuredEncoder,
        TrajectoryBatch,
        collate_trajectories,
        infer_space_spec,
    )

    assert all(
        value is not None
        for value in (
            ActionAdapter,
            EncodedTrajectory,
            EnvTransition,
            GymnasiumEnvAdapter,
            MinariTrajectorySource,
            ObservationAdapter,
            OfflineTrajectorySource,
            SpaceSpec,
            StructuredEncoder,
            TrajectoryBatch,
            collate_trajectories,
            infer_space_spec,
        )
    )
