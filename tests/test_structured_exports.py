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


def test_supervised_training_pair_is_public_lattice_export():
    from lattice import evaluate_supervised_model, train_supervised_model

    assert callable(evaluate_supervised_model)
    assert callable(train_supervised_model)


def test_networks_all_lists_only_importable_names():
    import lattice.networks as networks

    assert all(hasattr(networks, name) for name in networks.__all__)
    assert {"make_mlp", "MLP", "SequenceTransformer"} <= set(networks.__all__)
