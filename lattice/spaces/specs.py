from __future__ import annotations

import dataclasses
import math
from typing import Literal

import gymnasium as gym
import numpy as np


SpaceKind = Literal["box", "discrete", "multidiscrete", "dict", "tuple"]


@dataclasses.dataclass(frozen=True)
class SpaceSpec:
    kind: SpaceKind
    shape: tuple[int, ...] = ()
    dtype: str = ""
    children: tuple[tuple[str, "SpaceSpec"], ...] = ()
    cardinalities: tuple[int, ...] = ()

    @property
    def flat_dim(self) -> int:
        if self.kind == "box":
            return math.prod(self.shape)
        if self.kind in {"discrete", "multidiscrete"}:
            return sum(self.cardinalities)
        return sum(child.flat_dim for _, child in self.children)


def infer_space_spec(space: gym.Space) -> SpaceSpec:
    if isinstance(space, gym.spaces.Box):
        return SpaceSpec(
            kind="box",
            shape=tuple(int(size) for size in space.shape),
            dtype=np.dtype(space.dtype).name,
        )
    if isinstance(space, gym.spaces.Discrete):
        return SpaceSpec(
            kind="discrete",
            dtype=np.dtype(space.dtype).name,
            cardinalities=(int(space.n),),
        )
    if isinstance(space, gym.spaces.MultiDiscrete):
        return SpaceSpec(
            kind="multidiscrete",
            shape=tuple(int(size) for size in space.shape),
            dtype=np.dtype(space.dtype).name,
            cardinalities=tuple(int(value) for value in space.nvec.reshape(-1)),
        )
    if isinstance(space, gym.spaces.Dict):
        return SpaceSpec(
            kind="dict",
            children=tuple(
                (name, infer_space_spec(child))
                for name, child in space.spaces.items()
            ),
        )
    if isinstance(space, gym.spaces.Tuple):
        return SpaceSpec(
            kind="tuple",
            children=tuple(
                (str(index), infer_space_spec(child))
                for index, child in enumerate(space.spaces)
            ),
        )
    raise TypeError(f"unsupported Gymnasium space: {type(space).__name__}")
