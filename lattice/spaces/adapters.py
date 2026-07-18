from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers

import gymnasium as gym
import numpy as np

from .specs import SpaceSpec, infer_space_spec


def _require_finite(vector: np.ndarray, *, label: str) -> None:
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain only finite values")


def _encode_box(space: gym.spaces.Box, value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != space.shape:
        raise ValueError(
            f"Box value shape must be {space.shape}, got {array.shape}"
        )
    encoded = np.asarray(array, dtype=np.float32).reshape(-1)
    _require_finite(encoded, label="Box value")
    return encoded


def _encode_discrete(space: gym.spaces.Discrete, value: object) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("Discrete value must be an integer in the space")
    index = int(value) - int(space.start)
    if index < 0 or index >= int(space.n):
        raise ValueError("Discrete value must be an integer in the space")
    encoded = np.zeros(int(space.n), dtype=np.float32)
    encoded[index] = 1.0
    return encoded


def _encode_multidiscrete(
    space: gym.spaces.MultiDiscrete, value: object
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != space.shape:
        raise ValueError(
            f"MultiDiscrete value shape must be {space.shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("MultiDiscrete value must contain integers in the space")
    flat = array.reshape(-1)
    starts = np.asarray(space.start).reshape(-1)
    cardinalities = np.asarray(space.nvec).reshape(-1)
    parts: list[np.ndarray] = []
    for native, start, cardinality in zip(
        flat, starts, cardinalities, strict=True
    ):
        index = int(native) - int(start)
        if index < 0 or index >= int(cardinality):
            raise ValueError("MultiDiscrete value must contain integers in the space")
        one_hot = np.zeros(int(cardinality), dtype=np.float32)
        one_hot[index] = 1.0
        parts.append(one_hot)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _encode(space: gym.Space, value: object) -> np.ndarray:
    if isinstance(space, gym.spaces.Box):
        return _encode_box(space, value)
    if isinstance(space, gym.spaces.Discrete):
        return _encode_discrete(space, value)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return _encode_multidiscrete(space, value)
    if isinstance(space, gym.spaces.Dict):
        if not isinstance(value, Mapping) or set(value) != set(space.spaces):
            raise ValueError("Dict value keys must exactly match the space keys")
        parts = [_encode(child, value[name]) for name, child in space.spaces.items()]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    if isinstance(space, gym.spaces.Tuple):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != len(space.spaces)
        ):
            raise ValueError("Tuple value length must match the space length")
        parts = [_encode(child, item) for child, item in zip(space.spaces, value)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    raise TypeError(f"unsupported Gymnasium space: {type(space).__name__}")


def _decode(
    space: gym.Space, vector: np.ndarray, offset: int
) -> tuple[object, int]:
    spec = infer_space_spec(space)
    end = offset + spec.flat_dim
    segment = vector[offset:end]
    if isinstance(space, gym.spaces.Box):
        native = segment.reshape(space.shape)
        native = np.clip(native, space.low, space.high).astype(space.dtype)
        return native, end
    if isinstance(space, gym.spaces.Discrete):
        return int(np.argmax(segment)) + int(space.start), end
    if isinstance(space, gym.spaces.MultiDiscrete):
        values: list[int] = []
        cursor = offset
        for cardinality, start in zip(
            np.asarray(space.nvec).reshape(-1),
            np.asarray(space.start).reshape(-1),
            strict=True,
        ):
            component_end = cursor + int(cardinality)
            values.append(
                int(np.argmax(vector[cursor:component_end])) + int(start)
            )
            cursor = component_end
        native = np.asarray(values, dtype=space.dtype).reshape(space.shape)
        return native, end
    if isinstance(space, gym.spaces.Dict):
        decoded: dict[str, object] = {}
        cursor = offset
        for name, child in space.spaces.items():
            decoded[name], cursor = _decode(child, vector, cursor)
        return decoded, end
    if isinstance(space, gym.spaces.Tuple):
        decoded_items: list[object] = []
        cursor = offset
        for child in space.spaces:
            item, cursor = _decode(child, vector, cursor)
            decoded_items.append(item)
        return tuple(decoded_items), end
    raise TypeError(f"unsupported Gymnasium space: {type(space).__name__}")


class ObservationAdapter:
    def __init__(self, space: gym.Space):
        self.space = space
        self.spec: SpaceSpec = infer_space_spec(space)

    @property
    def model_dim(self) -> int:
        return self.spec.flat_dim

    def encode(self, value: object) -> np.ndarray:
        return _encode(self.space, value)


class ActionAdapter:
    def __init__(self, space: gym.Space):
        self.space = space
        self.spec: SpaceSpec = infer_space_spec(space)

    @property
    def model_dim(self) -> int:
        return self.spec.flat_dim

    def encode(self, value: object) -> np.ndarray:
        return _encode(self.space, value)

    def decode(self, vector: object) -> object:
        encoded = np.asarray(vector, dtype=np.float32)
        if encoded.ndim != 1 or encoded.size != self.model_dim:
            raise ValueError(
                f"action vector length must be {self.model_dim}, got {encoded.size}"
            )
        _require_finite(encoded, label="action vector")
        native, offset = _decode(self.space, encoded, 0)
        if offset != encoded.size:
            raise RuntimeError("action decoder did not consume the complete vector")
        return native
