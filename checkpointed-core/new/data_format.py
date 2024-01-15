from __future__ import annotations

import abc
import typing

_registry: dict[str, type[DataFormat]] | None = None


def initialise_format_registry():
    global _registry
    _registry = {}


def is_initialised() -> bool:
    return _registry is not None


def register_format(name: str, data_format: type[DataFormat]):
    if _registry is None:
        raise RuntimeError("Format registry is not initialized")
    if name in _registry:
        raise ValueError(f"Format {name!r} is already registered")
    _registry[name] = data_format


def get_format(name: str) -> type[DataFormat]:
    if _registry is None:
        raise RuntimeError("Format registry is not initialized")
    if name not in _registry:
        raise ValueError(f"Format {name!r} is not registered")
    return _registry[name]


def is_registered(name: str) -> bool:
    if _registry is None:
        raise RuntimeError("Format registry is not initialized")
    return name in _registry


class DataFormat(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def store(path: str, data: typing.Any):
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> typing.Any:
        pass
