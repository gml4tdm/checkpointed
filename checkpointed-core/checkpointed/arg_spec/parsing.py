from __future__ import annotations

import abc
import logging
import typing

from . import arguments
from . import constraints
from . import core


class ArgumentConsumer(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {}

    @classmethod
    @abc.abstractmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []

    @classmethod
    def parse_arguments(cls,
                        params: dict[str, typing.Any],
                        logger: logging.Logger | None = None) -> core.Config:
        inner = arguments.NestedArgumentGroup(
            name=cls.__name__,
            description=f'Argument parser for the {cls.__class__} class.',
            nested=cls.get_arguments(),
            constraint_items=cls.get_constraints(),
        )
        if logger is None:
            return inner.validate(params)
        return inner.validate_with_logging(params, logger)
