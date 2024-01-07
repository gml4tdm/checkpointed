from __future__ import annotations

import abc
import logging
import typing

from . import arg_spec
from .arg_spec import core


class PipelineStep(arg_spec.ArgumentConsumer, abc.ABC):

    def __init__(self, config: dict[str, typing.Any], logger: logging.Logger):
        self.config = self.parse_arguments(config)
        self.logger = logger
        self._input_storage_formats: dict[str, str] | None = None
        self._execution_context: core.Config | None = None

    @property
    def execution_context(self) -> core.Config:
        if self._execution_context is None:
            raise ValueError("Execution context not set")
        return self._execution_context

    @execution_context.setter
    def execution_context(self, context: core.Config):
        if self._execution_context is not None:
            raise ValueError("Execution context already set")
        self._execution_context = context

    @property
    def input_storage_formats(self) -> dict[str, str]:
        if self._input_storage_formats is None:
            raise ValueError("Input storage formats not set")
        return self._input_storage_formats

    @input_storage_formats.setter
    def input_storage_formats(self, formats: dict[str, str]):
        if self._input_storage_formats is not None:
            raise ValueError("Input storage formats already set")
        self._input_storage_formats = formats

    @classmethod
    @abc.abstractmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        raise ValueError(f"Step {cls} does not have an input labelled {label!r}")

    @staticmethod
    @abc.abstractmethod
    def get_input_labels() -> list[str | type(...)]:
        """Get input labels. Ellipsis (...) denotes arbitrary keyword arguments.
        """

    @abc.abstractmethod
    async def execute(self, **inputs) -> typing.Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_data_format() -> str:
        pass

    @abc.abstractmethod
    def get_checkpoint_metadata(self) -> typing.Any:
        pass

    @abc.abstractmethod
    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        pass
