from __future__ import annotations

import abc
import typing

from . import arg_spec


class PipelineStep(arg_spec.ArgumentConsumer, abc.ABC):

    def __init__(self, config):
        self.config = self.parse_arguments(config)

    @classmethod
    @abc.abstractmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        raise ValueError(f"Step {step} does not have an input labelled {label!r}")

    @staticmethod
    @abc.abstractmethod
    def get_input_labels() -> list[str | ...]:
        """Get input labels. Ellipsis (...) denotes arbitrary keyword arguments.
        """

    @abc.abstractmethod
    async def execute(self, **inputs) -> typing.Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_result(path: str):
        pass

    @staticmethod
    @abc.abstractmethod
    def is_deterministic() -> bool:
        pass

    @abc.abstractmethod
    def get_checkpoint_metadata(self) -> typing.Any:
        pass

    @abc.abstractmethod
    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        pass
