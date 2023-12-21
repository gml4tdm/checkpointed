from __future__ import annotations

import abc
import typing


class PipelineStep(abc.ABC):

    def __init__(self, config):
        self.config = config

    @classmethod
    @abc.abstractmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        pass

    @abc.abstractmethod
    async def execute(self, *inputs) -> typing.Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_result(path: str):
        pass


class NoopStep(PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        return True

    async def execute(self, *_):
        pass

    @staticmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    def load_result(path: str):
        pass
