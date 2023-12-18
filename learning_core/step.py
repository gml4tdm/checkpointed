from __future__ import annotations

import abc
import typing


class PipelineStep(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        pass

    @abc.abstractmethod
    def execute(self, input_data):
        pass

    @staticmethod
    @abc.abstractmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_result(path: str, result: typing.Any):
        pass


class NoopStep(PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        return True

    def execute(self, input_data):
        pass

    @staticmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    def load_result(path: str, result: typing.Any):
        pass
