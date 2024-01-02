from __future__ import annotations

import abc
import typing

from . import arg_spec
from .arg_spec import arguments
from .arg_spec.constraints import Constraint


class PipelineStep(arg_spec.ArgumentConsumer, abc.ABC):

    def __init__(self, config):
        self.config = self.parse_arguments(config)

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


class NoopStep(PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        return True

    async def execute(self, *_):
        if self.config['echo-execute']:
            print('Executing')
        if self.config['echo-io']:
            print('Saving (not actually) ;P')

    @staticmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    def load_result(path: str):
        pass

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'echo-execute': arg_spec.arguments.BoolArgument(
                name='echo-execute',
                description='If True, print messages to stdout when .execute is called. Implies echo-io == True',
                default=False
            ),
            'echo-io': arg_spec.arguments.BoolArgument(
                name='echo-io',
                description='If True, print messages to stdout when normally I/O would be performed',
                default=False
            )
        }

    @classmethod
    def get_constraints(cls) -> list[Constraint]:
        return [
            arg_spec.constraints.Forbids(
                main=arg_spec.constraints.Equal(
                    arg_spec.constraints.ArgumentRef('echo-execute'),
                    arg_spec.constraints.Constant(True)
                ),
                forbids=[
                    arg_spec.constraints.Equal(
                        arg_spec.constraints.ArgumentRef('echo-io'),
                        arg_spec.constraints.Constant(False)
                    )
                ],
                message='echo-io must be True if echo-execute is True.'
            )
        ]

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        return None

    def checkpoint_is_valid(self, _) -> bool:
        return True
