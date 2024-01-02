import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments


class DocumentFrequency(checkpointed_core.PipelineStep):


    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        pass

    async def execute(self, **inputs) -> typing.Any:
        pass

    @staticmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    def load_result(path: str):
        pass

    @staticmethod
    def is_deterministic() -> bool:
        pass

    def get_checkpoint_metadata(self) -> typing.Any:
        pass

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        pass

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        pass

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        pass