import abc
import typing

from .pipeline import Pipeline
from .step import PipelineStep
from .arg_spec import arguments, constraints


class ScatterGather(abc.ABC, PipelineStep):

    def __init__(self, config):
        super().__init__(config)
        self._checkpoint_directory = None
        self._output_directory = None

    def set_storage_directories(self,
                                checkpoint_directory: str,
                                output_directory: str):
        # Special-cases method in the pipeline;
        # required to propagate storage locations
        # to nested pipelines
        self._checkpoint_directory = checkpoint_directory
        self._output_directory = output_directory

    async def execute(self, **inputs) -> typing.Any:
        streams = await self.scatter(**inputs)
        results = {}
        return await self.gather(**results)

    @abc.abstractmethod
    def get_scatter(self) -> PipelineStep:
        pass

    @abc.abstractmethod
    def get_inner_pipeline(self) -> Pipeline:
        pass

    @abc.abstractmethod
    def get_gather(self) -> PipelineStep:
        pass


class _ScatterLoadAdapter(PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        pass

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        pass

    async def execute(self, **inputs) -> typing.Any:
        pass

    @staticmethod
    def save_result(path: str, result: typing.Any):
        pass

    @staticmethod
    def load_result(path: str):
        pass

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
