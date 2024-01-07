import abc
import os
import typing

from .arg_spec import constraints, arguments
from .pipeline import Pipeline
from .step import PipelineStep
from .handle import PipelineStepHandle


class ScatterGather(abc.ABC, PipelineStep):

    def __init__(self, config):
        super().__init__(config)

    async def execute(self, **inputs) -> typing.Any:
        groups = self.scatter(**inputs)
        # TODO: build pipeline
        template_pipeline, config = self.get_inner_pipeline()
        pipeline = Pipeline(template_pipeline.pipeline_name)
        inputs_per_step = {}
        output_steps_to_groups = {}
        plan = pipeline.build(config, logger=self.logger)
        results_per_group = await plan.execute_async(
            output_directory=os.path.join(
                self._output_directory, 'nested'
            ),
            checkpoint_directory=os.path.join(
                self._checkpoint_directory, 'nested'
            ),
            precomputed_inputs_by_step=inputs_per_step,
            __return_values=set(output_steps_to_groups.keys())
        )
        return self.gather(
            **{output_steps_to_groups[k]: v
               for k, v in results_per_group.items()}
        )

    @abc.abstractmethod
    def scatter(self, **inputs) -> dict[str, typing.Any]:
        pass

    @abc.abstractmethod
    def get_inner_pipeline(self) -> tuple[Pipeline, dict[PipelineStepHandle, dict[str, typing.Any]]]:
        pass

    @abc.abstractmethod
    def gather(self, **results) -> typing.Any:
        pass

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        # A scatter/gather operation cannot be validated,
        # because its validity depends on the validity of
        # all steps in the inner pipeline.
        # However, generally,  a scatter/gather operation
        # is computationally cheap;
        # The overhead incurred should be minimal,
        # especially since steps in the inner pipeline
        # can still be cached.
        return False


class ScatterGatherInput(PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        return True

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return [...]

    async def execute(self, **inputs) -> typing.Any:
        return inputs

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
