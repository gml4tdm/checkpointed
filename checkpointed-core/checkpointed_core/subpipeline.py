import abc
import collections
import logging
import typing

from . import executor
from . import step
from . import pipeline
from . import handle
from . import data_store


class InnerPipelineSpec(typing.NamedTuple):
    sub_pipeline: pipeline.Pipeline
    start_step_handle: handle.PipelineStepHandle
    config_by_handle: dict[handle.PipelineStepHandle, dict[str, typing.Any]]


class SubPipeline(step.PipelineStep, abc.ABC):

    def __init__(self, config: dict[str, typing.Any], logger: logging.Logger):
        super().__init__(config, logger)
        self._inner_pipeline_checkpoints_all_valid = True

    async def execute(self, *, streamed_inputs: list[str] | None = None, **inputs) -> typing.Any:
        groups = self.scatter(streamed_inputs, **inputs)
        inner, config_by_step, outputs_by_group = self._build_inner_pipeline(groups)
        store = self.execution_context.get_casted(
            'system.executor.storage-manager', data_store.ResultStore
        )
        sub_store = store.sub_storage(
            parent_handle=self.execution_context.get_casted(
                'system.step.handle', handle.PipelineStepHandle
            ),
            graph=inner.as_graph(),
            config_by_step=config_by_step,
        )
        task_executor = self.execution_context.get_casted(
            'system.executor.current-executor', executor.TaskExecutor
        )
        logger = self.logger.getChild(self.__class__.__name__)
        plan = inner.build(config_by_step)
        result = await plan.execute_async(
            _precomputed_inputs=...,
            _return_values=...,
            _sub_store=sub_store,
            logger=logger,
            loop=task_executor.loop
        )


    def _build_inner_pipeline(self, groups: dict[str, typing.Any]):
        inner = pipeline.Pipeline(name=f"{self.__class__.__name__}-inner")
        output_steps_by_group = collections.defaultdict(set)
        config_by_handle = {}
        for key, value in groups.items():
            spec = self.get_inner_pipeline(key)
            handle_mapping = {}
            # Step 1: Copy nodes
            for node in spec.sub_pipeline.as_graph().vertices:
                if node.is_output and node.name is None:
                    raise ValueError(f"Output node {node} in sub-pipeline {spec.sub_pipeline} has no name")
                if node.is_input and node.is_output:
                    new_handle = inner.add_source_sink(factory=node.factory,
                                                       filename=node.output_filename + '__' + key,
                                                       name=node.name + '-' + key)
                elif node.is_input:
                    new_handle = inner.add_source(factory=node.factory,
                                                  name=node.name + '-' + key if node.name is not None else None)
                elif node.is_output:
                    new_handle = inner.add_sink(factory=node.factory,
                                                filename=node.output_filename + '__' + key,
                                                name=node.name + '-' + key)
                else:
                    new_handle = inner.add_step(factory=node.factory,
                                                name=node.name + '-' + key if node.name is not None else None)
                handle_mapping[node.handle] = new_handle
                if node.is_input and node.handle == spec.start_step_handle:
                    raise NotImplementedError('Load data from group')
                if node.is_output:
                    output_steps_by_group[key].add(new_handle)
            # Step 2: Copy Edges
            for connection in spec.sub_pipeline.as_graph().edges:
                inner.connect(handle_mapping[connection.source],
                              handle_mapping[connection.target],
                              connection.label)
            # Step 3: Copy config
            config_by_handle |= {handle_mapping[h] for h, v in spec.config_by_handle.items()}
        return inner, config_by_handle, output_steps_by_group


    @classmethod
    def get_output_storage_format(cls) -> str:
        pass

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return False    # Because checkpoint is dynamic

    @classmethod
    def has_dynamic_checkpoint(cls) -> bool:
        return True

    def dynamic_checkpoint_is_valid(self) -> bool:
        return self._inner_pipeline_checkpoints_all_valid

    @abc.abstractmethod
    def scatter(self,
                streamed_inputs: list[str] | None = None,
                **inputs) -> dict[str, typing.Any]:
        pass

    @abc.abstractmethod
    def get_inner_pipeline(self, channel: str) -> InnerPipelineSpec:
        pass

    @abc.abstractmethod
    def gather(self, *, key: str, **results) -> typing.Any:
        pass


