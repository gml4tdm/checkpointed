import abc
import collections
import itertools
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
        pipeline, config, inputs_per_step, output_steps_by_step, output_steps_by_group = self._build_pipeline(groups)
        plan = pipeline.build(config, logger=self.logger)
        pipeline_results = await plan.execute_async(
            output_directory=os.path.join(
                self._output_directory, 'nested'
            ),
            checkpoint_directory=os.path.join(
                self._checkpoint_directory, 'nested'
            ),
            precomputed_inputs_by_step=inputs_per_step,
            __return_values=set(
                itertools.chain.from_iterable(output_steps_by_step.values())
            )
        )
        return self._build_output(pipeline_results,
                                  output_steps_by_group,
                                  output_steps_by_step)

    def _build_pipeline(self, groups: dict[str, typing.Any]):
        template_pipeline, start_node_handle, config = self.get_inner_pipeline()
        pipeline = Pipeline(template_pipeline.pipeline_name)
        inputs_per_step = {}
        output_steps_by_group = collections.defaultdict(set)
        output_steps_by_step = collections.defaultdict(set)
        for key, value in groups.items():
            node_mapping = {}
            for node in template_pipeline.nodes:
                if node.is_input and node.is_output:
                    if node.name is None:
                        raise ValueError('Output node must have a set name.')
                    handle = pipeline.add_source(node.factory,
                                                 is_sink=True,
                                                 filename=node.output_filename + '__' + key,
                                                 name=node.name + '-' + key)
                    output_steps_by_step[node.name].add(handle)
                    output_steps_by_group[key].add(handle)
                elif node.is_input:
                    handle = pipeline.add_source(node.factory,
                                                 name=(node.name + '-' + key) if node.name is not None else None)
                elif node.is_output:
                    if node.name is None:
                        raise ValueError('Output node must have a set name.')
                    handle = pipeline.add_sink(node.factory,
                                               filename=node.output_filename + '__' + key,
                                               name=node.name + '-' + key)
                    output_steps_by_step[node.name].add(handle)
                    output_steps_by_group[key].add(handle)
                else:
                    handle = pipeline.add_step(node.factory,
                                               name=(node.name + '-' + key) if node.name is not None else None)
                node_mapping[node.handle] = handle
                if node.handle == start_node_handle:
                    inputs_per_step[handle] = value
            for connection in template_pipeline.connections:
                pipeline.connect(node_mapping[connection.source],
                                 node_mapping[connection.target],
                                 connection.label)
        return pipeline, config, inputs_per_step, output_steps_by_step, output_steps_by_group

    def _build_output(self,
                      pipeline_results: dict[PipelineStepHandle, typing.Any],
                      output_steps_by_group: dict[str, set[PipelineStepHandle]],
                      output_steps_by_step: dict[str, set[PipelineStepHandle]]) -> dict[str, typing.Any]:
        group_by_handle = {}
        for group, steps in output_steps_by_group.items():
            for step in steps:
                group_by_handle[step] = group
        step_by_handle = {}
        for name, steps in output_steps_by_step.items():
            for step in steps:
                step_by_handle[step] = name
        outputs = collections.defaultdict(dict)
        for handle, result in pipeline_results.items():
            outputs[step_by_handle[handle]][group_by_handle[handle]] = result
        return {
            step: self.gather(key=step, **group)
            for step, group in outputs.items()
        }

    @abc.abstractmethod
    def scatter(self, **inputs) -> dict[str, typing.Any]:
        pass

    @abc.abstractmethod
    def get_inner_pipeline(self) -> tuple[Pipeline, PipelineStepHandle, dict[PipelineStepHandle, dict[str, typing.Any]]]:
        pass

    @abc.abstractmethod
    def gather(self, *, key: str, **results) -> typing.Any:
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
