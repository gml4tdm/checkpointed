import abc
import collections
import itertools
import logging
import os
import typing

from .store import ResultStore
from .pipeline import Pipeline
from .step import PipelineStep
from .handle import PipelineStepHandle
from .arg_spec import arguments, constraints


class ScatterGather(PipelineStep, abc.ABC):

    async def execute(self, **inputs) -> typing.Any:
        groups = self.scatter(**inputs)
        pipeline, config, inputs_per_step, output_steps_by_step, output_steps_by_group = self._build_pipeline(groups)
        logger = logging.getLogger(self.logger.name + '.' + self.__class__.__name__)
        plan = pipeline.build(config, logger=logger)
        store = self.execution_context.get_casted(
            'system.executor.storage-manager',
            ResultStore
        )
        sub_store = store.sub_storage(
            parent_handle=self.execution_context.get_casted(
                'system.step.handle', PipelineStepHandle
            ),
            graph=plan.graph,
            factories_by_step=plan.factories_by_step,
        )
        pipeline_results = await plan.execute_async(
            result_store=sub_store,
            precomputed_inputs_by_step=inputs_per_step,
            _return_values=set(
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
        new_config = {}
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
                new_config[handle] = config[node.handle]
                if node.handle == start_node_handle:
                    assert issubclass(node.factory, _ScatterGatherInput)
                    fmts = self.input_storage_formats
                    if len(fmts) != 1:
                        if getattr(node.factory, '$_data_format') is None:
                            raise ValueError(
                                '`scatter_gather_input`s in a scatter/gather with more '
                                'than 1 input must have explicit data format.'
                            )
                    else:
                        node.factory.inherited_data_format = next(iter(self.input_storage_formats.values()))
                    inputs_per_step[handle] = {'scatter-gather-input': value}
            for connection in template_pipeline.connections:
                pipeline.connect(node_mapping[connection.source],
                                 node_mapping[connection.target],
                                 connection.label)
        return pipeline, new_config, inputs_per_step, output_steps_by_step, output_steps_by_group

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


def scatter_gather_input(*, data_format=None, marker_classes):
    """Generate a step which takes the input from the
    scatter step in a scatter/gather pipeline.

    If `data_format` is `None`, the data format will be inherited
    from the ScatterGather parent class itself.
    """
    kwargs = {
        '$_is_scatter_gather_input': True,
        '$_data_format': data_format
    }
    if data_format is None:
        return type(
            'InheritedScatterGatherInput',
            (_ScatterGatherInput,) + tuple(marker_classes),
            kwargs
        )
    return type(
        f'ScatterGatherInput__{data_format.replace("-", "_")}',
        (_ScatterGatherInput,) + tuple(marker_classes),
        kwargs
    )


class _ScatterGatherInput(PipelineStep):

    inherited_data_format = None

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return []

    async def execute(self, **inputs) -> typing.Any:
        return inputs['scatter-gather-input']   # Hacky input

    @classmethod
    def get_data_format(cls) -> str:
        fmt = getattr(cls, '$_data_format')
        if fmt is None:
            return cls.inherited_data_format
        return fmt

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
