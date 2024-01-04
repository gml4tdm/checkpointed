import os
import typing

import hdbscan
import numpy

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from .... import bases


class HDBSCAN(checkpointed_core.PipelineStep, bases.LabelAssignment):
    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'data':
            return issubclass(step, bases.DenseNumericalVectorData)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['data']

    async def execute(self, **inputs) -> typing.Any:
        if (min_samples := self.config.get_casted('params.min-samples', int)) == -1:
            min_samples = self.config.get_casted('params.min-cluster-size', int)
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.config.get_casted('params.min-cluster-size', int),
            min_samples=min_samples,
            cluster_selection_epsilon=self.config.get_casted('params.cluster-selection-epsilon', float),
            cluster_selection_method=self.config.get_casted('params.cluster-selection-method', str),
            allow_single_cluster=self.config.get_casted('params.allow-single-cluster', bool)
        )
        clustered = model.fit(inputs['data'])
        return clustered.labels_    # Ignore IDE warnings

    @staticmethod
    def save_result(path: str, result: typing.Any):
        numpy.save(os.path.join(path, 'main.npy'), result)

    @staticmethod
    def load_result(path: str):
        return numpy.load(os.path.join(path, 'main.npy'))

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'min-cluster-size': arguments.IntArgument(
                name='min-cluster-size',
                description='The minimum number of documents in a cluster.',
                minimum=1
            ),
            'min-samples': arguments.IntArgument(
                name='min-samples',
                description='See '
                            'https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples . '
                            'Use -1 to set equal to min_cluster_size',
                default=-1
            ),
            'cluster-selection-epsilon': arguments.FloatArgument(
                name='cluster-selection-epsilon',
                description='Configure clustering distance.',
                minimum=0.0
            ),
            'cluster-selection-method': arguments.EnumArgument(
                name='cluster-selection-method',
                description='Configure clustering method. '
                            '"eom" generally prefers large clusters, while "leaf" tends to prefer smaller clusters',
                options=['eom', 'leaf']
            ),
            'allow-single-cluster': arguments.BoolArgument(
                name='allow-single-cluster',
                description='Allow single cluster. '
                            'By default, HDBSCAN does not allow a single cluster (i.e. no structure)',
                default=False
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
