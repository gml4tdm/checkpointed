import os
import typing

import numpy
import umap

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from .... import bases

import pickle


class UMAPTraining(checkpointed_core.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'data':
            return issubclass(step, bases.NumericalVectorData)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['data']

    async def execute(self, **inputs) -> typing.Any:
        if self.config.get_casted('params.seed', int) != -1:
            numpy.random.seed(self.config.get_casted('params.seed', int))
        model = umap.UMAP(
            n_neighbors=self.config.get_casted('params.n-neighbors', int),
            min_dist=self.config.get_casted('params.min-dist', float),
            n_components=self.config.get_casted('params.n-components', int),
            metric=self.config.get_casted('params.metric', str),
        )
        return model.fit(inputs['data'])

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as f:
            return pickle.load(f)

    def get_checkpoint_metadata(self) -> typing.Any:
        return {'seed': self.config.get_casted('params.seed', int)}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return metadata['seed'] != -1

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'seed': arguments.IntArgument(
                name='seed',
                description='The seed to use for the UMAP algorithm. -1 for no seed.',
                default=-1,
            ),
            'n-neighbors': arguments.IntArgument(
                name='n_neighbors',
                description='The number of neighbors to use for the UMAP algorithm.',
                minimum=1
            ),
            'min-dist': arguments.FloatArgument(
                name='min_dist',
                description='The minimum distance to use for the UMAP algorithm.',
                minimum=0.0
            ),
            'n-components': arguments.IntArgument(
                name='n_components',
                description='The number of components to use for the UMAP algorithm.',
                minimum=1
            ),
            'metric': arguments.EnumArgument(
                name='metric',
                description='The metric to use for the UMAP algorithm.',
                options=[
                    # Minkowski-style metrics
                    'euclidian', 'manhattan', 'chebyshev', 'minkowski',

                    # Miscellaneous spatial metrics
                    'canberra', 'braycurtis', 'haversine',

                    # Normalised spatial metrics
                    'mahalanobis', 'wminkowski', 'seuclidean',

                    # Angular and correlation metrics
                    'cosine', 'correlation',

                    # Metrics for binary data
                    'hamming', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto',
                    'russellrao', 'sokalmichener', 'sokalsneath', 'yule',
                ]
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []


class UMAPTransform(checkpointed_core.PipelineStep, bases.DenseNumericalVectorData):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'data':
            return issubclass(step, bases.NumericalVectorData)
        if label == 'umap-model':
            return issubclass(step, UMAPTraining)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['data', 'umap-model']

    async def execute(self, **inputs) -> typing.Any:
        return inputs['umap-model'].transform(inputs['data'])

    @staticmethod
    def save_result(path: str, result: typing.Any):
        numpy.save(os.path.join(path, 'main.npy'), result)

    @staticmethod
    def load_result(path: str):
        return numpy.load(os.path.join(path, 'main.npy'))

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
