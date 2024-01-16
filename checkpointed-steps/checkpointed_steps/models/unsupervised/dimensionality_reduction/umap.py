import typing

import numpy
import umap

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from .... import bases


class UMAPTraining(checkpointed_core.PipelineStep):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'data': (bases.NumericalVectorData,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

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

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-pickle'

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
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'data': (bases.NumericalVectorData,),
            'umap-model': (UMAPTraining,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        return inputs['umap-model'].transform(inputs['data'])

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'numpy-array'

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
