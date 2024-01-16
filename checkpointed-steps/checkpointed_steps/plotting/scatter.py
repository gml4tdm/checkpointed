import typing

import matplotlib
import matplotlib.pyplot as pyplot

import checkpointed_core
import numpy
from checkpointed_core.parameters import constraints, arguments

from .. import bases


class LabelledScatter(checkpointed_core.PipelineStep):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'data': (bases.NumericalVectorData,),
            'labels': (bases.LabelAssignment,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        fig, ax = pyplot.subplots()
        points = inputs['data']
        colors = inputs['labels']
        cmap = pyplot.cm.get_cmap(self.config.get_casted('params.cmap', str))
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = numpy.linspace(colors.min(), colors.max(), colors.max() - colors.min() + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            cmap=cmap,
            norm=norm,
            alpha=self.config.get_casted('params.alpha', float)
        )
        color_bar = pyplot.colorbar(scatter, spacing='proportional', ticks=bounds, ax=ax)
        # Legend corresponding to label colors
        # ax.legend(
        #     handles=list(scatter.legend_elements()[0]),
        #     labels=list(map(str, range(colors.min(), colors.max() + 1))),
        #     bbox_to_anchor=(1.04, 0.5),
        #     loc="center left",  # Center right
        #     borderaxespad=0
        # )
        return fig, [ax]

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'matplotlib-png'

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'alpha': arguments.FloatArgument(
                name='alpha',
                description='Set the transparency of points.',
                default=1.0
            ),
            'cmap': arguments.StringArgument(
                name='cmap',
                description='Colormap used to color points.',
                default='viridis'
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
