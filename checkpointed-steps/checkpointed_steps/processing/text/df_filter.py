import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases
from .df import DocumentFrequency


class DocumentFrequencyFilter(checkpointed_core.PipelineStep, bases.WordIndexDictionarySource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'df': (DocumentFrequency,),
            'documents': (bases.FlattenedTokenizedDocumentSource,),
            'word-to-index-dictionary': (bases.WordIndexDictionarySource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        result = {}
        total_documents = len(inputs['documents'])
        for token in inputs['word-to-index-dictionary']:
            try:
                count = inputs['df'][token]
            except KeyError:
                raise ValueError(f'Word in dictionary not contained in document frequency mapping: {token}')
            match self.config.get_casted('params.minimum-inclusion-check-mode', str):
                case 'count':
                    if count < self.config.get_casted('params.minimum-inclusion-count', int):
                        continue
                case 'fraction':
                    if count / total_documents < self.config.get_casted('params.minimum-inclusion-fraction', float):
                        continue
            match self.config.get_casted('params.maximum-inclusion-check-mode', str):
                case 'count':
                    if count > self.config.get_casted('params.maximum-inclusion-count', int):
                        continue
                case 'fraction':
                    if count / total_documents > self.config.get_casted('params.maximum-inclusion-fraction', float):
                        continue
            result[token] = len(result)
        return result

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-pickle'

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'minimum-inclusion-check-mode': arguments.EnumArgument(
                name='minimum-inclusion-check-mode',
                description='How to determine the minimum criteria for a word to be included. '
                            'Either `count` or `fraction`.',
                options=['count', 'fraction']
            ),
            'minimum-inclusion-count': arguments.IntArgument(
                name='minimum-inclusion-count',
                description='The minimum number of documents a word must be included in.',
                default=1,
                minimum=1,
                enabled_if=constraints.Equal(
                    constraints.ArgumentRef('minimum-inclusion-check-mode'),
                    constraints.Constant('count')
                )
            ),
            'minimum-inclusion-fraction': arguments.FloatArgument(
                name='minimum-inclusion-fraction',
                description='The minimum fraction of documents a word must be included in.',
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                enabled_if=constraints.Equal(
                    constraints.ArgumentRef('minimum-inclusion-check-mode'),
                    constraints.Constant('fraction')
                )
            ),
            'maximum-inclusion-check-mode': arguments.EnumArgument(
                name='maximum-inclusion-check-mode',
                description='How to determine the maximum criteria for a word to be included. '
                            'Either `count` or `fraction`.',
                options=['count', 'fraction']
            ),
            'maximum-inclusion-count': arguments.IntArgument(
                name='maximum-inclusion-count',
                description='The maximum number of documents a word must be included in.',
                default=1,
                minimum=1,
                enabled_if=constraints.Equal(
                    constraints.ArgumentRef('maximum-inclusion-check-mode'),
                    constraints.Constant('count')
                )
            ),
            'maximum-inclusion-fraction': arguments.FloatArgument(
                name='maximum-inclusion-fraction',
                description='The maximum fraction of documents a word must be included in.',
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                enabled_if=constraints.Equal(
                    constraints.ArgumentRef('maximum-inclusion-check-mode'),
                    constraints.Constant('fraction')
                )
            ),
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return [
            constraints.BooleanConstraint(
                expr=constraints.LessThanOrEqual(
                    constraints.ArgumentRef('params.minimum-inclusion-count'),
                    constraints.ArgumentRef('params.maximum-inclusion-count')
                ),
                message='The minimum inclusion count must be less than or equal to the maximum inclusion count.'
            ),
            constraints.BooleanConstraint(
                expr=constraints.LessThanOrEqual(
                    constraints.ArgumentRef('params.minimum-inclusion-fraction'),
                    constraints.ArgumentRef('params.maximum-inclusion-fraction')
                ),
                message='The minimum inclusion fraction must be less than or equal to the maximum inclusion fraction.'
            )
        ]
