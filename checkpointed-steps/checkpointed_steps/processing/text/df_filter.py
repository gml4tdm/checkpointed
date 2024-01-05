import os
import pickle
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases
from .df import DocumentFrequency


class DocumentFrequencyFilter(checkpointed_core.PipelineStep, bases.WordIndexDictionarySource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'df':
            return issubclass(step, DocumentFrequency)
        elif label == 'documents':
            return issubclass(step, bases.FlattenedTokenizedDocumentSource)
        elif label == 'word-to-index-dictionary':
            return issubclass(step, bases.WordIndexDictionarySource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['df', 'documents', 'word-to-index-dictionary']

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

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as file:
            return pickle.load(file)

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
