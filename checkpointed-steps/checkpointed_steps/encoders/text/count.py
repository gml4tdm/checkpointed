import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases
from ...processing.text import TermFrequency


class CountVectors(checkpointed_core.PipelineStep, bases.DocumentDictEncoder):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'tf': (TermFrequency,),
            'dictionary': (bases.WordIndexDictionarySource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        tf_values = inputs['tf']
        dictionary = inputs['dictionary']
        unknown_word_policy = self.config.get_casted('params.unknown-word-policy', str)
        result = []
        for document in tf_values:
            document_result = {}
            for token, tf in document.items():
                if token in dictionary:
                    document_result[token] = tf
                elif unknown_word_policy == 'error':
                    raise ValueError(f'Unknown word for count vectorisation: {token}')
                else:
                    assert unknown_word_policy == 'ignore'
                    continue
            result.append(document_result)
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
            'unknown-word-policy': arguments.EnumArgument(
                name='unknown-word-policy',
                description='Policy on how to handle words not contained in the given word embedding.',
                options=['ignore', 'error']
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
