import itertools
import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class GenerateWordToIndexDictionary(checkpointed_core.PipelineStep, bases.WordIndexDictionarySource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.FlattenedTokenizedDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        word_index_mapping = {}
        for token in itertools.chain.from_iterable(inputs['documents']):
            if token not in word_index_mapping:
                word_index_mapping[token] = len(word_index_mapping)
        return word_index_mapping

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-pickle'

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
