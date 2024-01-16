import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class TermFrequency(checkpointed_core.PipelineStep):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.FlattenedTokenizedDocumentSource,),
            'dictionary': (bases.WordIndexDictionarySource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        result = []
        dictionary = inputs['dictionary']
        for document in inputs['documents']:
            tf = {}
            for token in document:
                if token in dictionary:
                    tf[token] = tf.get(token, 0) + 1
            result.append(tf)
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
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
