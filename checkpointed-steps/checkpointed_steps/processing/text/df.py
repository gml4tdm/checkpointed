import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class DocumentFrequency(checkpointed_core.PipelineStep):

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
        df = {}
        dictionary = inputs['dictionary']
        for document in inputs['documents']:
            for token in set(document):
                if token not in dictionary:
                    continue
                df[token] = df.get(token, 0) + 1
        return df

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
