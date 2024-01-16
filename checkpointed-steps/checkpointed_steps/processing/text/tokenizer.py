import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

import nltk.tokenize

from ... import bases


class Tokenize(checkpointed_core.PipelineStep, bases.TokenizedDocumentSource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.TextDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        documents = inputs['documents']
        return [
            [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(document)]
            for document in documents
        ]

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
