import typing

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class RemoveStopwords(checkpointed_core.PipelineStep, bases.TokenizedDocumentSource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.TokenizedDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        documents = inputs['documents']
        return [
            [[word for word in sent if word not in stopwords] for sent in document]
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
