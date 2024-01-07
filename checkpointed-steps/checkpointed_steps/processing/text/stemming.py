import typing

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class PorterStemming(checkpointed_core.PipelineStep, bases.TokenizedDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TokenizedDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        documents = inputs['documents']
        stemmer = nltk.stem.PorterStemmer()
        return [
            [[stemmer.stem(word) for word in sent] for sent in document]
            for document in documents
        ]

    @staticmethod
    def get_data_format() -> str:
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
