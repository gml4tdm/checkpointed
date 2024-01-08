import typing

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class Lemmatization(checkpointed_core.PipelineStep, bases.PartOfSpeechTokenizedDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return isinstance(step, bases.PartOfSpeechTokenizedDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['documents']

    @staticmethod
    def _map_tag(tag: str) -> str | None:
        # based on https://www.holisticseo.digital/python-seo/nltk/lemmatize
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    async def execute(self, **inputs) -> typing.Any:
        lemmatizer = WordNetLemmatizer()
        return [
            [
                [lemmatizer.lemmatize(word, pos=self._map_tag(tag)) for word, tag in sentence]
                for sentence in document
            ]
            for document in inputs['documents']
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
