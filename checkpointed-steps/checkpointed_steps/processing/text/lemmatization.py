import typing

import nltk

try:
    nltk.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


class Lemmatization(checkpointed_core.PipelineStep, bases.PartOfSpeechTokenizedDocumentSource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.PartOfSpeechTokenizedDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}


    @staticmethod
    def _map_tag(tag: str) -> str | None:
        return POS_CONVERSION.get(tag, 'n')

    async def execute(self, **inputs) -> typing.Any:
        lemmatizer = WordNetLemmatizer()
        return [
            [
                [(lemmatizer.lemmatize(word, pos=self._map_tag(tag)), tag) for word, tag in sentence]
                for sentence in document
            ]
            for document in inputs['documents']
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
