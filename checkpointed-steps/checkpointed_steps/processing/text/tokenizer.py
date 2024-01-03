import os
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

import nltk.tokenize

from ... import bases

import pickle


class Tokenize(checkpointed_core.PipelineStep, bases.TokenizedDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TextDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    async def execute(self, **inputs) -> typing.Any:
        documents = inputs['documents']
        return [
            [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(document)]
            for document in documents
        ]

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def is_deterministic() -> bool:
        return True

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
