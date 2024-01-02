import typing

import checkpointed
from checkpointed import PipelineStep
from checkpointed.arg_spec import constraints, arguments

import nltk.tokenize

from ... import bases

import pickle


class Tokenize(checkpointed.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        return isinstance(step, bases.TextDocumentSource)

    async def execute(self, *inputs) -> typing.Any:
        documents, = inputs
        return [
            [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(document)]
            for document in documents
        ]

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(path, 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(path, 'rb') as file:
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
