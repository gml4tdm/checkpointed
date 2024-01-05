import itertools
import os
import pickle
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class GenerateWordToIndexDictionary(checkpointed_core.PipelineStep, bases.WordIndexDictionarySource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.FlattenedTokenizedDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        word_index_mapping = {}
        for token in itertools.chain.from_iterable(inputs['documents']):
            if token not in word_index_mapping:
                word_index_mapping[token] = len(word_index_mapping)
        return word_index_mapping

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as file:
            return pickle.load(file)

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
