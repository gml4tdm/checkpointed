import os
import pickle
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases
from ...processing.text import TermFrequency


class CountVectors(checkpointed_core.PipelineStep, bases.DocumentDictEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        match label:
            case 'tf':
                return issubclass(step, TermFrequency)
            case 'dictionary':
                return issubclass(step, bases.WordIndexDictionarySource)
            case _:
                return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['tf', 'dictionary']

    async def execute(self, **inputs) -> typing.Any:
        tf_values = inputs['tf']
        dictionary = inputs['dictionary']
        unknown_word_policy = self.config.get_casted('params.unknown-word-policy', str)
        result = []
        for document in tf_values:
            document_result = {}
            for token, tf in document.items():
                if token in dictionary:
                    document_result[token] = tf
                elif unknown_word_policy == 'error':
                    raise ValueError(f'Unknown word for count vectorisation: {token}')
                else:
                    assert unknown_word_policy == 'ignore'
                    continue
            result.append(document_result)
        return result

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
        return {
            'unknown-word-policy': arguments.EnumArgument(
                name='unknown-word-policy',
                description='Policy on how to handle words not contained in the given word embedding.',
                options=['ignore', 'error']
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
