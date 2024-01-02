import typing

import checkpointed
from checkpointed import PipelineStep
from checkpointed.arg_spec import arguments, constraints

import numpy

from ... import bases
from ...data_loaders import Word2VecLoader, GloveLoader


class Word2VecEncoder(checkpointed.PipelineStep, bases.WordVectorEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep]) -> bool:
        return step == Word2VecLoader or step == GloveLoader

    async def execute(self, *inputs) -> typing.Any:
        vectors, documents = inputs
        replacement_vector = None
        if self.config.get_casted('params.unknown-word-policy', str) == 'replace':
            key = self.config.get_casted('params.unknown-word-replacement', str)
            try:
                replacement_vector = vectors[key]
            except KeyError:
                raise ValueError(f'Replacement word "{key}" not found in word embedding')
        result = []
        for document in documents:
            document_vectors = []
            for word in document:
                try:
                    document_vectors.append(vectors[word])
                except KeyError:
                    match self.config.get_casted('params.unknown-word-policy', str):
                        case 'ignore':
                            pass
                        case 'error':
                            raise KeyError(f'Word "{word}" not found in word embedding')
                        case 'replace':
                            document_vectors.append(replacement_vector)
            result.append(numpy.vstack(document_vectors))
        return numpy.vstack(result)

    @staticmethod
    def save_result(path: str, result: typing.Any):
        numpy.save(path, result)

    @staticmethod
    def load_result(path: str):
        return numpy.load(path)

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'unknown-word-policy': arguments.EnumArgument(
                name='unknown-word-policy',
                description='Policy on how to handle words not contained in the given word embedding.',
                options=['ignore', 'error', 'replace']
            ),
            'unknown-word-replacement': arguments.StringArgument(
                name='unknown-word-replacement',
                description='Replacement for words not contained in the given word embedding. '
                            'Only used if unknown-word-policy == "replace"',
                enabled_if=constraints.Equal(
                    constraints.ArgumentRef('unknown-word-policy'),
                    constraints.Constant('replace')
                )
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True
