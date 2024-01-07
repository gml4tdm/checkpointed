import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import arguments, constraints

import numpy

from ... import bases
from ...data_loaders import CWord2VecLoader, GensimWord2VecLoader, GloveLoader


class Word2VecEncoder(checkpointed_core.PipelineStep, bases.WordVectorEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TokenizedDocumentSource)
        elif label == 'vectors':
            return step in [
                CWord2VecLoader, GensimWord2VecLoader, GloveLoader
            ]
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents', 'vectors']

    async def execute(self, **inputs) -> typing.Any:
        vectors = inputs['vectors']
        documents = inputs['documents']
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
            for sent in document:
                for word in sent:
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
    def get_data_format() -> str:
        return 'numpy-array'

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

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True
