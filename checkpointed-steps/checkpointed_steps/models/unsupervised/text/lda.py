import json
import os
import typing

from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import Sparse2Corpus

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from .... import bases


class LdaModel(checkpointed_core.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents-matrix':
            return issubclass(step, bases.DocumentSparseVectorEncoder)
        if label == 'dictionary':
            return issubclass(step, bases.WordIndexDictionarySource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents-matrix', 'dictionary']

    async def execute(self, **inputs) -> typing.Any:
        model = LdaMulticore(
            Sparse2Corpus(inputs['documents-matrix'], documents_columns=False),
            id2word={v: k for k, v in inputs['dictionary'].items()},
            num_topics=self.config.get_casted('params.number-of-topics', int),
            workers=self.config.get_casted('params.number-of-workers', int)
        )
        return model

    @staticmethod
    def save_result(path: str, result: typing.Any):
        result.save(os.path.join(path, 'main.bin'))

    @staticmethod
    def load_result(path: str):
        return LdaMulticore.load(os.path.join(path, 'main.bin'))

    @staticmethod
    def is_deterministic() -> bool:
        return False

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'number-of-topics': arguments.IntArgument(
                name='number-of-topics',
                description='Number of topics to generate.',
                default=10,
                minimum=1
            ),
            'number-of-workers': arguments.IntArgument(
                name='number-of-workers',
                description='Number of workers to use.',
                default=1,
                minimum=1
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []


class ExtractLdaTopics(checkpointed_core.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'lda-model':
            return issubclass(step, LdaModel)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['lda-model']

    async def execute(self, **inputs) -> typing.Any:
        model: LdaMulticore = inputs['lda-model']
        return model.show_topics(
            num_topics=self.config.get_casted('params.number-of-topics', int),
            num_words=self.config.get_casted('params.number-of-words', int)
        )

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.json'), 'w') as file:
            json.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.json'), 'r') as file:
            return json.load(file)

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
            'number-of-topics': arguments.IntArgument(
                name='number-of-topics',
                description='Number of topics to generate.',
                minimum=1
            ),
            'number-of-words': arguments.IntArgument(
                name='number-of-words',
                description='Number of words to generate for each topic.',
                default=10,
                minimum=1
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
