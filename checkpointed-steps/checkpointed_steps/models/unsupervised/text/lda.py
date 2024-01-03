import json
import typing

from gensim.models.ldamulticore import LdaMulticore

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

    async def execute(self, **inputs) -> typing.Any:
        model = LdaMulticore(
            inputs['documents-matrix'],
            id2word={v: k for k, v in inputs['dictionary'].items()},
            num_topics=self.config.get_casted('params.number-of-topics', int),
            workers=self.config.get_casted('params.number-of-workers', int)
        )
        return model

    @staticmethod
    def save_result(path: str, result: typing.Any):
        result.save(path)

    @staticmethod
    def load_result(path: str):
        return LdaMulticore.load(path)

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

    async def execute(self, **inputs) -> typing.Any:
        model: LdaMulticore = inputs['lda-model']
        return model.show_topics(
            num_topics=self.config.get_casted('params.number-of-topics', int),
            num_words=self.config.get_casted('params.number-of-words', int)
        )

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(path, 'w') as f:
            json.dump(result, f)

    @staticmethod
    def load_result(path: str):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    def get_arguments(self) -> dict[str, arguments.Argument]:
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
