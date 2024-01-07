import typing

from gensim.models.lsimodel import LsiModel as _LsiModel
from gensim.matutils import Sparse2Corpus

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from .... import bases


class LsiModel(checkpointed_core.PipelineStep):

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
        model = _LsiModel(
            Sparse2Corpus(inputs['documents-matrix'], documents_columns=False),
            id2word={v: k for k, v in inputs['dictionary'].items()},
            num_topics=self.config.get_casted('params.number-of-topics', int),
        )
        return model

    @staticmethod
    def get_data_format() -> str:
        return 'gensim-lsi'

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
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []


class ExtractLsiTopics(checkpointed_core.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'lsi-model':
            return issubclass(step, LsiModel)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['lsi-model']

    async def execute(self, **inputs) -> typing.Any:
        model:  _LsiModel = inputs['lsi-model']
        topics = model.show_topics(
            num_topics=-1,
            num_words=self.config.get_casted('params.number-of-words', int),
            formatted=False
        )
        return {
            num: [
                {
                    'word': word,
                    'prob': prob
                }
                for word, prob in words
            ]
            for num, words in topics
        }

    @staticmethod
    def get_data_format() -> str:
        return 'std-json'

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
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
