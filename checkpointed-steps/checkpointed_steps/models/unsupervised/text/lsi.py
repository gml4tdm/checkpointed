import typing

from gensim.models.lsimodel import LsiModel as _LsiModel
from gensim.matutils import Sparse2Corpus

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from .... import bases


class LsiModel(checkpointed_core.PipelineStep):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents-matrix': (bases.DocumentSparseVectorEncoder,),
            'dictionary': (bases.WordIndexDictionarySource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        model = _LsiModel(
            Sparse2Corpus(inputs['documents-matrix'], documents_columns=False),
            id2word={v: k for k, v in inputs['dictionary'].items()},
            num_topics=self.config.get_casted('params.number-of-topics', int),
        )
        return model

    @classmethod
    def get_output_storage_format(cls) -> str:
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
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'lsi-model': (LsiModel,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

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

    @classmethod
    def get_output_storage_format(cls) -> str:
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
