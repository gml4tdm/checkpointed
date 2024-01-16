import typing

from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import Sparse2Corpus

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from .... import bases


class LdaModel(checkpointed_core.PipelineStep):

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
        model = LdaMulticore(
            Sparse2Corpus(inputs['documents-matrix'], documents_columns=False),
            id2word={v: k for k, v in inputs['dictionary'].items()},
            num_topics=self.config.get_casted('params.number-of-topics', int),
            workers=self.config.get_casted('params.number-of-workers', int)
        )
        return model

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'gensim-lda'

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
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'lda-model': (LdaModel,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        model: LdaMulticore = inputs['lda-model']
        return model.show_topics(
            num_topics=self.config.get_casted('params.number-of-topics', int),
            num_words=self.config.get_casted('params.number-of-words', int)
        )

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
