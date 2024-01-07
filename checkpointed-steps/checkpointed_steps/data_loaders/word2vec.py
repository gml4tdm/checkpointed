import os.path
import typing

import gensim.models
from checkpointed_core.arg_spec import arguments
from gensim.models import KeyedVectors

from .shared import GenericFileLoader


class CWord2VecLoader(GenericFileLoader):

    async def execute(self, **inputs) -> typing.Any:
        assert len(inputs) == 0
        return KeyedVectors.load_word2vec_format(
            self.config.get_casted('params.filename', str),
            binary=self.config.get_casted('params.file-is-binary', bool),
        )

    @staticmethod
    def get_data_format() -> str:
        return 'gensim-c-word2vec'

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return super(cls, cls).get_arguments() | {
            'file-is-binary': arguments.BoolArgument(
                name='file-is-binary',
                description='Indicate whether the word embedding file is stored in text or binary format.',
                default=False
            )
        }


class GensimWord2VecLoader(GenericFileLoader):

    async def execute(self, *inputs) -> typing.Any:
        return gensim.models.Word2Vec.load(
            self.config.get_casted('params.filename', str)
        )

    @staticmethod
    def get_data_format() -> str:
        return 'gensim-word2vec'

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return super(cls, cls).get_arguments()
