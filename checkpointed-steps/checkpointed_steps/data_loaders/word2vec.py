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
    def save_result(path: str, result: typing.Any):
        #result.save(path, binary=True)
        result.save_word2vec_format(path, binary=True)

    @staticmethod
    def load_result(path: str):
        return KeyedVectors.load_word2vec_format(path, binary=True)

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
    def save_result(path: str, result: typing.Any):
        result.wv.save_word2vec_format(path, binary=True)

    @staticmethod
    def load_result(path: str):
        return KeyedVectors.load_word2vec_format(path, binary=True)

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return super(cls, cls).get_arguments()
