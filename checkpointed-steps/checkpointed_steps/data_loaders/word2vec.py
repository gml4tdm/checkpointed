import typing

from gensim.models import KeyedVectors

from .shared import GenericFileLoader


class Word2VecLoader(GenericFileLoader):

    async def execute(self, *inputs) -> typing.Any:
        return KeyedVectors.load_word2vec_format(self.config['filename'])

    @staticmethod
    def save_result(path: str, result: typing.Any):
        result.save(path)

    @staticmethod
    def load_result(path: str):
        return KeyedVectors.load_word2vec_format(path)

