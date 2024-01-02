import typing

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from .shared import GenericFileLoader


class GloveLoader(GenericFileLoader):

    async def execute(self, *inputs) -> typing.Any:
        temp_file = self.config['filename'] + '_converted.temp'
        glove2word2vec(self.config['filename'], temp_file)
        return KeyedVectors.load_word2vec_format(temp_file)

    @staticmethod
    def save_result(path: str, result: typing.Any):
        result.save(path)

    @staticmethod
    def load_result(path: str):
        return KeyedVectors.load_word2vec_format(path)

