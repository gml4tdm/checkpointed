import typing

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from .shared import GenericFileLoader


class GloveLoader(GenericFileLoader):

    async def execute(self, **inputs) -> typing.Any:
        assert len(inputs) == 0
        temp_file = self.config.get_casted('params.filename', str) + '_converted.temp'
        glove2word2vec(self.config.get_casted('params.filename', str), temp_file)
        return KeyedVectors.load_word2vec_format(temp_file)

    @staticmethod
    def get_data_format() -> str:
        return 'gensim-word2vec'
