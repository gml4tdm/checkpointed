import typing

from gensim.models import FastText

from .shared import GenericFileLoader


class FastTextLoader(GenericFileLoader):

    async def execute(self, **inputs) -> typing.Any:
        assert len(inputs) == 0
        return FastText.load(self.config.get_casted('params.filename', str))

    @staticmethod
    def get_data_format() -> str:
        return 'gensim-fasttext'
