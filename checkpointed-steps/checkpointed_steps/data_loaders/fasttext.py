import typing

from gensim.models import FastText

from .shared import GenericFileLoader


class FastTextLoader(GenericFileLoader):

    async def execute(self, *inputs) -> typing.Any:
        return FastText.load(self.config["filename"])

    @staticmethod
    def save_result(path: str, result: typing.Any):
        result.save(path)

    @staticmethod
    def load_result(path: str):
        return FastText.load(path)

