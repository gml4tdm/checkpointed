import pickle
import typing

from .. import bases

from . import shared


class LoadWordToIndexDictionary(shared.GenericFileLoader, bases.WordIndexDictionarySource):

    async def execute(self, **inputs) -> typing.Any:
        return self.load_result(self.config.get_casted('params.filename', str))

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(path, 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(path, 'rb') as file:
            return pickle.load(file)
