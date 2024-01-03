import os
import pickle
import typing

from .. import bases

from . import shared


class LoadWordToIndexDictionary(shared.GenericFileLoader, bases.WordIndexDictionarySource):

    async def execute(self, **inputs) -> typing.Any:
        return self.load_result(self.config.get_casted('params.filename', str))

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as file:
            return pickle.load(file)
