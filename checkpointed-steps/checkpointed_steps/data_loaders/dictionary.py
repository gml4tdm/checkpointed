import pickle
import typing

from .. import bases

from . import shared


class LoadWordToIndexDictionary(shared.GenericFileLoader, bases.WordIndexDictionarySource):

    async def execute(self, **inputs) -> typing.Any:
        with open(self.config.get_casted('params.filename', str), 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-pickle'
