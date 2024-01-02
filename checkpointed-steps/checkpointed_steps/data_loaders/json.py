import json
import typing

from .shared import GenericFileLoader


class JsonLoader(GenericFileLoader):

    async def execute(self, **inputs) -> typing.Any:
        assert len(inputs) == 0
        with open(self.config.get_casted('params.filename', str)) as file:
            return json.load(file)

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(path, 'w') as file:
            json.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(path, 'r') as file:
            return json.load(file)
