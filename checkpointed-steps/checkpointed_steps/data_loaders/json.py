import json
import typing

from .shared import GenericFileLoader


class JsonLoader(GenericFileLoader):

    async def execute(self, **inputs) -> typing.Any:
        assert len(inputs) == 0
        with open(self.config.get_casted('params.filename', str)) as file:
            return json.load(file)

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-json'
