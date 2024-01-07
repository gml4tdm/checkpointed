import typing

import pandas

from .shared import GenericFileLoader


class CSVLoader(GenericFileLoader):

    async def execute(self, *inputs) -> typing.Any:
        assert len(inputs) == 0
        return pandas.read_csv(
            self.config.get_casted('params.filename', str),
        )

    @staticmethod
    def get_data_format() -> str:
        return 'pandas-pickle'
