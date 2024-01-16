import typing

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class CaseTransform(checkpointed_core.PipelineStep, bases.TextDocumentSource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.TextDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        match self.config.get_casted('params.mode', str):
            case 'lower':
                return [document.lower() for document in inputs['documents']]
            case 'upper':
                return [document.upper() for document in inputs['documents']]

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'std-pickle'

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'mode': arguments.EnumArgument(
                name='mode',
                description='The case transform to perform. Either "lower" or "upper".',
                options=['lower', 'upper']
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
