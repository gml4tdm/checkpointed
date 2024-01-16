import typing

import contractions

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class ExpandContractions(checkpointed_core.PipelineStep, bases.TextDocumentSource):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.TextDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        return [
            contractions.fix(document, slang=self.config.get_casted('params.fix-slang', bool))
            for document in inputs['documents']
        ]

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
            'fix-slang': arguments.BoolArgument(
                name='fix-slang',
                description='If True, also fix slang contractions such as U -> You.',
                default=False
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
