import typing

import contractions

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class ExpandContractions(checkpointed_core.PipelineStep, bases.TextDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TextDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        return [
            contractions.fix(document, slang=self.config.get_casted('params.fix-slang', bool))
            for document in inputs['documents']
        ]

    @staticmethod
    def get_data_format() -> str:
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
