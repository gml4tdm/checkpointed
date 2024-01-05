import os
import pickle
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class CaseTransform(checkpointed_core.PipelineStep, bases.TextDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TextDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        match self.config.get_casted('params.mode', str):
            case 'lower':
                return [document.lower() for document in inputs['documents']]
            case 'upper':
                return [document.upper() for document in inputs['documents']]

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(os.path.join(path, 'main.pickle'), 'rb') as file:
            return pickle.load(file)

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
