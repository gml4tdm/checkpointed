import abc
import hashlib
import typing

import checkpointed
from checkpointed.arg_spec.constraints import Constraint
from checkpointed.arg_spec.arguments import Argument, StringArgument

from .. import bases


class GenericFileLoader(checkpointed.PipelineStep, bases.DataLoader, abc.ABC):

    @classmethod
    def supports_step_as_input(cls, step: type[checkpointed.PipelineStep]) -> bool:
        return False

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return {
            'filename': StringArgument(
                name='filename',
                description='Path to the file to load'
            )
        }

    @classmethod
    def get_constraints(cls) -> list[Constraint]:
        return []

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        with open(self.config.get('params.filename'), 'rb') as file:
            return {'file_hash': hashlib.sha256(file.read()).hexdigest()}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        with open(self.config.get('params.filename'), 'rb') as file:
            return metadata['file_hash'] == hashlib.sha256(file.read()).hexdigest()
