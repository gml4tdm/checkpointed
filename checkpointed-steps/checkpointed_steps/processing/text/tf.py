import pickle
import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class TermFrequency(checkpointed_core.PipelineStep):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.FlattenedTokenizedDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    async def execute(self, **inputs) -> typing.Any:
        result = []
        for document in inputs['documents']:
            tf = {}
            for token in document:
                tf[token] = tf.get(token, 0) + 1
            result.append(tf)
        return result

    @staticmethod
    def save_result(path: str, result: typing.Any):
        with open(path, 'wb') as file:
            pickle.dump(result, file)

    @staticmethod
    def load_result(path: str):
        with open(path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def is_deterministic() -> bool:
        return True

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
