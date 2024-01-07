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
        if label == 'dictionary':
            return issubclass(step, bases.WordIndexDictionarySource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['documents', 'dictionary']

    async def execute(self, **inputs) -> typing.Any:
        result = []
        dictionary = inputs['dictionary']
        for document in inputs['documents']:
            tf = {}
            for token in document:
                if token in dictionary:
                    tf[token] = tf.get(token, 0) + 1
            result.append(tf)
        return result

    @staticmethod
    def get_data_format() -> str:
        return 'std-pickle'

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
