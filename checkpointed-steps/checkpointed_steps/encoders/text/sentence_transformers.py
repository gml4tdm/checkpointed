import os
import typing

import numpy
from sentence_transformers import SentenceTransformer

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class SentenceTransformersDocumentEncoder(checkpointed_core.PipelineStep, bases.DocumentVectorEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TextDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        model = SentenceTransformer(
            self.config.get_casted('params.sentence-transformer-model', str)
        )
        return model.encode(inputs['documents'], convert_to_tensor=True, show_progress_bar=True)

    @staticmethod
    def save_result(path: str, result: typing.Any):
        numpy.save(os.path.join(path, 'main.npy'), result)

    @staticmethod
    def load_result(path: str):
        return numpy.load(os.path.join(path, 'main.npy'))

    def get_checkpoint_metadata(self) -> typing.Any:
        return {}

    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        return True

    @classmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {
            'sentence-transformer-model': arguments.StringArgument(
                name='sentence-transformer-model',
                description='The name of the sentence transformer model to use for document encoding.',
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
