import typing

from sentence_transformers import SentenceTransformer

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


class SentenceTransformersDocumentEncoder(checkpointed_core.PipelineStep, bases.DocumentVectorEncoder):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.FlattenedTokenizedDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        model = SentenceTransformer(
            self.config.get_casted('params.sentence-transformer-model', str)
        )
        return model.encode(inputs['documents'], convert_to_tensor=True, show_progress_bar=True)

    @classmethod
    def get_output_storage_format(cls) -> str:
        return 'numpy-array'

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
