import typing

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import checkpointed_core
from checkpointed_core.parameters import constraints, arguments

from ... import bases


def mean_pooling(model_output, attention_mask):
    # Adapted from
    # https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HuggingFaceDocumentEncoder(checkpointed_core.PipelineStep, bases.DocumentVectorEncoder):

    @classmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {
            'documents': (bases.FlattenedTokenizedDocumentSource,)
        }

    @classmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        return {}

    async def execute(self, **inputs) -> typing.Any:
        # model = transformers.pipeline(
        #     model=self.config.get_casted('params.huggingface-model', str)
        # )
        # TODO: checkpointed_core base for documents divided into sentences
        # TODO: loop over documents, encode each documents
        # TODO: convert documents to numpy arrays
        model_name = self.config.get_casted('params.huggingface-model', str)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return model(inputs['documents'])

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
            'huggingface-model': arguments.StringArgument(
                name='huggingface-model',
                description='The name of the HuggingFace model to use for document encoding. '
                            'This is passed to the transformers.pipeline() function.',
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
