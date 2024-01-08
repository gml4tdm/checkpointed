import typing

import nltk

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class PartOfSpeechTagging(checkpointed_core.PipelineStep, bases.PartOfSpeechTokenizedDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.TextDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        return [
            [nltk.tag.pos_tag(sent) for sent in nltk.tokenize.sent_tokenize(document)]
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
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []


class DropPartOfSpeech(checkpointed_core.PipelineStep, bases.TokenizedDocumentSource):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.PartOfSpeechTokenizedDocumentSource)
        return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list[str | type(...)]:
        return ['documents']

    async def execute(self, **inputs) -> typing.Any:
        return [
            [[word for word, _ in sentence] for sentence in document]
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
        return {}

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
