import typing

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases
from ...processing.text import TermFrequency, DocumentFrequency, DocumentFrequencyFilter

import math


class TFIDF(checkpointed_core.PipelineStep, bases.DocumentDictEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        match label:
            case 'tf':
                return issubclass(step, TermFrequency)
            case 'df':
                return issubclass(step, (DocumentFrequency, DocumentFrequencyFilter))
            case 'dictionary':
                return issubclass(step, bases.WordIndexDictionarySource)
            case _:
                return super(cls, cls).supports_step_as_input(step, label)

    @staticmethod
    def get_input_labels() -> list:
        return ['tf', 'df', 'dictionary']

    async def execute(self, **inputs) -> typing.Any:
        tf_values = inputs['tf']
        df_by_word = inputs['df']
        dictionary = inputs['dictionary']
        n = len(tf_values)
        unknown_word_policy = self.config.get_casted('unknown-word-policy', str)
        result = []
        for document in tf_values:
            document_result = {}
            for token, tf in document.items():
                if unknown_word_policy == 'adjusted-idf':
                    if token in dictionary:
                        try:
                            df = df_by_word[token]
                        except KeyError:
                            raise ValueError(f'Word in dictionary not contained in document frequency mapping: {token}')
                    else:
                        df = 0
                    idf = math.log((n + 1) / (df + 1))
                elif token in dictionary:
                    try:
                        df = df_by_word[token]
                    except KeyError:
                        raise ValueError(f'Word in dictionary not contained in document frequency mapping: {token}')
                    idf = math.log(n / df)
                elif unknown_word_policy == 'error':
                    raise ValueError(f'Unknown word: {token}')
                else:
                    assert unknown_word_policy == 'ignore'
                    continue
                document_result[token] = tf * idf
            result.append(document_result)
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
        return {
             'unknown-word-policy': arguments.EnumArgument(
                name='unknown-word-policy',
                description='Policy on how to handle words not contained in the given word embedding.',
                options=['ignore', 'error', 'adjusted-idf']
            )
        }

    @classmethod
    def get_constraints(cls) -> list[constraints.Constraint]:
        return []
