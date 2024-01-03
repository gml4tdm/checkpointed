import pickle
import typing

import scipy.sparse

import checkpointed_core
from checkpointed_core import PipelineStep
from checkpointed_core.arg_spec import constraints, arguments

from ... import bases


class DictToSparseArray(checkpointed_core.PipelineStep, bases.DocumentSparseVectorEncoder):

    @classmethod
    def supports_step_as_input(cls, step: type[PipelineStep], label: str) -> bool:
        if label == 'documents':
            return issubclass(step, bases.DocumentDictEncoder)
        if label == 'word-to-index-dictionary':
            return issubclass(step, bases.WordIndexDictionarySource)
        return super(cls, cls).supports_step_as_input(step, label)

    async def execute(self, **inputs) -> typing.Any:
        word_to_index = inputs['word-to-index-dictionary']
        documents = inputs['documents']
        data = []
        row_ind = []
        col_ind = []
        for row, document in enumerate(documents):
            for token, col in document.items():
                data.append(col)
                row_ind.append(row)
                try:
                    col_ind.append(word_to_index[token])
                except KeyError:
                    raise ValueError(f'Dictionary has no entry for word: {token}')
        return scipy.sparse.csr_array((data, (row_ind, col_ind)))

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
