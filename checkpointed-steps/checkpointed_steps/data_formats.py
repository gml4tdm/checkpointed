##############################################################################
##############################################################################
# Imports
##############################################################################

from __future__ import annotations

import json
import os.path
import pickle
import typing

import gensim
import numpy
import pandas
import scipy

from checkpointed_core import data_format
from checkpointed_core.data_format import DataFormat


##############################################################################
##############################################################################
# Functions
##############################################################################


def register_formats():
    data_format.register_format('std-json', JsonFormat)
    data_format.register_format('std-pickle', PickleFormat)
    data_format.register_format('std-text', TextFormat)
    data_format.register_format('std-binary', BinaryFormat)
    data_format.register_format('numpy-array', NumpyFormat)
    data_format.register_format('gensim-word2vec', GensimWord2VecFormat)
    data_format.register_format('gensim-c-word2vec', GensimCWord2VecFormat)
    data_format.register_format('gensim-fasttext', GensimFastTextFormat)
    data_format.register_format('gensim-lda', GensimLdaFormat)
    data_format.register_format('gensim-lsi', GensimLsiFormat)
    data_format.register_format('pandas-pickle', PandasPickleFormat)
    data_format.register_format('scipy-sparse-matrix', ScipySparseFormat)


def format_from_functions(name: str, *,
                          saving_function,
                          loading_function) -> type[DataFormat]:
    return typing.cast(
        type[DataFormat],
        type(
            name,
            (DataFormat,),
            {
                'store': staticmethod(saving_function),
                'load': staticmethod(loading_function),
            }
        )
    )


##############################################################################
##############################################################################
# Data Format Implementations
##############################################################################


class JsonFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        with open(os.path.join(path, 'main.json'), 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(path: str) -> typing.Any:
        with open(os.path.join(path, 'main.json'), 'r') as f:
            return json.load(f)


class PickleFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        with open(os.path.join(path, 'main.pickle'), 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str) -> typing.Any:
        with open(os.path.join(path, 'main.pickle'), 'rb') as f:
            return pickle.load(f)


class TextFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        assert isinstance(data, str)
        with open(os.path.join(path, 'main.txt'), 'w') as f:
            f.write(data)

    @staticmethod
    def load(path: str) -> typing.Any:
        with open(os.path.join(path, 'main.txt'), 'r') as f:
            return f.read()


class BinaryFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        assert isinstance(data, (bytes, bytearray))
        with open(os.path.join(path, 'main.bin'), 'wb') as f:
            f.write(data)

    @staticmethod
    def load(path: str) -> typing.Any:
        with open(os.path.join(path, 'main.bin'), 'rb') as f:
            return f.read()


class NumpyFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        numpy.save(os.path.join(path, 'main.npy'), data)

    @staticmethod
    def load(path: str) -> typing.Any:
        return numpy.load(os.path.join(path, 'main.npy'))


class GensimWord2VecFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.wv.save_word2vec_format(
            os.path.join(path, 'main.bin'), binary=True
        )

    @staticmethod
    def load(path: str) -> typing.Any:
        return gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(path, 'main.bin'), binary=True
        )


class GensimCWord2VecFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.save_word2vec_format(
            os.path.join(path, 'main.bin'), binary=True
        )

    @staticmethod
    def load(path: str) -> typing.Any:
        return gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(path, 'main.bin'), binary=True
        )


class GensimFastTextFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.save(os.path.join(path, 'main.bin'))

    @staticmethod
    def load(path: str) -> typing.Any:
        return gensim.models.FastText.load(
            os.path.join(path, 'main.bin')
        )


class GensimLdaFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.save(os.path.join(path, 'main.bin'))

    @staticmethod
    def load(path: str) -> typing.Any:
        return gensim.models.LdaMulticore.load(
            os.path.join(path, 'main.bin')
        )


class GensimLsiFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.save(os.path.join(path, 'main.bin'))

    @staticmethod
    def load(path: str) -> typing.Any:
        return gensim.models.LsiModel.load(
            os.path.join(path, 'main.bin')
        )


class PandasPickleFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        data.to_pickle(os.path.join(path, 'main.pickle'))

    @staticmethod
    def load(path: str) -> typing.Any:
        return pandas.read_pickle(os.path.join(path, 'main.pickle'))


class ScipySparseFormat(DataFormat):

    @staticmethod
    def store(path: str, data: typing.Any):
        scipy.sparse.save_npz(os.path.join(path, 'main.npz'), data)

    @staticmethod
    def load(path: str) -> typing.Any:
        return scipy.sparse.load_npz(os.path.join(path, 'main.npz'))
