class DataLoader:
    """Generic base class for all data loaders.
    """


class LabelAssignment:
    pass


class NumericalVectorData:
    pass


class DenseNumericalVectorData(NumericalVectorData):
    pass


class SparseNumericalVectorData(NumericalVectorData):
    pass


class TextDocumentSource:
    """Base class for all steps that return documents.

    Result type: list[str]
    """


class TokenizedDocumentSource:
    """Base class for all steps that return
    tokenized documents.

    Result type: list[list[list[str]]]

    A list of tokenized documents, where each document is
    split up in a list of sentences, where each sentence is
    a list of words.
    """


class PartOfSpeechTokenizedDocumentSource:
    """Base class for all steps that return tokenized documents
    with part of speech tags attached.
    
    Result type: list[list[list[tuple[str, str]]]
    
    Here, the first entry is the word itself, and the second 
    tuple entry is the POS tag as returned by NLTK.
    """


class FlattenedTokenizedDocumentSource:
    """Base class for all steps that return
    flattened tokenized documents.

    Result type: list[list[str]]

    A list of tokenized documents, with no preservation
    of sentence boundaries.
    """


class WordVectorEncoder:
    """Base class for all steps that return encoded word vectors.

    This means that documents are returned as matrices.
    """


class DocumentVectorEncoder(DenseNumericalVectorData):
    """Base class for all steps that return encoded document vectors.

    This means that documents are returned as vectors.
    """


class DocumentSparseVectorEncoder(SparseNumericalVectorData):
    """Base class for all steps that return encoded document vectors.

    This means that documents are returned as sparse vectors.
    """


class DocumentDictEncoder:
    """Base class for all steps that return encoded document dicts.

    This means that documents are returned as dicts.
    """


class WordIndexDictionarySource:
    """Base class for all classes that return word-to-index dictionaries
    for generating bag of words-like features.
    """
