class DataLoader:
    """Generic base class for all data loaders.
    """


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


class DocumentVectorEncoder:
    """Base class for all steps that return encoded document vectors.

    This means that documents are returned as vectors.
    """


class DocumentDictEncoder:
    """Base class for all steps that return encoded document dicts.

    This means that documents are returned as dicts.
    """


class WordIndexDictionarySource:
    """Base class for all classes that return word-to-index dictionaries
    for generating bag of words-like features.
    """
