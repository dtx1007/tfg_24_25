import pandas as pd

from collections import Counter


def build_vocab_from_iterator(
    iterator: list[list[str]], specials: list[str] | None = None, min_freq: int = 1
) -> dict[str, int]:
    """
    Builds a vocabulary from an iterator of token lists.

    This function replaces torchtext.vocab.build_vocab_from_iterator and returns
    a simple dictionary mapping tokens to indices.

    Parameters
    ----------
    iterator : list[list[str]]
        An iterator yielding lists of tokens.
    specials : list[str], optional
        A list of special tokens to add to the beginning of the vocabulary.
    min_freq : int, optional
        The minimum frequency for a token to be included in the vocabulary.

    Returns
    -------
    dict[str, int]
        A dictionary mapping tokens to their integer indices.
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    # Sort by frequency (descending) and then by token (ascending)
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    vocab = {}
    index = 0

    # Add special tokens first
    if specials:
        for token in specials:
            if token not in vocab:
                vocab[token] = index
                index += 1

    # Add tokens from the data
    for word, freq in sorted_items:
        if freq >= min_freq and word not in vocab:
            vocab[word] = index
            index += 1

    return vocab


def create_vocab_for_column(
    df: pd.DataFrame, col: str, specials: list[str] = ["<PAD>", "<UNK>", "<EMPTY>"]
) -> dict[str, int]:
    """
    Create a vocabulary for a specific column in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    col : str
        The column name for which to create the vocabulary.
    specials : list
        List of special tokens to include in the vocabulary.

    Returns
    -------
    vocab : torchtext.vocab.Vocab
        The vocabulary object mapping tokens to indices.
    """
    all_tokens = [token for token_list in df[col] for token in token_list]
    return build_vocab_from_iterator([all_tokens], specials=specials)


def create_char_vocab(
    char_set: set[str], specials: list[str] = ["<PAD>", "<UNK>", "<EMPTY>"]
) -> dict[str, int]:
    """
    Create a character-level vocabulary from a set of characters.
    Parameters
    ----------
    char_set : set[str]
        Set of characters to include in the vocabulary.
    specials : list

        List of special tokens to include in the vocabulary.
    Returns
    -------
    vocab : torchtext.vocab.Vocab
        The vocabulary object mapping characters to indices.
    """
    # Sort characters for consistent vocabulary order
    return build_vocab_from_iterator(
        [[c] for c in sorted(list(char_set))], specials=specials, min_freq=1
    )
