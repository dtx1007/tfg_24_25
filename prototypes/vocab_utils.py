import pandas as pd
from torchtext.vocab import Vocab, build_vocab_from_iterator


def create_vocab_for_column(
    df: pd.DataFrame, col: str, specials: list[str] = ["<PAD>", "<UNK>", "<EMPTY>"]
) -> Vocab:
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

    # Build vocabulary from all tokens
    vocab = build_vocab_from_iterator([all_tokens], specials=specials)
    # Set the default index for unknown tokens
    vocab.set_default_index(vocab["<UNK>"])
    return vocab


def create_char_vocab(
    char_set: set[str], specials: list[str] = ["<PAD>", "<UNK>", "<EMPTY>"]
) -> Vocab:
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

    # Create a vocabulary from the character set
    vocab = build_vocab_from_iterator(
        [[c] for c in char_set], specials=specials, min_freq=1
    )
    # Set the default index for unknown characters
    vocab.set_default_index(vocab["<UNK>"])
    return vocab
