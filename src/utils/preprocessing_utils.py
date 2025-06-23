import pandas as pd
import pprint as pp
import numpy as np
import string
import torch

from pathlib import Path

from torchtext.vocab import Vocab
from sklearn.preprocessing import StandardScaler

from .vocab_utils import create_vocab_for_column, create_char_vocab


def _calculate_max_lengths(df, sequence_cols, char_cols):
    max_lengths = {}

    for col in sequence_cols:
        if col in df.columns:
            max_lengths[col] = (
                df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
            )

    for col in char_cols:
        if col in df.columns:
            max_lengths[col] = (
                df[col].apply(lambda x: len(x) if isinstance(x, str) else 0).max()
            )

    return max_lengths


def preprocess_data_for_nn(
    df_raw: pd.DataFrame,
    sequence_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    scalar_cols: list[str],
    vector_dims: dict[str, int],
    vocab_dict: dict[str, Vocab] | None = None,
    max_lengths: dict[str, int] | None = None,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
    empty_token: str = "<EMPTY>",
) -> tuple[pd.DataFrame, dict[str, Vocab]]:
    """
    Comprehensive preprocessing for APK dataset for the APKAnalysisModel.
    Handles:
    1. Parsing string representations of lists and vectors.
    2. Creating vocabularies for sequence and character features if not provided.
    3. Using existing vocabularies to tokenize features.
    4. Converting character strings (e.g., hashes) to lists of indices.
    5. Tokenizing sequence features (lists of strings) to lists of indices.
    6. Padding all sequence and character features to specified max_lengths.
    7. Ensuring scalar features are numeric.
    """
    df = df_raw.copy()

    def parse_list_string(s):
        if isinstance(s, float) and pd.isna(s):
            return []
        if not isinstance(s, str):
            return s
        s = s.strip("[]")
        items = [item.strip().strip("'").strip('"') for item in s.split(",")]
        return [item for item in items if item]

    def parse_vector_string(s, default_dim_key):
        default_dim_val = vector_dims.get(default_dim_key, 256)
        if isinstance(s, float) and pd.isna(s):
            return [0.0] * default_dim_val
        if not isinstance(s, str):
            if isinstance(s, list) and len(s) == default_dim_val:
                return s
            elif isinstance(s, list):
                print(
                    f"Warning: Pre-parsed vector for {default_dim_key} has incorrect length {len(s)}, expected {default_dim_val}. Using default."
                )
                return [0.0] * default_dim_val
            return [0.0] * default_dim_val

        s = s.strip("[]")
        try:
            vector = [float(item.strip()) for item in s.split(",") if item.strip()]
            if len(vector) != default_dim_val:
                print(
                    f"Warning: Parsed vector for {default_dim_key} has length {len(vector)}, expected {default_dim_val}. Adjusting or using default."
                )

                if len(vector) < default_dim_val:
                    vector.extend([0.0] * (default_dim_val - len(vector)))
                else:
                    vector = vector[:default_dim_val]
            return vector
        except ValueError:
            print(
                f"Warning: Failed to parse vector string for {default_dim_key}: {s[:30]}... Using default."
            )
            return [0.0] * default_dim_val

    # 1. Parse list-like strings for SEQUENCE_FEATURES
    for col in sequence_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_string)

    # 2. Parse vector-like strings for VECTOR_COLS
    for col in vector_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_vector_string(x, col))

    if max_lengths is None:
        max_lengths = _calculate_max_lengths(df, sequence_cols, char_cols)
        print("Calculated max lengths for sequence and char columns:")
        pp.pprint(max_lengths)

    # 3. Create vocabularies if not provided
    if vocab_dict is None:
        print("No vocab_dict provided, creating new vocabularies...")
        vocab_dict = {}
        specials = [pad_token, unk_token, empty_token]

        # Vocabs for SEQUENCE_FEATURES
        for col in sequence_cols:
            if col in df.columns:
                vocab_dict[col] = create_vocab_for_column(df, col, specials=specials)
                print(
                    f"Created vocab for sequence column: {col}, size: {len(vocab_dict[col])}"
                )

        # Vocabs for CHAR_COLS
        all_defined_chars = (
            string.ascii_lowercase + string.ascii_uppercase + string.digits + "+/"
        )
        generic_char_vocab = create_char_vocab(
            set(all_defined_chars), specials=specials
        )

        for col in char_cols:
            if col in df.columns:
                vocab_dict[col] = generic_char_vocab
                print(
                    f"Using generic char vocab for char column: {col}, size: {len(vocab_dict[col])}"
                )
    else:
        print("Using provided vocab_dict for tokenization.")

    # 4. Convert CHAR_COLS strings to lists of indices
    def convert_string_to_indices(text_str, vocab, unk_t, empty_t):
        stoi = vocab.get_stoi()
        if isinstance(text_str, str):
            return [stoi.get(char, stoi[unk_t]) for char in text_str]
        return [stoi[empty_t]]

    for col in char_cols:
        if col in df.columns and col in vocab_dict:
            df[col] = df[col].apply(
                lambda x: convert_string_to_indices(
                    x, vocab_dict[col], unk_token, empty_token
                )
            )
        else:
            print(
                f"Warning: Char column {col} not found in DataFrame or vocab_dict for index conversion."
            )

    # 5. Tokenize SEQUENCE_COLS (lists of strings) to lists of indices & PAD
    for col in sequence_cols:
        if col in df.columns and col in vocab_dict:
            stoi = vocab_dict[col].get_stoi()
            pad_idx = stoi.get(pad_token)  # Vocab should guarantee pad_token
            unk_idx = stoi.get(unk_token)
            max_len = max_lengths.get(col)

            if max_len is None:
                raise ValueError(f"max_lengths not provided for sequence column: {col}")
            if pad_idx is None:
                raise ValueError(
                    f"pad_token '{pad_token}' not found in vocab for column: {col}"
                )

            tokenized_and_padded_sequences = []
            for token_list in df[col]:
                indices = [stoi.get(token, unk_idx) for token in token_list]
                padding_needed = max_len - len(indices)
                if padding_needed > 0:
                    padded_indices = indices + [pad_idx] * padding_needed
                else:
                    padded_indices = indices[:max_len]
                tokenized_and_padded_sequences.append(padded_indices)
            df[col] = tokenized_and_padded_sequences
        else:
            print(
                f"Warning: Sequence column {col} not found in DataFrame or vocab_dict for tokenization/padding."
            )

    # 6. Pad CHAR_COLS (already lists of indices)
    for col in char_cols:
        if col in df.columns and col in vocab_dict:
            stoi = vocab_dict[col].get_stoi()
            pad_idx = stoi.get(pad_token)
            max_len = max_lengths.get(col)

            if max_len is None:
                raise ValueError(f"max_lengths not provided for char column: {col}")
            if pad_idx is None:
                raise ValueError(
                    f"pad_token '{pad_token}' not found in vocab for char column: {col}"
                )

            padded_sequences = []
            for indices in df[col]:
                padding_needed = max_len - len(indices)
                if padding_needed > 0:
                    padded_indices = indices + [pad_idx] * padding_needed
                else:
                    padded_indices = indices[:max_len]
                padded_sequences.append(padded_indices)
            df[col] = padded_sequences
        else:
            print(
                f"Warning: Char column {col} not found in DataFrame or vocab_dict for padding."
            )

    # 7. Ensure SCALAR_FEATURES are numeric
    for col in scalar_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, vocab_dict


def apply_scalers_to_dataframe(
    df: pd.DataFrame,
    scalar_cols: list[str],
    vector_cols: list[str],
    scalers: dict[str, StandardScaler] | None = None,
    fit_scalers: bool = False,
) -> tuple[pd.DataFrame, dict[str, StandardScaler]]:
    """
    Apply (and optionally fit) scalers to scalar and vector columns of a DataFrame.
    If fit_scalers is True, fits new scalers on the data and returns them.
    If fit_scalers is False, uses provided scalers to transform the data.
    Returns a new DataFrame with normalized columns and the scalers used.
    """
    df_processed = df.copy()
    scalers_used = scalers.copy() if scalers else {}

    if fit_scalers:
        scalers_used = {}
        for col in scalar_cols:
            if col in df_processed.columns:
                scaler = StandardScaler()
                df_processed[col] = scaler.fit_transform(df_processed[[col]])
                scalers_used[col] = scaler
        for col in vector_cols:
            if col in df_processed.columns:
                vectors = np.array(df_processed[col].tolist(), dtype=np.float32)
                if vectors.ndim == 2:
                    scaler = StandardScaler()
                    scaled_vectors = scaler.fit_transform(vectors)
                    df_processed[col] = [list(vec) for vec in scaled_vectors]
                    scalers_used[col] = scaler
    else:
        if not scalers_used:
            raise ValueError(
                "No scalers provided for normalization (fit_scalers=False)."
            )
        for col in scalar_cols:
            if col in df_processed.columns and col in scalers_used:
                df_processed[col] = scalers_used[col].transform(df_processed[[col]])
        for col in vector_cols:
            if col in df_processed.columns and col in scalers_used:
                vectors = np.array(df_processed[col].tolist(), dtype=np.float32)
                if vectors.ndim == 2:
                    scaled_vectors = scalers_used[col].transform(vectors)
                    df_processed[col] = [list(vec) for vec in scaled_vectors]
    return df_processed, scalers_used


def load_dataset(
    path_to_dataset_dir: Path | str,
    sequence_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    scalar_cols: list[str],
    vector_dims: dict[str, int],
    load_fresh: bool = False,
    sample_size: int | None = None,
):
    """
    Load the dataset, optionally reloading it fresh.
    """
    PATH_TO_DATASET_DIR = Path(path_to_dataset_dir)
    PATH_TO_DATASET = PATH_TO_DATASET_DIR / "apk_analysis_dataset.csv"
    PATH_TO_PROCESSED_DATASET = (
        PATH_TO_DATASET_DIR / "processed_apk_analysis_dataset.pkl"
    )
    PATH_TO_VOCAB_DICT = PATH_TO_DATASET_DIR / "processed_vocab_dict.pth"

    if not PATH_TO_DATASET_DIR.exists():
        print(f"Creating dataset directory at {PATH_TO_DATASET_DIR}")
        PATH_TO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if load_fresh:
        print("Loading dataset fresh...")
        df = pd.read_csv(PATH_TO_DATASET)

        if sample_size is not None:
            print(f"Sampling {sample_size} rows from the dataset...")
            df = df.sample(sample_size, random_state=42)

        df, vocab_dict = preprocess_data_for_nn(
            df,
            sequence_cols=sequence_cols,
            char_cols=char_cols,
            vector_cols=vector_cols,
            scalar_cols=scalar_cols,
            vector_dims=vector_dims,
        )

        print("Saving preprocessed dataset and vocab_dict...")

        df.to_pickle(PATH_TO_PROCESSED_DATASET)
        torch.save(vocab_dict, PATH_TO_VOCAB_DICT)

        print("Preprocessing complete and saved.")

    else:
        print("Loading last preprocessed dataset...")
        df = pd.read_pickle(PATH_TO_PROCESSED_DATASET)
        vocab_dict = torch.load(PATH_TO_VOCAB_DICT)

    return df, vocab_dict
