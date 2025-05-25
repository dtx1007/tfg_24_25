import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

from dataclasses import dataclass, field
from typing import Literal, Self, Any

from torchtext.vocab.vocab import Vocab


@dataclass(frozen=True)
class NNHyperparams:
    """
    Dataclass to hold hyperparameters for the neural network models.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch during training
    learning_rate : float
        Step size for optimizer updates
    epochs : int
        Maximum number of complete passes through the training dataset
    early_stopping : bool
        Whether to stop training when validation loss stops improving
    patience : int
        Number of epochs to wait for improvement before stopping early
    optimizer : str
        Algorithm to use for optimization ('adam', 'adamw', 'sgd')
    weight_decay : float
        L2 regularization parameter for controlling model complexity
    embedding_dim : int
        Dimension of the embedding vectors for categorical features
    hidden_dims : list
        List of hidden layer dimensions in the classifier component
    dropout : float
        Dropout probability for regularization (0.0 to 1.0)
    seq_pooling : str
        Method to pool sequence embeddings ('mean' or 'max')
    n_classes : int
        Number of classes for classification
    label_col : str
        Name of the column containing target labels
    """

    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    early_stopping: bool = True
    patience: int = 5
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = 1e-5
    embedding_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.5
    seq_pooling: Literal["mean", "max"] = "mean"
    n_classes: int = 2
    label_col: str = "malware"


def preprocess_dataframe(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Preprocess dataframe for model input.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be preprocessed
    feature_names : list of str
        List of feature names to be processed

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with features converted to appropriate types
    """
    # Handle missing values
    df = df.replace(np.nan, "")

    # Detect list-based columns (comma-separated strings)
    list_features = []
    for col in feature_names:
        if col.endswith("_list") and col in df.columns:
            list_features.append(col)

    # Convert comma-separated strings to lists for list features
    df = df.assign(
        **{col: df[col].str.split(",") for col in list_features if col in df.columns}
    )
    df = df.assign(
        **{
            col: df[col].apply(lambda x: [f for f in x if f and f.strip() != ""])
            for col in list_features
            if col in df.columns
        }
    )

    # Ensure file_size is numeric
    if "file_size" in df.columns:
        df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce").fillna(0)

    return df


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


class DebrimDataset(Dataset):
    """
    PyTorch Dataset for APK analysis data with mixed input types.
    """

    def __init__(
        self,
        df,
        vocab_dict,
        sequence_cols=None,
        scalar_cols=None,
        char_cols=None,
        vector_cols=None,
        label_col="is_malware",
    ):
        self.df = df
        self.vocab_dict = vocab_dict
        self.sequence_cols = sequence_cols or []  # Regular sequence features
        self.scalar_cols = scalar_cols or []  # Numeric scalar features
        self.char_cols = char_cols or []  # Character-level features
        self.vector_cols = vector_cols or []  # Fixed-length vector features
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Process standard sequence features (explicitly using sequence_cols)
        seq_features = {}
        for col in self.sequence_cols:
            if col in row.index:
                stoi = self.vocab_dict[col].get_stoi()
                indices = [stoi.get(token, stoi["<UNK>"]) for token in row[col]]
                seq_features[col] = torch.tensor(indices, dtype=torch.long)

        # Process character-level features
        char_features = {
            col: torch.tensor(row[col], dtype=torch.long)
            for col in self.char_cols
            if col in row.index
        }

        # Process vector features (no type checks or filtering necessary)
        vector_features = {
            col: torch.tensor(row[col], dtype=torch.float)
            for col in self.vector_cols
            if col in row.index
        }

        # Process scalar features (no type checks or filtering necessary)
        scalars = {
            col: torch.tensor(row[col], dtype=torch.float)
            for col in self.scalar_cols
            if col in row.index
        }

        # Convert label
        if self.label_col in row.index:
            label = torch.tensor(row[self.label_col], dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)

        return seq_features, char_features, vector_features, scalars, label


def collate_fn(batch):
    """
    Custom collate function for batching different types of features.

    Parameters
    ----------
    batch : list
        List of tuples (seq_features, char_features, vector_features, scalars, label)

    Returns
    -------
    tuple
        Tuple containing batched features
    """
    # Unzip the batch into separate components
    seq_features, char_features, vector_features, scalars, labels = zip(*batch)

    # Pad sequences to the same length within the batch
    padded_seq_features = {}
    for col in seq_features[0].keys():
        sequences = [f[col] for f in seq_features if col in f]
        if sequences:
            padded_seq_features[col] = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True
            )

    # Pad character sequences to the same length
    padded_char_features = {}
    for col in char_features[0].keys():
        char_seqs = [f[col] for f in char_features if col in f]
        if char_seqs:
            padded_char_features[col] = torch.nn.utils.rnn.pad_sequence(
                char_seqs, batch_first=True
            )

    # Stack vector features
    stacked_vector_features = {}
    for col in vector_features[0].keys():
        vectors = [f[col] for f in vector_features if col in f]
        if vectors:
            stacked_vector_features[col] = torch.stack(vectors)

    # Stack scalar features and labels
    stacked_scalars = {}
    for col in scalars[0].keys():
        stacked_scalars[col] = torch.stack([s[col] for s in scalars if col in s])

    stacked_labels = torch.stack(labels)

    return (
        padded_seq_features,
        padded_char_features,
        stacked_vector_features,
        stacked_scalars,
        stacked_labels,
    )


class DebrimEmbedder(nn.Module):
    """Feature embedding module for the APK analysis model."""

    def __init__(
        self,
        vocab_dict,
        sequence_cols=None,
        embedding_dim=128,
        seq_pooling="mean",
        char_cols=None,
        vector_cols=None,
        vector_dims=None,
    ):
        super().__init__()
        self.vocab_dict = vocab_dict
        self.embedding_dim = embedding_dim
        self.seq_pooling = seq_pooling
        self.sequence_cols = sequence_cols or []  # Store sequence column names
        self.char_cols = char_cols or []
        self.vector_cols = vector_cols or []
        self.vector_dims = vector_dims or {}

        # Create embedding layer for each sequence feature
        self.seq_embedders = nn.ModuleDict(
            {
                col: nn.Embedding(
                    num_embeddings=len(vocab_dict[col]),
                    embedding_dim=embedding_dim,
                    padding_idx=vocab_dict[col]["<PAD>"],
                )
                for col in self.sequence_cols
                if col in vocab_dict
            }
        )
        # self.seq_embedders = nn.ModuleDict(
        #     {
        #         col: nn.EmbeddingBag(
        #             num_embeddings=len(vocab_dict[col]),
        #             embedding_dim=embedding_dim,
        #             mode="mean" if seq_pooling == "mean" else "max",
        #             padding_idx=vocab_dict[col]["<PAD>"],
        #         )
        #         for col in self.sequence_cols
        #         if col in vocab_dict
        #     }
        # )

        # Character embeddings and GRU
        if any(col in self.char_cols for col in vocab_dict):
            self.char_embedders = nn.ModuleDict(
                {
                    col: nn.Embedding(
                        num_embeddings=len(vocab_dict[col]),
                        embedding_dim=embedding_dim // 2,
                        padding_idx=vocab_dict[col]["<PAD>"],
                    )
                    for col in self.char_cols
                    if col in vocab_dict
                }
            )

            self.char_gru = nn.ModuleDict(
                {
                    col: nn.GRU(
                        input_size=embedding_dim // 2,
                        hidden_size=embedding_dim,
                        batch_first=True,
                        bidirectional=False,
                    )
                    for col in self.char_cols
                }
            )

        # Vector reducers
        self.vector_reducers = nn.ModuleDict(
            {
                col: nn.Sequential(
                    nn.Linear(self.vector_dims.get(col, 256), embedding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(),
                )
                for col in self.vector_cols
            }
        )

        # Calculate total output dimension - now cleaner with explicit column lists
        self.output_dim = (
            len(self.sequence_cols) * embedding_dim
            + len(self.char_cols) * embedding_dim
            + len(self.vector_cols) * embedding_dim
        )

    def forward(
        self,
        seq_feats: dict[str, torch.Tensor],
        char_feats: dict[str, torch.Tensor],
        vector_feats: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Process and embed all feature types."""
        pooled_embeddings = []

        # Process sequence features - simplify by only iterating over known sequence columns
        for feature_name, sequence in seq_feats.items():
            if feature_name in self.sequence_cols:
                embeddings = self.seq_embedders[feature_name](sequence)
                padding_mask = (sequence != 0).unsqueeze(-1)

                if self.seq_pooling == "mean":
                    summed = (embeddings * padding_mask).sum(dim=1)
                    token_counts = padding_mask.sum(dim=1).clamp(min=1)
                    pooled = summed / token_counts
                else:
                    masked_embeddings = embeddings.masked_fill(
                        ~padding_mask.bool(), float("-inf")
                    )
                    pooled = masked_embeddings.max(dim=1).values

                pooled_embeddings.append(pooled)

        # Process sequence features with EmbeddingBag (modified)
        # for feature_name, sequence in seq_feats.items():
        #     if (
        #         feature_name in self.sequence_cols
        #         and feature_name in self.seq_embedders
        #     ):
        #         # Create offsets tensor for EmbeddingBag
        #         # Each sequence gets processed as a whole unit
        #         batch_size = sequence.size(0)

        #         # Filter out padding tokens (0) for more efficient processing
        #         mask = sequence != 0
        #         indices = torch.masked_select(sequence, mask)

        #         # Create offsets tensor - marks the start of each sequence
        #         # Each sequence will be processed together by EmbeddingBag
        #         offsets = torch.zeros(
        #             batch_size, dtype=torch.long, device=sequence.device
        #         )

        #         # Calculate offsets based on cumulative sum of valid tokens per sequence
        #         if batch_size > 1:
        #             seq_lengths = mask.sum(dim=1)
        #             offsets[1:] = torch.cumsum(seq_lengths[:-1], dim=0)

        #         # Use EmbeddingBag to get pooled embeddings directly
        #         pooled = self.seq_embedders[feature_name](indices, offsets)
        #         pooled_embeddings.append(pooled)

        # Process character features - simplify by only iterating over known char columns
        for feature_name, char_sequence in char_feats.items():
            if feature_name in self.char_cols:
                # Use batch_first=True (already in your code)
                char_embeddings = self.char_embedders[feature_name](char_sequence)

                # Pack padded sequence for more efficient GRU processing
                lengths = lengths = char_sequence.ne(0).sum(1).to("cpu")
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    char_embeddings, lengths, batch_first=True, enforce_sorted=False
                )

                # Process through GRU
                _, hidden = self.char_gru[feature_name](packed)
                pooled_embeddings.append(hidden.squeeze(0))

        # Process vector features - simplify by only iterating over known vector columns
        for feature_name, vector in vector_feats.items():
            if feature_name in self.vector_cols:
                reduced = self.vector_reducers[feature_name](vector)
                pooled_embeddings.append(reduced)

        # Rest of the method remains the same
        if pooled_embeddings:
            concatenated_embeddings = torch.cat(pooled_embeddings, dim=1)
        else:
            concatenated_embeddings = torch.zeros(
                (1, self.output_dim), device=next(self.parameters()).device
            )

        return concatenated_embeddings, scalars


class DebrimClassifier(nn.Module):
    """
    MLP classifier head for the Debrim model.

    A multi-layer perceptron (MLP) classifier that processes the embedded features
    and produces class probabilities for malware detection.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (sum of embedding dimensions)
    hidden_dims : list of int
        List of hidden layer dimensions in order
    n_classes : int
        Number of output classes (typically 2 for binary classification)
    dropout : float
        Dropout probability for regularization (0.0 to 1.0)
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64],
        n_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        # Construct MLP layers with activation and dropout
        layers = []
        layer_dimensions = [input_dim] + hidden_dims

        # Create hidden layers with ReLU activation and dropout
        for input_dim, output_dim in zip(layer_dimensions, layer_dimensions[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Add final output layer (no activation - will be applied in loss function)
        layers.append(nn.Linear(layer_dimensions[-1], n_classes))

        # Combine all layers into a sequential model
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input features through the MLP classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor of shape [batch_size, input_dim]

        Returns
        -------
        torch.Tensor
            Logits tensor of shape [batch_size, n_classes]
        """
        return self.mlp(x)


class DebrimModel(nn.Module):
    """
    Complete APK Analysis model combining embedder and classifier components.

    This model integrates feature embedding and classification components into
    a complete end-to-end architecture for malware detection.

    Parameters
    ----------
    embedder : DebrimEmbedder
        The embedder component that processes different feature types
    classifier : DebrimClassifier, optional
        The classifier component that processes embedded features
    scalar_dim : int
        Number of scalar features to include in the model
    """

    def __init__(
        self,
        embedder,
        classifier=None,
        scalar_dim=0,
    ):
        super().__init__()

        self.embedder = embedder
        self.scalar_dim = scalar_dim

        # If no classifier is provided, create a default one with standard parameters
        if classifier is None:
            total_input_dim = embedder.output_dim + scalar_dim
            self.classifier = DebrimClassifier(
                input_dim=total_input_dim,
                hidden_dims=[128, 64],
                n_classes=2,
                dropout=0.5,
            )
        else:
            self.classifier = classifier

    @classmethod
    def from_config(
        cls,
        vocab_dict: dict[str, Vocab],
        sequence_cols: list[str] = None,
        scalar_cols: list[str] = None,
        char_cols: list[str] = None,
        vector_cols: list[str] = None,
        vector_dims: dict[str, int] = None,
        embedding_dim: int = 128,
        seq_pooling: Literal["mean", "max"] = "mean",
        hidden_dim: list[int] = [128, 64],
        n_classes: int = 2,
        dropout: float = 0.5,
    ) -> Self:
        """Create a model from configuration parameters."""
        sequence_cols = sequence_cols or []
        scalar_cols = scalar_cols or []
        char_cols = char_cols or []
        vector_cols = vector_cols or []
        vector_dims = vector_dims or {}

        embedder = DebrimEmbedder(
            vocab_dict=vocab_dict,
            sequence_cols=sequence_cols,
            embedding_dim=embedding_dim,
            seq_pooling=seq_pooling,
            char_cols=char_cols,
            vector_cols=vector_cols,
            vector_dims=vector_dims,
        )

        # Use the output dimension from the created embedder
        total_input_dim = embedder.output_dim + len(scalar_cols)

        classifier = DebrimClassifier(
            input_dim=total_input_dim,
            hidden_dims=hidden_dim,
            n_classes=n_classes,
            dropout=dropout,
        )

        return cls(
            embedder=embedder, classifier=classifier, scalar_dim=len(scalar_cols)
        )

    def forward(self, seq_feats, char_feats, vector_feats, scalars):
        """
        Forward pass through the complete model.

        Parameters
        ----------
        seq_feats : dict
            Dictionary mapping feature names to sequence tensors
        char_feats : dict
            Dictionary mapping feature names to character-level tensors
        vector_feats : dict
            Dictionary mapping feature names to vector features
        scalars : dict
            Dictionary mapping feature names to scalar tensors

        Returns
        -------
        torch.Tensor
            Logits tensor of shape [batch_size, n_classes]
        """
        # Get embeddings from embedder component
        embeddings, _ = self.embedder(seq_feats, char_feats, vector_feats)

        # Process scalar features if present
        if self.scalar_dim > 0 and scalars:
            # Convert scalar features dictionary to a list of tensors
            scalar_tensors = [scalars[col] for col in scalars]

            if scalar_tensors:
                # Concatenate scalar features into a single tensor
                scalar_tensor = (
                    torch.cat(scalar_tensors, dim=1)
                    if len(scalar_tensors) > 1
                    else scalar_tensors[0].unsqueeze(1)
                )

                # Concatenate with embeddings
                combined_features = torch.cat([embeddings, scalar_tensor], dim=1)
            else:
                combined_features = embeddings
        else:
            combined_features = embeddings

        # Pass through classifier to get output logits
        return self.classifier(combined_features)


def get_best_available_device() -> torch.device:
    """
    Detect and return the best available device for PyTorch computation.

    This function checks for hardware acceleration in the following order:
    1. NVIDIA CUDA GPUs
    2. Apple Metal Performance Shaders (M-series chips)
    3. CPU (fallback)

    Returns
    -------
    torch.device
        The best available computation device for PyTorch
    """
    import torch
    import platform

    # Check for NVIDIA GPU with CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        return device

    # Check for Apple Silicon with Metal Performance Shaders (MPS)
    # Available on macOS 12.3+ with M-series chips
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        mac_model = platform.machine()
        print(f"Using Metal device on {mac_model}")
        return device

    # Fall back to CPU if no accelerated hardware is available
    device = torch.device("cpu")
    print("GPU acceleration not available. Using CPU.")
    return device


# Training function for neural network models
def train_nn_model(
    df: pd.DataFrame,
    vocab_dict: dict[str, Vocab],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    vector_dims: dict[str, int],
    hyperparams: NNHyperparams = NNHyperparams(),
    device: torch.device | None = None,
) -> tuple[DebrimModel, list[float]]:
    """
    Train a neural network model on the given dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing features and labels
    vocab_dict : dict
        Dictionary mapping feature names to vocabulary objects
    scalar_cols : list
        List of scalar columns to include as numerical features
    char_cols : list
        List of columns to be processed character by character
    vector_cols : list
        List of columns containing fixed-length vectors
    vector_dims : dict
        Dictionary specifying the dimension of each vector feature
    hyperparams : NNHyperparams
        Dataclass containing all hyperparameters for model training
    device : torch.device or None
        Device to run the training on (defaults to best available device)

    Returns
    -------
    tuple
        (trained_model, training_losses)
    """
    # Create dataset and data loader for training
    dataset = DebrimDataset(
        df,
        vocab_dict,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=hyperparams.label_col,
    )

    data_loader = DataLoader(
        dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialize model with configuration from hyperparameters
    model = DebrimModel.from_config(
        vocab_dict,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        vector_dims=vector_dims,
        embedding_dim=hyperparams.embedding_dim,
        hidden_dim=hyperparams.hidden_dims,
        seq_pooling=hyperparams.seq_pooling,
        n_classes=hyperparams.n_classes,
        dropout=hyperparams.dropout,
    )

    # Configure optimizer based on hyperparameters
    match hyperparams.optimizer:
        case "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        case "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        case "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        case _:
            # Default to Adam if invalid optimizer specified
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )

    # Loss function for training
    criterion = nn.CrossEntropyLoss()

    # Select computation device
    if device is None:
        device = get_best_available_device()

    print(f"Using device: {device}")
    model = model.to(device)

    # Training loop
    model.train()
    training_losses = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(hyperparams.epochs):
        epoch_loss = 0.0

        # Process batches
        for seq_feats, char_feats, vector_feats, scalars, labels in data_loader:
            # Move batch data to the selected device
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(seq_feats, char_feats, vector_feats, scalars)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(data_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{hyperparams.epochs} — Loss: {avg_loss:.4f}")

        # Early stopping logic
        if hyperparams.early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= hyperparams.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return model, training_losses


def cross_val_train_nn_model(
    df: pd.DataFrame,
    vocab_dict: dict[str, Vocab],
    sequence_cols: list[str],  # Added this parameter
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    vector_dims: dict[str, int],
    hyperparams: NNHyperparams = NNHyperparams(),
    n_folds: int = 2,
    n_repetitions: int = 5,
    scoring_metric: Literal[
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ] = "f1",
    device: torch.device | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, Any], DebrimModel]:
    """
    Train a neural network model with cross-validation using the provided hyperparameters

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing features and labels
    vocab_dict : dict
        Dictionary of vocabularies for each feature column
    scalar_cols : list
        List of scalar columns to include
    hyperparams : NNHyperparams
        Dataclass containing hyperparameters for model training
    n_folds : int
        Number of folds for cross-validation
    n_repetitions : int
        Number of repetition rounds for cross-validation
    scoring_metric : str
        Primary metric to use for model evaluation and selection
        Options: "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    device : torch.device or None
        Device to use (if None, will use best available device)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (results_dict, best_model)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        auc,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
    from torch.utils.data import DataLoader
    import tempfile
    import os
    import time

    # Check if GPU acceleration is available
    if device is None:
        device = get_best_available_device()

    print(f"Using device: {device}")
    print(f"Using {scoring_metric} as the primary scoring metric")

    # Prepare for repeated stratified k-fold
    X = np.arange(len(df))  # Just indices, actual features handled by Dataset
    y = df[hyperparams.label_col].values

    # Compute class weights for imbalanced data
    expected_classes = np.array([0, 1])  # For binary classification
    if len(np.unique(y)) < len(expected_classes):
        print(
            "Warning: Not all classes present in dataset. Using uniform class weights."
        )
        class_weights = np.ones(len(expected_classes))
    else:
        class_weights = compute_class_weight(
            class_weight="balanced", classes=expected_classes, y=y
        )

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Using class weights: {class_weights}")

    # Setup cross-validation
    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repetitions, random_state=random_seed
    )

    # Convert hyperparams to dict for storing in results
    hyperparams_dict = {
        "batch_size": hyperparams.batch_size,
        "learning_rate": hyperparams.learning_rate,
        "epochs": hyperparams.epochs,
        "early_stopping": hyperparams.early_stopping,
        "patience": hyperparams.patience,
        "optimizer": hyperparams.optimizer,
        "weight_decay": hyperparams.weight_decay,
        "embedding_dim": hyperparams.embedding_dim,
        "hidden_dim": hyperparams.hidden_dims,
        "dropout": hyperparams.dropout,
        "seq_pooling": hyperparams.seq_pooling,
        "n_classes": hyperparams.n_classes,
        "label_col": hyperparams.label_col,
    }

    # Initialize results storage
    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "pr_auc": [],
        "roc_auc": [],
        "pr_curve_data": [],
        "roc_curve_data": [],
        "conf_matrices": [],
        "model_size": [],
        "train_losses": [],
        "val_losses": [],
        "training_time": [],
        "fold_info": [],
        "hyperparams": hyperparams_dict,
    }

    # Track the best model across all folds
    best_model = None
    best_score = -1  # Track best score for the selected metric
    best_model_state = None
    best_fold_idx = -1

    fold_count = 1
    # Iterate through repetitions and folds
    for rep_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        # Calculate current repetition and fold
        current_rep = (rep_idx // n_folds) + 1
        current_fold = (rep_idx % n_folds) + 1

        print(
            f"\n=== Repetition {current_rep}/{n_repetitions}, Fold {current_fold}/{n_folds} ==="
        )

        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Check class distribution in splits
        print(
            f"Training set class distribution: {train_df[hyperparams.label_col].value_counts().to_dict()}"
        )
        print(
            f"Validation set class distribution: {val_df[hyperparams.label_col].value_counts().to_dict()}"
        )

        # Create datasets and loaders
        train_dataset = DebrimDataset(
            train_df,
            vocab_dict=vocab_dict,
            sequence_cols=sequence_cols,
            char_cols=char_cols,
            vector_cols=vector_cols,
            scalar_cols=scalar_cols,
            label_col=hyperparams.label_col,
        )
        val_dataset = DebrimDataset(
            val_df,
            vocab_dict=vocab_dict,
            sequence_cols=sequence_cols,
            char_cols=char_cols,
            vector_cols=vector_cols,
            scalar_cols=scalar_cols,
            label_col=hyperparams.label_col,
        )

        print("Created datasets for training and validation")

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
        )

        print("Created loaders for training and validation")

        # Instantiate model with hyperparameters
        model = DebrimModel.from_config(
            vocab_dict,
            scalar_cols=scalar_cols,
            sequence_cols=sequence_cols,
            char_cols=char_cols,
            vector_cols=vector_cols,
            vector_dims=vector_dims,
            embedding_dim=hyperparams.embedding_dim,
            seq_pooling=hyperparams.seq_pooling,
            hidden_dim=hyperparams.hidden_dims,
            n_classes=hyperparams.n_classes,
            dropout=hyperparams.dropout,
        ).to(device)

        print("Initialized model with specified configuration")

        # Optimizer based on hyperparameters
        if hyperparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        elif hyperparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        elif hyperparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )
        else:
            # Default to Adam if invalid optimizer specified
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay,
            )

        criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        # Training trackers
        best_val_loss = float("inf")
        current_best_model_state = None
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)

        print("Starting training...")

        # Measure training time
        start_time = time.time()

        # Training loop
        for epoch in range(hyperparams.epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            batch_idx = 0

            for seq_feats, char_feats, vector_feats, scalars, label in train_loader:
                seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
                char_feats = {k: v.to(device) for k, v in char_feats.items()}
                vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
                scalars = {k: v.to(device) for k, v in scalars.items()}
                label = label.to(device)

                # with profile(
                #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #     profile_memory=True,
                #     record_shapes=True,
                # ) as prof:
                #     with record_function("model_training"):
                #         # Forward pass
                #         optimizer.zero_grad()
                #         logits = model(seq_feats, char_feats, vector_feats, scalars)
                #         loss = criterion(logits, label)
                #         # Backward
                #         loss.backward()
                #         optimizer.step()

                # print(
                #     prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                # )

                optimizer.zero_grad()
                logits = model(seq_feats, char_feats, vector_feats, scalars)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                if batch_idx % 2 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

                epoch_train_loss += loss.item()
                batch_idx += 1

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            print(
                f"Epoch {epoch + 1}/{hyperparams.epochs} — Train Loss: {avg_train_loss:.4f}"
            )

            # Validation phase
            model.eval()
            epoch_val_loss = 0
            y_true, y_pred, y_prob = [], [], []

            with torch.no_grad():
                for seq_feats, char_feats, vector_feats, scalars, label in val_loader:
                    seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
                    char_feats = {k: v.to(device) for k, v in char_feats.items()}
                    vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
                    scalars = {k: v.to(device) for k, v in scalars.items()}
                    label = label.to(device)

                    logits = model(seq_feats, char_feats, vector_feats, scalars)
                    loss = criterion(logits, label)
                    epoch_val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    probs = F.softmax(logits, dim=1)[
                        :, 1
                    ]  # Probability of positive class

                    y_true.extend(label.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())
                    y_prob.extend(probs.cpu().tolist())

            print(
                f"Epoch {epoch + 1}/{hyperparams.epochs} — Val Loss: {epoch_val_loss / len(val_loader):.4f}"
            )

            # Calculate metrics
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Choose metric to display based on selected scoring metric
            if scoring_metric == "f1":
                score = f1_score(y_true, y_pred, average="weighted")
            elif scoring_metric == "precision":
                score = precision_score(y_true, y_pred, average="weighted")
            elif scoring_metric == "recall":
                score = recall_score(y_true, y_pred, average="weighted")
            elif scoring_metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif scoring_metric == "roc_auc":
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                score = auc(fpr, tpr)
            elif scoring_metric == "pr_auc":
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                score = auc(recall, precision)
            else:
                score = f1_score(y_true, y_pred, average="weighted")  # Default to F1

            # Print progress
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}, {scoring_metric} = {score:.4f}"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                current_best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if hyperparams.early_stopping and epochs_no_improve >= hyperparams.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Training time
        training_time = time.time() - start_time

        # Load best model for final evaluation
        if current_best_model_state is not None:
            model.load_state_dict(current_best_model_state)

        # Final evaluation metrics
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for seq_feats, char_feats, vector_feats, scalars, label in val_loader:
                seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
                char_feats = {k: v.to(device) for k, v in char_feats.items()}
                vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
                scalars = {k: v.to(device) for k, v in scalars.items()}
                label = label.to(device)

                logits = model(seq_feats, char_feats, vector_feats, scalars)
                preds = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1)[:, 1]

                y_true.extend(label.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        # Calculate final metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_score = auc(pr_recall, pr_precision)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_score = auc(fpr, tpr)

        cm = confusion_matrix(y_true, y_pred)

        # Get model size
        fd, path = tempfile.mkstemp()
        os.close(fd)
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path) / 1024  # KB
        os.remove(path)

        # Determine current score based on chosen metric
        current_score = None
        if scoring_metric == "f1":
            current_score = f1
        elif scoring_metric == "precision":
            current_score = precision
        elif scoring_metric == "recall":
            current_score = recall
        elif scoring_metric == "accuracy":
            current_score = accuracy
        elif scoring_metric == "roc_auc":
            current_score = roc_auc_score
        elif scoring_metric == "pr_auc":
            current_score = pr_auc_score
        else:
            current_score = f1  # Default to F1 if invalid metric

        # Check if this is the best model so far based on selected metric
        if current_score > best_score:
            best_score = current_score
            best_model_state = current_best_model_state
            best_fold_idx = fold_count
            print(f"  * New best model: {scoring_metric.upper()}={current_score:.4f}")

        # Save results
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["pr_auc"].append(pr_auc_score)
        results["roc_auc"].append(roc_auc_score)
        results["pr_curve_data"].append((pr_precision, pr_recall))
        results["roc_curve_data"].append((fpr, tpr))
        results["conf_matrices"].append(cm)
        results["model_size"].append(size)
        results["train_losses"].append(train_losses)
        results["val_losses"].append(val_losses)
        results["training_time"].append(training_time)
        results["fold_info"].append(
            {
                "repetition": current_rep,
                "fold": current_fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
            }
        )

        print(
            f"Fold {fold_count} results: F1={f1:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, PR-AUC={pr_auc_score:.4f}, ROC-AUC={roc_auc_score:.4f}"
        )
        print(f"Training time: {training_time:.2f} seconds")
        fold_count += 1

    # Calculate means and standard deviations
    print("\n=== Overall Results ===")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "model_size",
        "training_time",
    ]:
        results[f"mean_{metric}"] = np.mean(results[metric], dtype=np.float64)
        results[f"std_{metric}"] = np.std(results[metric], dtype=np.float64)

        print(
            f"Mean {metric}: {results[f'mean_{metric}']:.4f} ± {results[f'std_{metric}']:.4f}"
        )

    results["best_fold_idx"] = best_fold_idx
    results["best_score"] = best_score
    results["scoring_metric"] = scoring_metric

    print(
        f"Best model from fold {best_fold_idx} with {scoring_metric}={best_score:.4f}"
    )

    # Initialize best model with best weights
    best_model = DebrimModel.from_config(
        vocab_dict,
        scalar_cols=scalar_cols,
        sequence_cols=sequence_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        vector_dims=vector_dims,
        embedding_dim=hyperparams.embedding_dim,
        seq_pooling=hyperparams.seq_pooling,
        hidden_dim=hyperparams.hidden_dims,
        n_classes=hyperparams.n_classes,
        dropout=hyperparams.dropout,
    )
    best_model.load_state_dict(best_model_state)

    return results, best_model


# def cross_val_train_nn_model(
#     df: pd.DataFrame,
#     vocab_dict: dict[str, Vocab],
#     scalar_cols: list[str],
#     char_cols: list[str],
#     vector_cols: list[str],
#     vector_dims: dict[str, int],
#     hyperparams: NNHyperparams = NNHyperparams(),
#     n_folds: int = 2,
#     n_repetitions: int = 5,
#     scoring_metric: Literal[
#         "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
#     ] = "f1",
#     device: torch.device | None = None,
#     random_seed: int = 42,
# ) -> tuple[dict[str, Any], DebrimModel]:
#     """
#     Train a neural network model with cross-validation using the provided hyperparameters

#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         The dataset containing features and labels
#     vocab_dict : dict
#         Dictionary of vocabularies for each feature column
#     scalar_cols : list
#         List of scalar columns to include
#     char_cols : list
#         List of columns to be processed character by character
#     vector_cols : list
#         List of columns containing fixed-length vectors
#     vector_dims : dict
#         Dictionary specifying the dimension of each vector feature
#     hyperparams : NNHyperparams
#         Dataclass containing hyperparameters for model training
#     n_folds : int
#         Number of folds for cross-validation
#     n_repetitions : int
#         Number of repetition rounds for cross-validation
#     scoring_metric : str
#         Primary metric to use for model evaluation and selection
#         Options: "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
#     device : torch.device or None
#         Device to use (if None, will use best available device)
#     random_seed : int
#         Random seed for reproducibility

#     Returns:
#     --------
#     tuple
#         (results_dict, best_model)
#     """
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from sklearn.model_selection import RepeatedStratifiedKFold
#     from sklearn.utils.class_weight import compute_class_weight
#     from sklearn.metrics import (
#         accuracy_score,
#         f1_score,
#         precision_score,
#         recall_score,
#         auc,
#         confusion_matrix,
#         precision_recall_curve,
#         roc_curve,
#     )
#     from torch.utils.data import DataLoader
#     import tempfile
#     import os
#     import time

#     # Check if GPU acceleration is available
#     if device is None:
#         device = get_best_available_device()

#     print(f"Using device: {device}")
#     print(f"Using {scoring_metric} as the primary scoring metric")

#     # Prepare for repeated stratified k-fold
#     X = np.arange(len(df))  # Just indices, actual features handled by Dataset
#     y = df[hyperparams.label_col].values

#     # Compute class weights for imbalanced data
#     unique_classes = np.unique(y)
#     class_weights = compute_class_weight(
#         class_weight="balanced", classes=unique_classes, y=y
#     )
#     weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

#     # Setup cross-validation
#     rskf = RepeatedStratifiedKFold(
#         n_splits=n_folds, n_repeats=n_repetitions, random_state=random_seed
#     )

#     # Convert hyperparams to dict for storing in results
#     hyperparams_dict = {
#         "batch_size": hyperparams.batch_size,
#         "learning_rate": hyperparams.learning_rate,
#         "epochs": hyperparams.epochs,
#         "early_stopping": hyperparams.early_stopping,
#         "patience": hyperparams.patience,
#         "optimizer": hyperparams.optimizer,
#         "weight_decay": hyperparams.weight_decay,
#         "embedding_dim": hyperparams.embedding_dim,
#         "hidden_dim": hyperparams.hidden_dims,
#         "dropout": hyperparams.dropout,
#         "seq_pooling": hyperparams.seq_pooling,
#         "n_classes": hyperparams.n_classes,
#         "label_col": hyperparams.label_col,
#     }

#     # Initialize results storage
#     results = {
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "pr_auc": [],
#         "roc_auc": [],
#         "pr_curve_data": [],
#         "roc_curve_data": [],
#         "conf_matrices": [],
#         "model_size": [],
#         "train_losses": [],
#         "val_losses": [],
#         "training_time": [],
#         "fold_info": [],
#         "hyperparams": hyperparams_dict,
#     }

#     # Track the best model across all folds
#     best_model = None
#     best_score = -1  # Track best score for the selected metric
#     best_model_state = None
#     best_fold_idx = -1

#     fold_count = 1
#     # Iterate through repetitions and folds
#     for rep_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
#         # Calculate current repetition and fold
#         current_rep = (rep_idx // n_folds) + 1
#         current_fold = (rep_idx % n_folds) + 1

#         print(
#             f"\n=== Repetition {current_rep}/{n_repetitions}, Fold {current_fold}/{n_folds} ==="
#         )

#         # Split data
#         train_df = df.iloc[train_idx].reset_index(drop=True)
#         val_df = df.iloc[val_idx].reset_index(drop=True)

#         # Check class distribution in splits
#         print(
#             f"Training set class distribution: {train_df[hyperparams.label_col].value_counts().to_dict()}"
#         )
#         print(
#             f"Validation set class distribution: {val_df[hyperparams.label_col].value_counts().to_dict()}"
#         )

#         # Create datasets and loaders
#         train_dataset = DebrimDataset(
#             train_df,
#             vocab_dict,
#             scalar_cols=scalar_cols,
#             char_cols=char_cols,
#             vector_cols=vector_cols,
#             label_col=hyperparams.label_col,
#         )
#         val_dataset = DebrimDataset(
#             val_df,
#             vocab_dict,
#             scalar_cols=scalar_cols,
#             char_cols=char_cols,
#             vector_cols=vector_cols,
#             label_col=hyperparams.label_col,
#         )

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=hyperparams.batch_size,
#             shuffle=True,
#             collate_fn=collate_fn,
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=hyperparams.batch_size,
#             shuffle=False,
#             collate_fn=collate_fn,
#         )

#         # Instantiate model with hyperparameters
#         model = DebrimModel.from_config(
#             vocab_dict,
#             scalar_cols=scalar_cols,
#             char_cols=char_cols,
#             vector_cols=vector_cols,
#             vector_dims=vector_dims,
#             embedding_dim=hyperparams.embedding_dim,
#             seq_pooling=hyperparams.seq_pooling,
#             hidden_dim=hyperparams.hidden_dims,
#             n_classes=hyperparams.n_classes,
#             dropout=hyperparams.dropout,
#         ).to(device)

#         # Optimizer based on hyperparameters
#         if hyperparams.optimizer == "adam":
#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=hyperparams.learning_rate,
#                 weight_decay=hyperparams.weight_decay,
#             )
#         elif hyperparams.optimizer == "adamw":
#             optimizer = torch.optim.AdamW(
#                 model.parameters(),
#                 lr=hyperparams.learning_rate,
#                 weight_decay=hyperparams.weight_decay,
#             )
#         elif hyperparams.optimizer == "sgd":
#             optimizer = torch.optim.SGD(
#                 model.parameters(),
#                 lr=hyperparams.learning_rate,
#                 weight_decay=hyperparams.weight_decay,
#             )
#         else:
#             # Default to Adam if invalid optimizer specified
#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=hyperparams.learning_rate,
#                 weight_decay=hyperparams.weight_decay,
#             )

#         criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

#         # Training trackers
#         best_val_loss = float("inf")
#         current_best_model_state = None
#         epochs_no_improve = 0
#         train_losses, val_losses = [], []

#         # Measure training time
#         start_time = time.time()

#         # Training loop
#         for epoch in range(hyperparams.epochs):
#             # Training phase
#             model.train()
#             epoch_train_loss = 0

#             for seq_feats, char_feats, vector_feats, scalars, label in train_loader:
#                 # Move batch data to device
#                 seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
#                 char_feats = {k: v.to(device) for k, v in char_feats.items()}
#                 vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
#                 scalars = {k: v.to(device) for k, v in scalars.items()}
#                 label = label.to(device)

#                 optimizer.zero_grad()
#                 logits = model(seq_feats, char_feats, vector_feats, scalars)
#                 loss = criterion(logits, label)
#                 loss.backward()
#                 optimizer.step()

#                 epoch_train_loss += loss.item()

#             avg_train_loss = epoch_train_loss / len(train_loader)
#             train_losses.append(avg_train_loss)

#             # Validation phase
#             model.eval()
#             epoch_val_loss = 0
#             y_true, y_pred, y_prob = [], [], []

#             with torch.no_grad():
#                 for seq_feats, char_feats, vector_feats, scalars, label in val_loader:
#                     # Move batch data to device
#                     seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
#                     char_feats = {k: v.to(device) for k, v in char_feats.items()}
#                     vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
#                     scalars = {k: v.to(device) for k, v in scalars.items()}
#                     label = label.to(device)

#                     logits = model(seq_feats, char_feats, vector_feats, scalars)
#                     loss = criterion(logits, label)
#                     epoch_val_loss += loss.item()

#                     preds = torch.argmax(logits, dim=1)
#                     probs = F.softmax(logits, dim=1)[
#                         :, 1
#                     ]  # Probability of positive class

#                     y_true.extend(label.cpu().tolist())
#                     y_pred.extend(preds.cpu().tolist())
#                     y_prob.extend(probs.cpu().tolist())

#             # Calculate metrics
#             avg_val_loss = epoch_val_loss / len(val_loader)
#             val_losses.append(avg_val_loss)

#             # Choose metric to display based on selected scoring metric
#             if scoring_metric == "f1":
#                 score = f1_score(y_true, y_pred, average="weighted")
#             elif scoring_metric == "precision":
#                 score = precision_score(y_true, y_pred, average="weighted")
#             elif scoring_metric == "recall":
#                 score = recall_score(y_true, y_pred, average="weighted")
#             elif scoring_metric == "accuracy":
#                 score = accuracy_score(y_true, y_pred)
#             elif scoring_metric == "roc_auc":
#                 fpr, tpr, _ = roc_curve(y_true, y_prob)
#                 score = auc(fpr, tpr)
#             elif scoring_metric == "pr_auc":
#                 precision, recall, _ = precision_recall_curve(y_true, y_prob)
#                 score = auc(recall, precision)
#             else:
#                 score = f1_score(y_true, y_pred, average="weighted")  # Default to F1

#             # Print progress
#             print(
#                 f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
#                 f"Val Loss = {avg_val_loss:.4f}, {scoring_metric} = {score:.4f}"
#             )

#             # Save best model
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 current_best_model_state = model.state_dict().copy()
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += 1

#             # Early stopping
#             if hyperparams.early_stopping and epochs_no_improve >= hyperparams.patience:
#                 print(f"Early stopping after {epoch + 1} epochs")
#                 break

#         # Training time
#         training_time = time.time() - start_time

#         # Load best model for final evaluation
#         if current_best_model_state is not None:
#             model.load_state_dict(current_best_model_state)

#         # Final evaluation metrics
#         model.eval()
#         y_true, y_pred, y_prob = [], [], []

#         with torch.no_grad():
#             for seq_feats, char_feats, vector_feats, scalars, label in val_loader:
#                 # Move batch data to device
#                 seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
#                 char_feats = {k: v.to(device) for k, v in char_feats.items()}
#                 vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
#                 scalars = {k: v.to(device) for k, v in scalars.items()}
#                 label = label.to(device)

#                 logits = model(seq_feats, char_feats, vector_feats, scalars)
#                 preds = torch.argmax(logits, dim=1)
#                 probs = F.softmax(logits, dim=1)[:, 1]

#                 y_true.extend(label.cpu().tolist())
#                 y_pred.extend(preds.cpu().tolist())
#                 y_prob.extend(probs.cpu().tolist())

#         # Calculate final metrics
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="weighted")
#         recall = recall_score(y_true, y_pred, average="weighted")
#         f1 = f1_score(y_true, y_pred, average="weighted")

#         pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
#         pr_auc_score = auc(pr_recall, pr_precision)

#         fpr, tpr, _ = roc_curve(y_true, y_prob)
#         roc_auc_score = auc(fpr, tpr)

#         cm = confusion_matrix(y_true, y_pred)

#         # Get model size
#         fd, path = tempfile.mkstemp()
#         os.close(fd)
#         torch.save(model.state_dict(), path)
#         size = os.path.getsize(path) / 1024  # KB
#         os.remove(path)

#         # Determine current score based on chosen metric
#         current_score = None
#         if scoring_metric == "f1":
#             current_score = f1
#         elif scoring_metric == "precision":
#             current_score = precision
#         elif scoring_metric == "recall":
#             current_score = recall
#         elif scoring_metric == "accuracy":
#             current_score = accuracy
#         elif scoring_metric == "roc_auc":
#             current_score = roc_auc_score
#         elif scoring_metric == "pr_auc":
#             current_score = pr_auc_score
#         else:
#             current_score = f1  # Default to F1 if invalid metric

#         # Check if this is the best model so far based on selected metric
#         if current_score > best_score:
#             best_score = current_score
#             best_model_state = current_best_model_state
#             best_fold_idx = fold_count
#             print(f"  * New best model: {scoring_metric.upper()}={current_score:.4f}")

#         # Save results
#         results["accuracy"].append(accuracy)
#         results["precision"].append(precision)
#         results["recall"].append(recall)
#         results["f1"].append(f1)
#         results["pr_auc"].append(pr_auc_score)
#         results["roc_auc"].append(roc_auc_score)
#         results["pr_curve_data"].append((pr_precision, pr_recall))
#         results["roc_curve_data"].append((fpr, tpr))
#         results["conf_matrices"].append(cm)
#         results["model_size"].append(size)
#         results["train_losses"].append(train_losses)
#         results["val_losses"].append(val_losses)
#         results["training_time"].append(training_time)
#         results["fold_info"].append(
#             {
#                 "repetition": current_rep,
#                 "fold": current_fold,
#                 "train_idx": train_idx,
#                 "val_idx": val_idx,
#                 "train_samples": len(train_idx),
#                 "val_samples": len(val_idx),
#             }
#         )

#         print(
#             f"Fold {fold_count} results: F1={f1:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, PR-AUC={pr_auc_score:.4f}, ROC-AUC={roc_auc_score:.4f}"
#         )
#         print(f"Training time: {training_time:.2f} seconds")
#         fold_count += 1

#     # Calculate means and standard deviations
#     print("\n=== Overall Results ===")

#     for metric in [
#         "accuracy",
#         "precision",
#         "recall",
#         "f1",
#         "pr_auc",
#         "roc_auc",
#         "model_size",
#         "training_time",
#     ]:
#         results[f"mean_{metric}"] = np.mean(results[metric], dtype=np.float64)
#         results[f"std_{metric}"] = np.std(results[metric], dtype=np.float64)

#         print(
#             f"Mean {metric}: {results[f'mean_{metric}']:.4f} ± {results[f'std_{metric}']:.4f}"
#         )

#     results["best_fold_idx"] = best_fold_idx
#     results["best_score"] = best_score
#     results["scoring_metric"] = scoring_metric

#     print(
#         f"Best model from fold {best_fold_idx} with {scoring_metric}={best_score:.4f}"
#     )

#     # Initialize best model with best weights
#     best_model = DebrimModel.from_config(
#         vocab_dict,
#         scalar_cols=scalar_cols,
#         char_cols=char_cols,
#         vector_cols=vector_cols,
#         vector_dims=vector_dims,
#         embedding_dim=hyperparams.embedding_dim,
#         seq_pooling=hyperparams.seq_pooling,
#         hidden_dim=hyperparams.hidden_dims,
#         n_classes=hyperparams.n_classes,
#         dropout=hyperparams.dropout,
#     )
#     best_model.load_state_dict(best_model_state)

#     return results, best_model


def predict(
    model: DebrimModel,
    df: pd.DataFrame,
    vocab_dict: dict[str, Vocab],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    device: torch.device | None = None,
    batch_size: int = 64,
    label_col: str = "is_malware",
) -> tuple[list[int], list[list[float]]]:
    """
    Generate predictions and probability scores using a trained DebrimModel.

    Parameters
    ----------
    model : DebrimModel
        Trained model to use for inference
    df : pandas.DataFrame
        Data to perform predictions on
    vocab_dict : dict
        Dictionary of vocabularies for feature columns
    scalar_cols : list
        List of scalar columns to include
    char_cols : list
        List of columns to be processed character by character
    vector_cols : list
        List of columns containing fixed-length vectors
    device : torch.device or None
        Device to run inference on
    batch_size : int
        Batch size for prediction
    label_col : str
        Name of the label column

    Returns
    -------
    tuple
        (predictions, probabilities)
    """
    # Check for device
    if device is None:
        device = get_best_available_device()

    # Prepare model for inference
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = DebrimDataset(
        df,
        vocab_dict,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=label_col,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_preds = []
    all_probs = []

    # Perform inference in batches
    with torch.no_grad():
        for seq_feats, char_feats, vector_feats, scalars, _ in loader:
            # Move batch data to device
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}

            # Get model outputs
            logits = model(seq_feats, char_feats, vector_feats, scalars)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Move results back to CPU and convert to Python objects
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_probs


def extract_embeddings(
    model: DebrimModel | DebrimEmbedder,
    df: pd.DataFrame,
    vocab_dict: dict[str, Vocab],
    sequence_cols: list[str],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    device: torch.device,
    batch_size: int = 64,
    label_col: str = "is_malware",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from the neural network model's embedder component.

    Parameters
    ----------
    model : DebrimModel | DebrimEmbedder
        Trained model or embedder to extract embeddings from
    df : pandas.DataFrame
        Data to extract embeddings for
    vocab_dict : dict
        Dictionary of vocabularies for feature columns
    scalar_cols : list
        List of scalar columns
    char_cols : list
        List of columns to be processed character by character
    vector_cols : list
        List of columns containing fixed-length vectors
    device : torch.device
        Device to run extraction on
    batch_size : int
        Batch size for processing
    label_col : str
        Name of the label column

    Returns
    -------
    tuple
        (embeddings, labels)
    """
    model.eval()

    # Create dataset and dataloader
    dataset = DebrimDataset(
        df,
        vocab_dict,
        scalar_cols=scalar_cols,
        sequence_cols=sequence_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=label_col,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_embeddings = []
    all_labels = []

    # Determine whether we're using a full model or just the embedder
    embedder = model.embedder if isinstance(model, DebrimModel) else model

    with torch.no_grad():
        for seq_feats, char_feats, vector_feats, scalars, labels in loader:
            # Move batch data to device
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}

            # Get embeddings directly from the embedder component
            embeddings, _ = embedder(seq_feats, char_feats, vector_feats, scalars)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_embeddings), np.concatenate(all_labels)
