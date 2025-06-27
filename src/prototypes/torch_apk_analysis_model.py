import platform
import tempfile
import os
import time
import copy
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from dataclasses import dataclass, field
from typing import Literal, Self, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    average_precision_score,
    recall_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    classification_report,
)

from ..utils.preprocessing_utils import apply_scalers_to_dataframe


@dataclass(frozen=True)
class NNHyperparams:
    """
    Dataclass to hold hyperparameters for the neural network models.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch during training
    max_learning_rate : float
        Maximum learning rate for the learning rate scheduler; represents
        step size for the optimizer
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
    dataloader_num_workers : int
        Number of subprocesses to use for data loading
    dataloader_pin_memory : bool
        Whether to pin memory for faster data transfer to GPU
    dataloader_persistent_workers : bool
        Whether to keep workers alive after the first epoch
    grad_scaler_max_norm : float
        Maximum norm for gradient scaling to prevent exploding gradients
    """

    batch_size: int = 64
    max_learning_rate: float = 1e-3
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

    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False

    grad_scaler_max_norm: float = 1.0


class ApkAnalysisDataset(Dataset):
    """
    PyTorch Dataset for APK analysis data with mixed input types.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_cols=None,
        scalar_cols=None,
        char_cols=None,
        vector_cols=None,
        label_col="is_malware",
    ):
        self.sequence_cols = sequence_cols or []
        self.scalar_cols = scalar_cols or []
        self.char_cols = char_cols or []
        self.vector_cols = vector_cols or []
        self.label_col = label_col

        # Extract necessary data into lists to reduce memory per worker
        self.processed_seq_features = {}
        for col in self.sequence_cols:
            if col in df.columns:
                self.processed_seq_features[col] = np.array(
                    df[col].tolist(), dtype=np.int32
                )

        self.processed_char_features = {}
        for col in self.char_cols:
            if col in df.columns:
                self.processed_char_features[col] = np.array(
                    df[col].tolist(), dtype=np.int32
                )

        self.processed_vector_features = {}
        for col in self.vector_cols:
            if col in df.columns:
                self.processed_vector_features[col] = np.array(
                    df[col].tolist(), dtype=np.float32
                )

        self.processed_scalars = {}
        for col in self.scalar_cols:
            if col in df.columns:
                # Scalars are single numbers, tolist() will create a list of these numbers
                self.processed_scalars[col] = df[col].to_numpy(dtype=np.float32)

        if self.label_col in df.columns:
            self.labels = df[self.label_col].to_numpy(dtype=np.int64)
        else:
            # Handle missing label column if necessary, e.g., for prediction without labels
            self.labels = [0] * len(df)  # Or raise error, or handle appropriately
            print(
                f"Warning: Label column '{self.label_col}' not found in DataFrame. Using dummy labels."
            )

        self._length = len(df)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        seq_features = {
            col: torch.tensor(self.processed_seq_features[col][idx], dtype=torch.long)
            for col in self.sequence_cols
            if col in self.processed_seq_features  # Check if key exists
        }

        char_features = {
            col: torch.tensor(self.processed_char_features[col][idx], dtype=torch.long)
            for col in self.char_cols
            if col in self.processed_char_features
        }

        vector_features = {
            col: torch.tensor(
                self.processed_vector_features[col][idx], dtype=torch.float
            )
            for col in self.vector_cols
            if col in self.processed_vector_features
        }

        # Scalars are already individual numbers in the lists
        scalars = {
            col: torch.tensor(
                self.processed_scalars[col][idx], dtype=torch.float
            ).unsqueeze(0)
            for col in self.scalar_cols
            if col in self.processed_scalars
        }

        label = torch.tensor(self.labels[idx], dtype=torch.long)

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
    (
        seq_features_list,
        char_features_list,
        vector_features_list,
        scalars_list,
        labels_list,
    ) = zip(*batch)

    batched_seq_features = {}
    if seq_features_list[0]:
        for col in seq_features_list[0].keys():
            batched_seq_features[col] = torch.stack(
                [f[col] for f in seq_features_list if col in f]
            )

    batched_char_features = {}
    if char_features_list[0]:
        for col in char_features_list[0].keys():
            batched_char_features[col] = torch.stack(
                [f[col] for f in char_features_list if col in f]
            )

    batched_vector_features = {}
    if vector_features_list[0]:
        for col in vector_features_list[0].keys():
            batched_vector_features[col] = torch.stack(
                [f[col] for f in vector_features_list if col in f]
            )

    batched_scalars = {}
    if scalars_list[0]:
        for col in scalars_list[0].keys():
            batched_scalars[col] = torch.stack(
                [s[col] for s in scalars_list if col in s]
            )

    batched_labels = torch.stack(labels_list)

    return (
        batched_seq_features,
        batched_char_features,
        batched_vector_features,
        batched_scalars,
        batched_labels,
    )


class APKFeatureEmbedder(nn.Module):
    """Feature embedding module for the APK analysis model."""

    def __init__(
        self,
        vocab_dict,
        sequence_cols=None,
        scalar_cols=None,
        embedding_dim=128,
        seq_pooling="mean",
        char_cols=None,
        vector_cols=None,
        vector_dims=None,
    ):
        super().__init__()
        self.seq_padding_indices = {}
        self.embedding_dim = embedding_dim
        self.seq_pooling = seq_pooling
        self.sequence_cols = sequence_cols or []
        self.scalar_cols = scalar_cols or []
        self.char_cols = char_cols or []
        self.vector_cols = vector_cols or []
        self.vector_dims = vector_dims or {}

        # Create embedding layer for each sequence feature
        self.seq_embedders = nn.ModuleDict()
        for col in self.sequence_cols:
            if col in vocab_dict:
                current_vocab = vocab_dict[col]
                padding_idx = current_vocab.get("<PAD>", 0)
                self.seq_embedders[col] = nn.Embedding(
                    num_embeddings=len(current_vocab),
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                )
                self.seq_padding_indices[col] = padding_idx

        self.char_padding_indices = {}  # Also for char features if a similar mask is needed
        if any(col in self.char_cols for col in vocab_dict):
            self.char_embedders = nn.ModuleDict()
            self.char_gru = nn.ModuleDict()

            for col in self.char_cols:
                if col in vocab_dict:
                    current_vocab = vocab_dict[col]
                    padding_idx = current_vocab.get("<PAD>", 0)
                    self.char_embedders[col] = nn.Embedding(
                        num_embeddings=len(current_vocab),
                        embedding_dim=embedding_dim // 2,
                        padding_idx=padding_idx,
                    )
                    self.char_padding_indices[col] = padding_idx
                    # Ensure GRU is created for each char_col that has an embedder
                    self.char_gru[col] = nn.GRU(
                        input_size=embedding_dim // 2,
                        hidden_size=embedding_dim,
                        batch_first=True,
                        bidirectional=False,
                    )
        else:  # Ensure these exist even if no char_cols are processed to avoid attribute errors
            self.char_embedders = nn.ModuleDict()
            self.char_gru = nn.ModuleDict()

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

        self.output_dim = 0
        if self.seq_embedders:
            self.output_dim += len(self.sequence_cols) * embedding_dim
        if self.char_embedders:
            self.output_dim += len(self.char_cols) * embedding_dim
        if self.vector_reducers:
            self.output_dim += len(self.vector_cols) * embedding_dim

    def forward(
        self,
        seq_feats: dict[str, torch.Tensor],
        char_feats: dict[str, torch.Tensor],
        vector_feats: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Process and embed all feature types."""
        pooled_embeddings = []

        # Process sequence features
        for feature_name, sequence in seq_feats.items():
            if (
                feature_name in self.seq_embedders
            ):  # Check if embedder exists for this feature
                embeddings = self.seq_embedders[feature_name](sequence)

                # Use pre-computed padding index
                padding_idx_for_mask = self.seq_padding_indices[feature_name]

                padding_mask = (sequence != padding_idx_for_mask).unsqueeze(-1).float()

                if self.seq_pooling == "mean":
                    summed = (embeddings * padding_mask).sum(dim=1)
                    token_counts = padding_mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / token_counts
                else:  # max pooling
                    masked_embeddings = embeddings.masked_fill(
                        padding_mask == 0, float("-inf")
                    )
                    pooled = masked_embeddings.max(dim=1).values
                pooled_embeddings.append(pooled)

        # Process character features
        for feature_name, char_sequence in char_feats.items():
            if (
                feature_name in self.char_embedders and feature_name in self.char_gru
            ):  # Check both exist
                char_embeddings = self.char_embedders[feature_name](char_sequence)
                gru_output, hidden = self.char_gru[feature_name](char_embeddings)
                pooled_embeddings.append(hidden.squeeze(0))

        # Process vector features
        for feature_name, vector in vector_feats.items():
            if feature_name in self.vector_reducers:  # Check if reducer exists
                reduced = self.vector_reducers[feature_name](vector)
                pooled_embeddings.append(reduced)

        if not pooled_embeddings:
            batch_size = 1
            # Try to determine batch_size from any of the input tensors
            if seq_feats:
                batch_size = next(iter(seq_feats.values())).size(0)
            elif char_feats:
                batch_size = next(iter(char_feats.values())).size(0)
            elif vector_feats:
                batch_size = next(iter(vector_feats.values())).size(0)
            elif scalars:
                batch_size = next(iter(scalars.values())).size(0)

            device = (
                next(self.parameters()).device
                if list(self.parameters())
                else torch.device("cpu")
            )

            # Ensure output_dim is sensible if no embeddings were generated
            effective_output_dim = self.output_dim if self.output_dim > 0 else 1
            # If output_dim is 0 because no embedders were configured, this will be 1.
            # The classifier's input_dim should align with this.

            concatenated_embeddings = torch.zeros(
                (batch_size, effective_output_dim), device=device
            )
            if self.output_dim == 0:
                print(
                    "Warning: APKFeatureEmbedder output_dim is 0. No embeddings generated, returning zeros."
                )
        else:
            concatenated_embeddings = torch.cat(pooled_embeddings, dim=1)

        return concatenated_embeddings, scalars


class APKClassifier(nn.Module):
    """
    MLP classifier head for the APKAnalysis model.

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


class APKAnalysisModel(nn.Module):
    """
    Complete APK Analysis model combining embedder and classifier components.

    This model integrates feature embedding and classification components into
    a complete end-to-end architecture for malware detection.

    Parameters
    ----------
    embedder : APKFeatureEmbedder
        The embedder component that processes different feature types
    classifier : APKClassifier, optional
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

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # If no classifier is provided, create a default one with standard parameters
        if classifier is None:
            total_input_dim = embedder.output_dim + scalar_dim
            self.classifier = APKClassifier(
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
        vocab_dict: dict[str, dict[str, int]],
        sequence_cols: list[str],
        scalar_cols: list[str],
        char_cols: list[str],
        vector_cols: list[str],
        vector_dims: dict[str, int],
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

        embedder = APKFeatureEmbedder(
            vocab_dict=vocab_dict,
            sequence_cols=sequence_cols,
            scalar_cols=scalar_cols,
            embedding_dim=embedding_dim,
            seq_pooling=seq_pooling,
            char_cols=char_cols,
            vector_cols=vector_cols,
            vector_dims=vector_dims,
        )

        # Use the output dimension from the created embedder
        total_input_dim = embedder.output_dim + len(scalar_cols)

        classifier = APKClassifier(
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
            scalar_tensors = [scalars[col] for col in scalars if col in scalars]

            if scalar_tensors:
                scalar_tensor = torch.cat(scalar_tensors, dim=1)
                combined_features = torch.cat([embeddings, scalar_tensor], dim=1)
            else:
                combined_features = embeddings
        else:
            combined_features = embeddings

        # Pass through classifier to get output logits
        quantized_features = self.quant(combined_features)
        quantized_logits = self.classifier(quantized_features)
        logits = self.dequant(quantized_logits)

        return logits


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


def train_nn_model(
    df: pd.DataFrame,
    vocab_dict: dict[str, dict[str, int]],
    sequence_cols: list[str],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    vector_dims: dict[str, int],
    hyperparams: NNHyperparams = NNHyperparams(),
    device: torch.device | None = None,
    train_split_ratio: float = 0.8,
    val_df_explicit: pd.DataFrame | None = None,
    scoring_metric: Literal[
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ] = "f1",
    random_seed: int = 42,
) -> tuple[APKAnalysisModel, dict[str, Any], dict[str, StandardScaler]]:
    """
    Train a neural network model on the given dataset.
    Can perform a single train/validation split or use explicitly provided validation data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing features and labels.
    vocab_dict : dict
        Dictionary mapping feature names to vocabulary objects.
    sequence_cols : list
        List of sequence columns.
    scalar_cols : list
        List of scalar columns to include as numerical features.
    char_cols : list
        List of columns to be processed character by character.
    vector_cols : list
        List of columns containing fixed-length vectors.
    vector_dims : dict
        Dictionary specifying the dimension of each vector feature.
    hyperparams : NNHyperparams
        Dataclass containing all hyperparameters for model training.
    device : torch.device or None
        Device to run the training on (defaults to best available device).
    train_split_ratio : float
        Proportion of the dataset to use for training (0.0 to 1.0).
    val_df_explicit : pandas.DataFrame or None
        Explicit validation DataFrame to use instead of splitting from training data.
        If provided, `train_split_ratio` is ignored.
    scoring_metric : str
        Primary metric to use for model evaluation.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (trained_model, results_dict, scalers_used)
    """
    if device is None:
        device = get_best_available_device()

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Using device: {device}")
    print(f"Using {scoring_metric} as the primary scoring metric for validation.")

    train_df = df
    val_df = val_df_explicit

    if val_df is None:
        print(
            f"Performing internal train/validation split with ratio: {train_split_ratio}"
        )
        train_df, val_df = train_test_split(
            df,
            train_size=train_split_ratio,
            random_state=random_seed,
            stratify=df[hyperparams.label_col]
            if hyperparams.label_col in df.columns
            else None,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    else:
        print("Using explicitly provided training and validation DataFrames.")
        train_df = train_df.copy().reset_index(drop=True)
        val_df = val_df.copy().reset_index(drop=True)

    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    if hyperparams.label_col in train_df.columns:  # Check train_df for label_col
        print(
            f"Training set class distribution: {train_df[hyperparams.label_col].value_counts().to_dict()}"
        )
    if hyperparams.label_col in val_df.columns:  # Check val_df for label_col
        print(
            f"Validation set class distribution: {val_df[hyperparams.label_col].value_counts().to_dict()}"
        )

    # Normalize scalar and vector features
    scalers_used = {}
    train_df, scalers_used = apply_scalers_to_dataframe(
        train_df, scalar_cols, vector_cols, fit_scalers=True
    )
    val_df, _ = apply_scalers_to_dataframe(
        val_df, scalar_cols, vector_cols, scalers=scalers_used, fit_scalers=False
    )

    # Compute class weights for imbalanced data based on the training set
    y_train = train_df[hyperparams.label_col].values
    expected_classes = np.array([0, 1])
    if len(np.unique(y_train)) < len(expected_classes) and hyperparams.n_classes == 2:
        print(
            "Warning: Not all classes present in training data. Using uniform class weights."
        )
        class_weights_arr = np.ones(hyperparams.n_classes)
    else:
        class_weights_arr = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        # Ensure weights array matches n_classes if some classes are missing
        if len(class_weights_arr) < hyperparams.n_classes:
            temp_weights = np.ones(hyperparams.n_classes)
            present_classes = np.unique(y_train)
            for i, cls_val in enumerate(present_classes):
                if cls_val < hyperparams.n_classes:  # Ensure class index is valid
                    temp_weights[cls_val] = class_weights_arr[i]
            class_weights_arr = temp_weights

    weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float).to(device)
    print(f"Using class weights: {class_weights_arr}")

    # Create datasets and data loaders
    train_dataset = ApkAnalysisDataset(
        train_df,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=hyperparams.label_col,
    )
    val_dataset = ApkAnalysisDataset(
        val_df,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=hyperparams.label_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=hyperparams.dataloader_num_workers,
        pin_memory=hyperparams.dataloader_pin_memory
        if device.type == "cuda"
        else False,
        persistent_workers=hyperparams.dataloader_persistent_workers
        and hyperparams.dataloader_num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hyperparams.dataloader_num_workers,
        pin_memory=hyperparams.dataloader_pin_memory
        if device.type == "cuda"
        else False,
        persistent_workers=hyperparams.dataloader_persistent_workers
        and hyperparams.dataloader_num_workers > 0,
    )

    # Initialize model
    model = APKAnalysisModel.from_config(
        vocab_dict,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        vector_dims=vector_dims,
        embedding_dim=hyperparams.embedding_dim,
        hidden_dim=hyperparams.hidden_dims,
        seq_pooling=hyperparams.seq_pooling,
        n_classes=hyperparams.n_classes,
        dropout=hyperparams.dropout,
    ).to(device)

    # Optimizer
    if hyperparams.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams.max_learning_rate,
            weight_decay=hyperparams.weight_decay,
        )
    elif hyperparams.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams.max_learning_rate,
            weight_decay=hyperparams.weight_decay,
        )
    elif hyperparams.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams.max_learning_rate,
            weight_decay=hyperparams.weight_decay,
            momentum=0.9,
        )  # Added momentum for SGD
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams.max_learning_rate,
            weight_decay=hyperparams.weight_decay,
        )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hyperparams.max_learning_rate,  # Use max_learning_rate from hyperparams
        steps_per_epoch=len(train_loader),
        epochs=hyperparams.epochs,
    )

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    train_losses_epoch, val_losses_epoch = [], []
    best_val_metric_score = (
        -float("inf")
        if scoring_metric
        in ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]
        else float("inf")
    )
    best_val_loss = float("inf")  # For saving model based on val loss

    patience_counter = 0
    best_model_state = None

    hyperparams_dict_log = {k: v for k, v in hyperparams.__dict__.items()}
    results = {
        "hyperparams": hyperparams_dict_log,
        "train_losses": [],
        "val_losses": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "pr_curve_data": [],
        "roc_curve_data": [],
        "conf_matrix": None,
        "model_size": None,
        "training_time": None,
        "best_epoch": -1,
        "final_metrics_best_model": {},
    }

    print("Starting training...")
    training_start_time = time.time()

    for epoch in range(hyperparams.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (
            seq_feats,
            char_feats,
            vector_feats,
            scalars,
            labels,
        ) in enumerate(train_loader):
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                logits = model(seq_feats, char_feats, vector_feats, scalars)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                print(
                    f"NaN loss detected at Epoch {epoch + 1}, Batch {batch_idx}! Stopping."
                )
                results["training_time"] = time.time() - training_start_time

                print("Warning: Training stopped due to NaN loss.")
                return model, results, scalers_used

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=hyperparams.grad_scaler_max_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Scheduler step per batch for OneCycleLR

            epoch_train_loss += loss.item()
            if batch_idx % (len(train_loader) // 5 + 1) == 0:  # Log ~5 times per epoch
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        results["train_losses"].append(avg_epoch_train_loss)
        print(
            f"Epoch {epoch + 1}/{hyperparams.epochs} — Train Loss: {avg_epoch_train_loss:.4f}"
        )

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        y_true_val, y_pred_val, y_prob_val = [], [], []
        with torch.no_grad():
            for seq_feats, char_feats, vector_feats, scalars, labels in val_loader:
                seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
                char_feats = {k: v.to(device) for k, v in char_feats.items()}
                vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
                scalars = {k: v.to(device) for k, v in scalars.items()}
                labels = labels.to(device)

                with torch.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    logits = model(seq_feats, char_feats, vector_feats, scalars)
                    loss = criterion(logits, labels)

                epoch_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1)[:, 1]  # Prob of positive class

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                y_prob_val.extend(probs.cpu().numpy())

        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        results["val_losses"].append(avg_epoch_val_loss)

        if not np.all(np.isfinite(y_prob_val)):
            print(
                f"  Warning: NaN detected in validation probabilities at epoch {epoch + 1}. Assigning worst possible score for this epoch."
            )
            # Assign worst scores and skip metric calculation to avoid crashing
            acc_val, prec_val, rec_val, f1_val, roc_auc_val, pr_auc_val = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            current_pr_curve_data, current_roc_curve_data = None, None
        else:
            # Calculate validation metrics
            acc_val = accuracy_score(y_true_val, y_pred_val)
            prec_val = precision_score(
                y_true_val, y_pred_val, average="binary", zero_division=0
            )
            rec_val = recall_score(
                y_true_val, y_pred_val, average="binary", zero_division=0
            )
            f1_val = f1_score(y_true_val, y_pred_val, average="binary", zero_division=0)

            roc_auc_val, pr_auc_val = 0.0, 0.0
            current_pr_curve_data, current_roc_curve_data = None, None
            if (
                len(np.unique(y_true_val)) > 1
            ):  # AUC scores require at least two classes
                fpr, tpr, _ = roc_curve(y_true_val, y_prob_val)
                roc_auc_val = auc(fpr, tpr)
                current_roc_curve_data = (fpr.tolist(), tpr.tolist())

                precision_vals, recall_vals, _ = precision_recall_curve(
                    y_true_val, y_prob_val
                )
                pr_auc_val = auc(
                    recall_vals, precision_vals
                )  # Recall is x-axis for PR-AUC
                current_pr_curve_data = (precision_vals.tolist(), recall_vals.tolist())

        results["accuracy"].append(acc_val)
        results["precision"].append(prec_val)
        results["recall"].append(rec_val)
        results["f1"].append(f1_val)
        results["roc_auc"].append(roc_auc_val)
        results["pr_auc"].append(pr_auc_val)

        current_metric_score = 0
        if scoring_metric == "f1":
            current_metric_score = f1_val
        elif scoring_metric == "recall":
            current_metric_score = rec_val
        elif scoring_metric == "precision":
            current_metric_score = prec_val
        elif scoring_metric == "accuracy":
            current_metric_score = acc_val
        elif scoring_metric == "roc_auc":
            current_metric_score = roc_auc_val
        elif scoring_metric == "pr_auc":
            current_metric_score = pr_auc_val
        else:
            current_metric_score = f1_val  # Default

        print(
            f"Epoch {epoch + 1} — Val Loss: {avg_epoch_val_loss:.4f}, Val {scoring_metric.capitalize()}: {current_metric_score:.4f} "
            f"(Acc: {acc_val:.4f}, P: {prec_val:.4f}, R: {rec_val:.4f}, F1: {f1_val:.4f}, ROC: {roc_auc_val:.4f}, PR: {pr_auc_val:.4f})"
        )

        # Early stopping logic based on validation loss
        # Save model if current val_loss is better
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            best_model_state = (
                model.state_dict().copy()
            )  # Save state of the best model based on val_loss
            results["best_epoch"] = epoch + 1
            # Store metrics for this best model
            results["final_metrics_best_model"] = {
                "accuracy": acc_val,
                "precision": prec_val,
                "recall": rec_val,
                "f1": f1_val,
                "roc_auc": roc_auc_val,
                "pr_auc": pr_auc_val,
                "val_loss": avg_epoch_val_loss,
            }
            results["pr_curve_data"] = current_pr_curve_data
            results["roc_curve_data"] = current_roc_curve_data
            results["conf_matrix"] = confusion_matrix(y_true_val, y_pred_val).tolist()
            patience_counter = 0
            print(f"  New best model (by val_loss) saved at epoch {epoch + 1}.")
        else:
            patience_counter += 1

        if hyperparams.early_stopping and patience_counter >= hyperparams.patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss."
            )
            break

    results["training_time"] = time.time() - training_start_time

    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(
            f"Loaded best model from epoch {results['best_epoch']} with Val Loss: {results['final_metrics_best_model'].get('val_loss', 'N/A'):.4f}"
        )
    else:  # Should not happen if training ran for at least one epoch
        print("Warning: No best model state was saved. Returning the last state.")

    # Get model size
    try:
        fd, path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        torch.save(model.state_dict(), path)
        results["model_size"] = os.path.getsize(path) / 1024
        os.remove(path)
    except Exception as e:
        print(f"Could not get model size: {e}")
        results["model_size"] = -1.0

    print(
        f"\nTraining finished. Total time: {results['training_time']:.2f}s. Model size: {results['model_size']:.2f} KB"
    )
    if results["best_epoch"] != -1:
        print(f"Best model (by val_loss) obtained at epoch {results['best_epoch']}:")
        for metric_name, metric_val in results["final_metrics_best_model"].items():
            print(f"  - {metric_name.capitalize()}: {metric_val:.4f}")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "val_losses",
    ]:
        if results.get(key) and len(results.get(key, [])) > 0:
            # Filter out None values that might have been appended
            valid_values = [v for v in results[key] if v is not None]
            if valid_values:
                results[f"mean_{key}"] = np.mean(valid_values)
                results[f"std_{key}"] = np.std(valid_values)
            else:
                results[f"mean_{key}"] = np.nan
                results[f"std_{key}"] = np.nan
        else:
            results[f"mean_{key}"] = np.nan
            results[f"std_{key}"] = np.nan

    # Explicit cleanup before returning
    del train_df, val_df, train_dataset, val_dataset, train_loader, val_loader
    del optimizer, scheduler, criterion, scaler
    if best_model_state:
        del best_model_state

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return model, results, scalers_used


def cross_val_train_nn_model(
    df: pd.DataFrame,
    vocab_dict: dict[str, dict[str, int]],
    sequence_cols: list[str],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    vector_dims: dict[str, int],
    hyperparams: NNHyperparams = NNHyperparams(),
    n_folds: int = 2,
    n_repetitions: int = 1,
    scoring_metric: Literal[
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ] = "f1",
    device: torch.device | None = None,
    random_seed: int = 42,
) -> tuple[APKAnalysisModel | None, dict[str, Any], dict[str, StandardScaler] | None]:
    """
    Train a neural network model with cross-validation using the provided hyperparameters.
    This function now calls train_nn_model for each fold.

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
        (best_model, results_dict, best_model_scalers)
    """

    if device is None:
        device = get_best_available_device()

    print(f"--- Cross-Validation Training ---")
    print(f"Using device: {device}")
    print(f"Primary scoring metric for best model selection: {scoring_metric.upper()}")
    print(f"Number of folds: {n_folds}, Number of repetitions: {n_repetitions}")

    X = np.arange(len(df))
    y = df[hyperparams.label_col].values

    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repetitions, random_state=random_seed
    )

    # Initialize results storage for aggregated metrics
    # We will store the 'final_metrics_best_model' from each fold's train_nn_model run
    aggregated_results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "val_loss": [],
        # Per-fold details
        "model_size": [],
        "training_time": [],
        "roc_curve_data": [],
        "pr_curve_data": [],
        "conf_matrices": [],
        "train_losses": [],
        "val_losses": [],
        # Metadata
        "fold_info": [],
        "hyperparams": {k: v for k, v in hyperparams.__dict__.items()},
    }

    overall_best_model_state = None
    overall_best_scalers = None
    overall_best_score = -float("inf")
    best_fold_info_str = "N/A"

    fold_counter = 0
    for rep_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        fold_counter += 1
        current_rep = (rep_idx // n_folds) + 1
        current_fold_in_rep = (rep_idx % n_folds) + 1

        print(
            f"\n--- Repetition {current_rep}/{n_repetitions}, Fold {current_fold_in_rep}/{n_folds} (Overall Fold {fold_counter}) ---"
        )

        train_df_fold = df.iloc[
            train_idx
        ]  # No reset_index here, train_nn_model will handle it
        val_df_fold = df.iloc[val_idx]  # No reset_index here

        # Call the main training function for this fold
        # Note: train_nn_model will handle its own normalization based on train_df_fold
        model_fold, results_fold, scalers_fold = train_nn_model(
            df=train_df_fold,  # This is the training data for this fold
            val_df_explicit=val_df_fold,  # This is the validation data for this fold
            vocab_dict=vocab_dict,
            sequence_cols=sequence_cols,
            scalar_cols=scalar_cols,
            char_cols=char_cols,
            vector_cols=vector_cols,
            vector_dims=vector_dims,
            hyperparams=hyperparams,  # Pass the same hyperparams for each fold
            device=device,
            scoring_metric=scoring_metric,  # train_nn_model uses this for its internal best model (by val_loss)
            random_seed=random_seed
            + fold_counter,  # Vary seed slightly per fold for optimizer init if desired
        )

        # Extract metrics from the 'final_metrics_best_model' dict returned by train_nn_model
        # This dict contains metrics of the model that had the best *validation loss* within that fold's training
        fold_final_metrics = results_fold.get("final_metrics_best_model", {})

        for metric_key in aggregated_results.keys():
            if metric_key in fold_final_metrics:
                aggregated_results[metric_key].append(fold_final_metrics[metric_key])

        aggregated_results["model_size"].append(results_fold.get("model_size"))
        aggregated_results["training_time"].append(results_fold.get("training_time"))
        aggregated_results["roc_curve_data"].append(results_fold.get("roc_curve_data"))
        aggregated_results["pr_curve_data"].append(results_fold.get("pr_curve_data"))
        aggregated_results["conf_matrices"].append(results_fold.get("conf_matrix"))
        aggregated_results["train_losses"].append(results_fold.get("train_losses"))
        aggregated_results["val_losses"].append(results_fold.get("val_losses"))

        aggregated_results["fold_info"].append(
            {
                "repetition": current_rep,
                "fold_in_rep": current_fold_in_rep,
                "overall_fold": fold_counter,
                "train_samples": len(train_df_fold),
                "val_samples": len(val_df_fold),
                "fold_score_metric": scoring_metric,
                "fold_score_value": fold_final_metrics.get(scoring_metric, -1),
            }
        )

        print(
            f"Fold {fold_counter} finished. Val {scoring_metric.capitalize()}: {fold_final_metrics.get(scoring_metric, 'N/A'):.4f} (from best model by val_loss within fold)"
        )

        # Compare this fold's model (based on the specified scoring_metric on its val set)
        # to find the overall best model across all folds
        current_fold_score = fold_final_metrics.get(scoring_metric, -float("inf"))
        if current_fold_score > overall_best_score:
            overall_best_score = current_fold_score
            overall_best_model_state = copy.deepcopy(
                model_fold.state_dict()
            )  # Save state of the best model
            overall_best_scalers = copy.deepcopy(scalers_fold)
            best_fold_info_str = f"Rep {current_rep}, Fold {current_fold_in_rep} (Overall Fold {fold_counter})"
            print(
                f"  *** New overall best model found based on {scoring_metric.upper()} across folds! Score: {overall_best_score:.4f} ***"
            )

        # Explicitly delete model and other large objects from the fold to free memory
        del (
            model_fold,
            results_fold,
            train_df_fold,
            val_df_fold,
            fold_final_metrics,
            scalers_fold,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    # Calculate mean and std for aggregated metrics
    summary_metrics = {}
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "val_loss",
        "model_size",
        "training_time",
    ]:
        if aggregated_results.get(key) and len(aggregated_results.get(key, [])) > 0:
            # Filter out None values that might have been appended
            valid_values = [v for v in aggregated_results[key] if v is not None]
            if valid_values:
                aggregated_results[f"mean_{key}"] = np.mean(valid_values)
                aggregated_results[f"std_{key}"] = np.std(valid_values)
            else:
                aggregated_results[f"mean_{key}"] = np.nan
                aggregated_results[f"std_{key}"] = np.nan
        else:
            aggregated_results[f"mean_{key}"] = np.nan
            aggregated_results[f"std_{key}"] = np.nan

    aggregated_results["summary_metrics"] = summary_metrics

    print("\n--- Overall Cross-Validation Summary ---")
    for key, value in aggregated_results.items():
        if key.startswith("mean_") or key.startswith("std_"):
            print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")

    print(
        f"Best model across all folds (based on {scoring_metric.upper()} on validation set of its fold): {best_fold_info_str} with score {overall_best_score:.4f}"
    )

    # Instantiate and load the overall best model
    if overall_best_model_state:
        print("Loading the overall best model state...")
        best_model_overall = APKAnalysisModel.from_config(
            vocab_dict,
            sequence_cols=sequence_cols,
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
        best_model_overall.load_state_dict(overall_best_model_state)
        best_model_overall.to(device)  # Ensure it's on the correct device
        best_model_overall.eval()
    else:
        print("Warning: No best model state was found during cross-validation.")
        best_model_overall = None
        overall_best_scalers = None

    return best_model_overall, aggregated_results, overall_best_scalers


def predict(
    model: APKAnalysisModel,
    df: pd.DataFrame,
    scalers: dict[str, StandardScaler],
    sequence_cols: list[str],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    device: torch.device | None = None,
    batch_size: int = 64,
    label_col: str = "is_malware",
) -> tuple[list[int], list[list[float]]]:
    """
    Generate predictions and probability scores using a trained APKAnalysisModel.

    Parameters
    ----------
    model : APKAnalysisModel
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
    device : torch.device | None
        Device to run inference on
    batch_size : int
        Batch size for prediction
    label_col : str
        Name of the label column
    scalers : dict[str, StandardScaler] | None
        Pre-fitted scalers for normalizing scalar and vector columns

    Returns
    -------
    tuple
        (predictions, probabilities)
    """
    if device is None:
        device = get_best_available_device()

    model = model.to(device)
    model.eval()

    df_processed, _ = apply_scalers_to_dataframe(
        df, scalar_cols, vector_cols, scalers=scalers, fit_scalers=False
    )

    dataset = ApkAnalysisDataset(
        df_processed,
        sequence_cols=sequence_cols,
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

    with torch.no_grad():
        for seq_feats, char_feats, vector_feats, scalars, _ in loader:
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}

            logits = model(seq_feats, char_feats, vector_feats, scalars)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_probs


def extract_embeddings(
    model: APKAnalysisModel | APKFeatureEmbedder,
    df: pd.DataFrame,
    scalers: dict[str, StandardScaler],
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
    model : APKAnalysisModel | APKFeatureEmbedder
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
    scalers : dict[str, StandardScaler] | None
        Pre-fitted scalers for normalizing scalar and vector columns

    Returns
    -------
    tuple
        (embeddings, labels)
    """
    if device is None:
        device = get_best_available_device()

    model.to(device)
    model.eval()

    df_processed, _ = apply_scalers_to_dataframe(
        df, scalar_cols, vector_cols, scalers=scalers, fit_scalers=False
    )

    dataset = ApkAnalysisDataset(
        df_processed,
        sequence_cols=sequence_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        scalar_cols=scalar_cols,
        label_col=label_col,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_embeddings = []
    all_labels = []
    embedder = model.embedder if isinstance(model, APKAnalysisModel) else model

    with torch.no_grad():
        for seq_feats, char_feats, vector_feats, scalars, labels in loader:
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}

            # Get embeddings from the embedder component
            embeddings, _ = embedder(seq_feats, char_feats, vector_feats, scalars)

            # Also include the original scalar features in the final "embedding" vector
            if scalar_cols and scalars:
                scalar_tensors = [scalars[col] for col in scalar_cols if col in scalars]
                if scalar_tensors:
                    scalar_tensor = torch.cat(scalar_tensors, dim=1)
                    final_embeddings = torch.cat([embeddings, scalar_tensor], dim=1)
                else:
                    final_embeddings = embeddings
            else:
                final_embeddings = embeddings

            all_embeddings.append(final_embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_embeddings), np.concatenate(all_labels)


def evaluate_model_on_test_set(
    model: APKAnalysisModel,
    df_test: pd.DataFrame,
    scalers: dict[str, StandardScaler],
    sequence_cols: list[str],
    scalar_cols: list[str],
    char_cols: list[str],
    vector_cols: list[str],
    hyperparams: NNHyperparams,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Evaluates a trained APKAnalysisModel on a completely unseen test set.

    Parameters
    ----------
    model : APKAnalysisModel
        The trained model to evaluate
    df_test : pandas.DataFrame
        The test dataset containing features and labels
    scalers : dict[str, StandardScaler]
        Pre-fitted scalers for normalizing scalar and vector columns
    sequence_cols : list[str]
        List of sequence columns to include
    scalar_cols : list[str]
        List of scalar columns to include
    char_cols : list[str]
        List of columns to be processed character by character
    vector_cols : list[str]
        List of columns containing fixed-length vectors
    vector_dims : dict[str, int]
        Dictionary mapping vector column names to their dimensions
    hyperparams : NNHyperparams
        Hyperparameters for the model, including label_col and batch_size
    device : torch.device or None
        Device to run evaluation on (if None, will use best available device)

    Returns
    -------
    dict
        A dictionary containing evaluation metrics and results.
    """

    if device is None:
        device = get_best_available_device()

    model.to(device)
    model.eval()

    print("--- Evaluating on Test Set ---")

    df_test_processed, _ = apply_scalers_to_dataframe(
        df_test, scalar_cols, vector_cols, scalers=scalers, fit_scalers=False
    )

    test_dataset = ApkAnalysisDataset(
        df_test_processed,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        label_col=hyperparams.label_col,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hyperparams.dataloader_num_workers,
        pin_memory=hyperparams.dataloader_pin_memory
        if device.type == "cuda"
        else False,
    )

    all_labels_test = []
    all_preds_test = []
    all_probs_test = []

    # measure iference time

    inference_start_time = time.time()
    with torch.no_grad():
        for seq_feats, char_feats, vector_feats, scalars, labels in test_loader:
            seq_feats = {k: v.to(device) for k, v in seq_feats.items()}
            char_feats = {k: v.to(device) for k, v in char_feats.items()}
            vector_feats = {k: v.to(device) for k, v in vector_feats.items()}
            scalars = {k: v.to(device) for k, v in scalars.items()}

            with autocast(enabled=(device.type == "cuda")):
                logits = model(seq_feats, char_feats, vector_feats, scalars)

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels_test.extend(labels.cpu().numpy())
            all_preds_test.extend(preds.cpu().numpy())
            all_probs_test.extend(probs.cpu().numpy())
    inference_time = time.time() - inference_start_time

    # Calculate metrics
    y_true = np.array(all_labels_test)
    y_pred = np.array(all_preds_test)
    y_prob = np.array(all_probs_test)

    test_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_binary": precision_score(
            y_true, y_pred, average="binary", zero_division=0
        ),
        "recall_binary": recall_score(
            y_true, y_pred, average="binary", zero_division=0
        ),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "inference_time": inference_time,
    }
    if len(np.unique(y_true)) > 1:
        test_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        test_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        test_metrics["roc_auc"] = np.nan
        test_metrics["pr_auc"] = np.nan

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"  Inference Time: {test_metrics['inference_time']:.2f} seconds")
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            print(f"  {metric_name.replace('_', ' ').capitalize()}: {metric_value:.4f}")
        elif metric_name == "confusion_matrix":
            print(f"  Confusion Matrix:\n{np.array(metric_value)}")
    print("---------------------------------")

    return test_metrics
