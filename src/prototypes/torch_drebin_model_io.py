import os
import json
import torch
import pickle
import numpy as np
from dataclasses import asdict
from datetime import datetime

from .torch_debrin_model import (
    DebrimModel,
    DebrimEmbedder,
    NNHyperparams,
    get_best_available_device,
)

from typing import Any
from pathlib import Path


def _convert_numpy_to_list(data: Any) -> Any:
    """
    Recursively convert numpy arrays to lists in data structures.

    This helper function makes complex data structures containing numpy
    arrays JSON-serializable by converting arrays to standard Python lists.

    Parameters
    ----------
    data : Any
        Input data that might contain numpy arrays (dict, list, ndarray, etc.)

    Returns
    -------
    Any
        Data with all numpy arrays converted to standard Python lists
    """
    if isinstance(data, dict):
        return {k: _convert_numpy_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_numpy_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_convert_numpy_to_list(item) for item in data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def save_model_with_metadata(
    model: DebrimModel,
    vocab_dict: dict[str, dict[str, int]],
    hyperparams: NNHyperparams,
    results: dict[str, Any] | None = None,
    save_dir: Path | str = "./model_artifacts/nn_models",
) -> dict[str, str]:
    """
    Save a trained neural network model with all its components and metadata.

    This function creates a timestamped directory and saves:
    - Complete model and its components (embedder, classifier)
    - Vocabulary dictionaries needed for tokenization
    - Hyperparameters used for training
    - Evaluation metrics and results

    Parameters
    ----------
    model : DebrimModel
        The trained neural network model to save
    vocab_dict : dict
        Dictionary mapping feature names to vocabulary objects
    hyperparams : NNHyperparams
        Hyperparameters configuration used for training
    results : dict, optional
        Dictionary containing training results and evaluation metrics
    save_dir : Path or str
        Base directory to save model artifacts

    Returns
    -------
    dict
        Dictionary of paths to all saved artifacts
    """
    # Create timestamp for version identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped directory for this model version
    model_dir = os.path.join(save_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # Define paths for all model artifacts
    paths = {
        "model_dir": model_dir,
        "full_model": os.path.join(model_dir, "debrim_model.pt"),
        "embedder": os.path.join(model_dir, "debrim_embedder.pt"),
        "classifier": os.path.join(model_dir, "debrim_classifier.pt"),
        "vocab": os.path.join(model_dir, "vocab_dict.pkl"),
        "hyperparams": os.path.join(model_dir, "hyperparams.json"),
        "metadata": os.path.join(model_dir, "metadata.json"),
        "summary_metrics": os.path.join(model_dir, "summary_metrics.json"),
        "all_metrics": os.path.join(model_dir, "all_metrics.json"),
    }  # Save model components using PyTorch serialization
    torch.save(model.state_dict(), paths["full_model"])
    torch.save(model.embedder.state_dict(), paths["embedder"])
    torch.save(model.classifier.state_dict(), paths["classifier"])

    # Save vocabulary dictionary using pickle
    with open(paths["vocab"], "wb") as f:
        pickle.dump(vocab_dict, f)

    # Save hyperparameters as JSON
    # Convert from dataclass to dictionary if needed
    if isinstance(hyperparams, dict):
        hyperparams_dict = hyperparams
    else:
        hyperparams_dict = asdict(hyperparams)

    with open(paths["hyperparams"], "w") as f:
        json.dump(hyperparams_dict, f, indent=2)

    # Prepare metadata about the model
    metadata = {
        "timestamp": timestamp,
        "model_type": "DebrimModel",
        "pytorch_version": torch.__version__,
        "feature_count": len(vocab_dict),
        "model_architecture": {
            "embedder_output_dim": model.embedder.output_dim,
            "classifier_input_dim": model.classifier.input_dim,
            "scalar_features": model.scalar_dim,
        },
        "paths": {k: os.path.basename(v) for k, v in paths.items()},
        "model_dir": model_dir,
    }

    # Save metadata to JSON
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    # Process and save evaluation metrics if available
    if results:
        # Separate summary metrics (means and standard deviations) from all metrics
        summary_metrics = {}
        all_metrics = {}

        for metric_name, metric_value in results.items():
            # Add to summary metrics if it's a mean or std metric
            if metric_name.startswith("mean_") or metric_name.startswith("std_"):
                summary_metrics[metric_name] = metric_value

            # Copy all metrics except hyperparams (saved separately)
            if metric_name != "hyperparams":
                all_metrics[metric_name] = metric_value

        # Convert numpy arrays to lists for JSON serialization
        all_metrics = _convert_numpy_to_list(all_metrics)  # Save metrics to JSON files
        with open(paths["summary_metrics"], "w") as f:
            json.dump(summary_metrics, f, indent=2)

        with open(paths["all_metrics"], "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Create a version tracking file to identify the latest model
    version_file = os.path.join(save_dir, "latest_version.txt")
    with open(version_file, "w") as f:
        f.write(timestamp)

    print(f"Model and artifacts saved to {model_dir}")
    return paths


def _resolve_model_directory(version: str | None, base_dir: Path | str) -> str:
    """
    Resolve the model directory path based on version specification.

    This helper function determines which model version to load by either:
    1. Using the explicitly specified version
    2. Finding the latest version from the version tracking file
    3. Finding the most recent version by directory timestamp

    Parameters
    ----------
    version : str or None
        Specific timestamp version to load (e.g., '20250516_123456')
        If None, the function will attempt to load the latest version
    base_dir : str or Path
        Base directory containing the model version subdirectories

    Returns
    -------
    str
        Absolute path to the resolved model directory

    Raises
    ------
    FileNotFoundError
        If the specified version, base directory, or any model directory cannot be found
    """
    # Case 1: Use explicitly specified version
    if version is not None:
        model_dir = os.path.join(base_dir, version)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model version directory not found: {model_dir}")
        return model_dir

    # Case 2: Try to load the latest version from the tracking file
    version_file = os.path.join(base_dir, "latest_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            latest_version = f.read().strip()
            model_dir = os.path.join(base_dir, latest_version)
            if os.path.isdir(model_dir):
                print(f"Loading latest model version: {latest_version}")
                return model_dir
            # If directory doesn't exist, fall through to case 3

    # Case 3: Find most recent directory by timestamp naming
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Get all subdirectories that have timestamp format names (start with digit)
    timestamp_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()
    ]

    if not timestamp_dirs:
        raise FileNotFoundError(
            f"No model versions found in {base_dir}"
        )  # Sort directories by name (timestamp format ensures chronological order)
    latest_version = sorted(timestamp_dirs)[-1]
    model_dir = os.path.join(base_dir, latest_version)
    print(f"Loading latest model version: {latest_version}")
    return model_dir


def _load_model_metadata(model_dir: str) -> dict[str, Any]:
    """
    Load and combine all metadata files from a model directory.

    This helper function aggregates information from:
    - model metadata (architecture, versions)
    - hyperparameters used for training
    - evaluation metrics and results (if available)

    Parameters
    ----------
    model_dir : str
        Path to the model directory containing metadata files

    Returns
    -------
    dict
        Comprehensive metadata dictionary with all available information
    """
    # Define paths to metadata files
    metadata_path = os.path.join(model_dir, "metadata.json")
    hyperparams_path = os.path.join(model_dir, "hyperparams.json")
    summary_metrics_path = os.path.join(model_dir, "summary_metrics.json")
    all_metrics_path = os.path.join(model_dir, "all_metrics.json")

    # Initialize metadata dictionary
    metadata = {}

    # Load base metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        print(f"Warning: Metadata file not found: {metadata_path}")

    # Add hyperparameters
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
            metadata["hyperparameters"] = hyperparams
    else:
        print(f"Warning: Hyperparameters file not found: {hyperparams_path}")
        metadata["hyperparameters"] = {}  # Ensure hyperparameters key exists

    # Add summary metrics
    if os.path.exists(summary_metrics_path):
        with open(summary_metrics_path, "r") as f:
            summary_metrics = json.load(f)
            metadata["summary_metrics"] = summary_metrics
    else:
        print(f"Warning: Summary metrics file not found: {summary_metrics_path}")

    # Add all metrics
    if os.path.exists(all_metrics_path):
        with open(all_metrics_path, "r") as f:
            all_metrics = json.load(f)
            metadata["all_metrics"] = all_metrics
    else:
        print(f"Warning: All metrics file not found: {all_metrics_path}")

    # Add model directory info
    metadata["model_dir"] = str(model_dir)
    metadata["version"] = os.path.basename(model_dir)

    return metadata


def load_debrim_metadata(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
) -> dict[str, Any]:
    """
    Load only the metadata from a specific version directory.

    Parameters
    ----------
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories

    Returns
    -------
    dict
        Metadata dictionary containing model information and performance metrics
    """
    # Get model directory path
    model_dir = _resolve_model_directory(version, base_dir)

    # Load and return metadata
    return _load_model_metadata(model_dir)


def load_debrim_model_from_version(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
    device: torch.device | None = None,
) -> tuple[DebrimModel, dict[str, dict[str, int]], dict[str, Any]]:
    """
    Load a complete DebrimModel from a specific version directory.

    Parameters
    ----------
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories
    device : torch.device, optional
        Device to load the model onto (if None, uses best available)

    Returns
    -------
    tuple
        (model, vocab_dict, metadata) where metadata contains model info and metrics
    """
    # Check if GPU acceleration is available
    if device is None:
        device = get_best_available_device()

    # Get model directory path
    model_dir = _resolve_model_directory(version, base_dir)

    # Define standard file paths
    model_path = os.path.join(model_dir, "debrim_model.pt")
    vocab_path = os.path.join(model_dir, "vocab_dict.pkl")

    # Verify required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    # Load vocabulary dictionary
    with open(vocab_path, "rb") as f:
        vocab_dict = pickle.load(f)

    # Load metadata (includes hyperparameters)
    metadata = _load_model_metadata(model_dir)
    hyperparams = metadata.get("hyperparameters", {})

    # Configure model parameters
    embedding_dim = hyperparams.get("embedding_dim", 128)
    seq_pooling = hyperparams.get("seq_pooling", "mean")
    hidden_dim = hyperparams.get("hidden_dims", [128, 64])
    n_classes = hyperparams.get("n_classes", 2)
    dropout = hyperparams.get("dropout", 0.5)

    # Get scalar dimension from metadata
    scalar_dim = 0
    if "embedder_output_dim" in metadata and "classifier_input_dim" in metadata:
        scalar_dim = metadata["classifier_input_dim"] - metadata["embedder_output_dim"]

    # Create model with the right architecture
    model = DebrimModel.from_config(
        vocab_dict=vocab_dict,
        scalar_dim=scalar_dim,
        embedding_dim=embedding_dim,
        seq_pooling=seq_pooling,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        dropout=dropout,
    )

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_dir}")
    return model, vocab_dict, metadata


def load_debrim_embedder_from_version(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
    device: torch.device | None = None,
) -> tuple[DebrimEmbedder, dict[str, dict[str, int]], dict[str, Any]]:
    """
    Load only the embedder component from a specific version directory.

    Parameters
    ----------
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories
    device : torch.device, optional
        Device to load the model onto (if None, uses best available)

    Returns
    -------
    tuple
        (embedder, vocab_dict, metadata) where metadata contains model info and metrics
    """
    # Check if GPU acceleration is available
    if device is None:
        device = get_best_available_device()

    # Get model directory path
    model_dir = _resolve_model_directory(version, base_dir)

    # Define standard file paths
    embedder_path = os.path.join(model_dir, "debrim_embedder.pt")
    vocab_path = os.path.join(model_dir, "vocab_dict.pkl")

    # Verify required files exist
    if not os.path.exists(embedder_path):
        raise FileNotFoundError(f"Embedder file not found: {embedder_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    # Load vocabulary dictionary
    with open(vocab_path, "rb") as f:
        vocab_dict = pickle.load(f)

    # Load metadata (includes hyperparameters)
    metadata = _load_model_metadata(model_dir)
    hyperparams = metadata.get("hyperparameters", {})

    # Create embedder
    embedding_dim = hyperparams.get("embedding_dim", 128)
    seq_pooling = hyperparams.get("seq_pooling", "mean")

    embedder = DebrimEmbedder(
        vocab_dict=vocab_dict, embedding_dim=embedding_dim, seq_pooling=seq_pooling
    )

    # Load embedder weights
    embedder.load_state_dict(torch.load(embedder_path, map_location=device))
    embedder = embedder.to(device)
    embedder.eval()

    print(f"Embedder loaded from {model_dir}")
    return embedder, vocab_dict, metadata
