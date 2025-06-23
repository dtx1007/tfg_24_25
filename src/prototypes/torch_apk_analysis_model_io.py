import os
import json
import torch
import pickle
import joblib
import numpy as np

from dataclasses import asdict
from datetime import datetime

from .torch_apk_analysis_model import (
    APKAnalysisModel,
    APKFeatureEmbedder,
    NNHyperparams,
    get_best_available_device,
)

from torchtext.vocab import Vocab

from typing import Any
from pathlib import Path

from sklearn.preprocessing import StandardScaler


def _convert_numpy_to_list(data: Any) -> Any:
    """
    Recursively convert numpy arrays to lists in data structures.
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
    model: APKAnalysisModel,
    vocab_dict: dict[str, Vocab],
    hyperparams: NNHyperparams,
    scalers: dict[str, StandardScaler],
    results: dict[str, Any] | None = None,
    save_dir: Path | str = "./model_artifacts/nn_models",
) -> dict[str, str]:
    """
    Save a trained neural network model with all its components, metadata, and scalers.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    paths = {
        "model_dir": model_dir,
        "full_model": os.path.join(model_dir, "apk_analysis_model.pt"),
        "embedder": os.path.join(model_dir, "apk_feature_embedder.pt"),
        "classifier": os.path.join(model_dir, "apk_classifier.pt"),
        "vocab": os.path.join(model_dir, "vocab_dict.pkl"),
        "scalers": os.path.join(model_dir, "scalers.joblib"),
        "hyperparams": os.path.join(model_dir, "hyperparams.json"),
        "metadata": os.path.join(model_dir, "metadata.json"),
        "summary_metrics": os.path.join(model_dir, "summary_metrics.json"),
        "all_metrics": os.path.join(model_dir, "all_metrics.json"),
    }

    torch.save(model.state_dict(), paths["full_model"])
    torch.save(model.embedder.state_dict(), paths["embedder"])
    torch.save(model.classifier.state_dict(), paths["classifier"])

    with open(paths["vocab"], "wb") as f:
        pickle.dump(vocab_dict, f)

    joblib.dump(scalers, paths["scalers"])
    print(f"Scalers saved to {paths['scalers']}")

    if isinstance(hyperparams, dict):
        hyperparams_dict = hyperparams
    else:
        hyperparams_dict = asdict(hyperparams)

    with open(paths["hyperparams"], "w") as f:
        json.dump(hyperparams_dict, f, indent=2)

    # Prepare metadata about the model architecture
    metadata = {
        "timestamp": timestamp,
        "model_type": "APKAnalysisModel",
        "pytorch_version": torch.__version__,
        "feature_count": len(vocab_dict),
        "model_architecture": {
            "embedder_output_dim": model.embedder.output_dim,
            "classifier_input_dim": model.classifier.input_dim,
            "scalar_features": model.embedder.scalar_cols,
            "sequence_cols": model.embedder.sequence_cols,
            "char_cols": model.embedder.char_cols,
            "vector_cols": model.embedder.vector_cols,
            "vector_dims": model.embedder.vector_dims,
        },
        "paths": {k: os.path.basename(v) for k, v in paths.items()},
        "model_dir": model_dir,
    }

    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    if results:
        all_metrics = _convert_numpy_to_list(results.copy())

        summary_metrics = {
            key: value
            for key, value in all_metrics.items()
            if key.startswith("mean_") or key.startswith("std_")
        }

        with open(paths["summary_metrics"], "w") as f:
            json.dump(summary_metrics, f, indent=2)

        all_metrics.pop("hyperparams", None)

        with open(paths["all_metrics"], "w") as f:
            json.dump(all_metrics, f, indent=2)

    version_file = os.path.join(save_dir, "latest_version.txt")
    with open(version_file, "w") as f:
        f.write(timestamp)

    print(f"Model and artifacts saved to {model_dir}")
    return paths


def _resolve_model_directory(version: str | None, base_dir: Path | str) -> str:
    """
    Resolve the model directory path based on version specification.
    """
    if version is not None:
        model_dir = os.path.join(base_dir, version)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model version directory not found: {model_dir}")
        return model_dir

    version_file = os.path.join(base_dir, "latest_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            latest_version = f.read().strip()
            model_dir = os.path.join(base_dir, latest_version)
            if os.path.isdir(model_dir):
                print(f"Loading latest model version: {latest_version}")
                return model_dir

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    timestamp_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()
    ]

    if not timestamp_dirs:
        raise FileNotFoundError(f"No model versions found in {base_dir}")

    latest_version = sorted(timestamp_dirs)[-1]
    model_dir = os.path.join(base_dir, latest_version)
    print(f"Loading latest model version: {latest_version}")
    return model_dir


def _load_model_metadata(model_dir: str) -> dict[str, Any]:
    """
    Load and combine all metadata files from a model directory.
    """
    metadata_path = os.path.join(model_dir, "metadata.json")
    hyperparams_path = os.path.join(model_dir, "hyperparams.json")
    summary_metrics_path = os.path.join(model_dir, "summary_metrics.json")
    all_metrics_path = os.path.join(model_dir, "all_metrics.json")

    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        print(f"Warning: Metadata file not found: {metadata_path}")

    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
            metadata["hyperparameters"] = hyperparams
    else:
        print(f"Warning: Hyperparameters file not found: {hyperparams_path}")
        metadata["hyperparameters"] = {}

    if os.path.exists(summary_metrics_path):
        with open(summary_metrics_path, "r") as f:
            summary_metrics = json.load(f)
            metadata["summary_metrics"] = summary_metrics
    else:
        print(f"Warning: Summary metrics file not found: {summary_metrics_path}")

    if os.path.exists(all_metrics_path):
        with open(all_metrics_path, "r") as f:
            all_metrics = json.load(f)
            metadata["all_metrics"] = all_metrics
    else:
        print(f"Warning: All metrics file not found: {all_metrics_path}")

    metadata["model_dir"] = str(model_dir)
    metadata["version"] = os.path.basename(model_dir)

    return metadata


def load_apk_analysis_model_metadata(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
) -> dict[str, Any]:
    """
    Load only the metadata from a specific version directory.
    """
    model_dir = _resolve_model_directory(version, base_dir)
    return _load_model_metadata(model_dir)


def load_apk_analysis_model_from_version(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
    device: torch.device | None = None,
) -> tuple[
    APKAnalysisModel, dict[str, Vocab], dict[str, StandardScaler], dict[str, Any]
]:
    """
    Load a complete APKAnalysisModel, its vocabulary, scalers, and metadata from a specific version directory.
    Returns:
    --------
    tuple
        (model, vocab_dict, scalers, metadata)
    """
    if device is None:
        device = get_best_available_device()

    model_dir = _resolve_model_directory(version, base_dir)
    model_path = os.path.join(model_dir, "apk_analysis_model.pt")
    vocab_path = os.path.join(model_dir, "vocab_dict.pkl")
    scalers_path = os.path.join(model_dir, "scalers.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with open(vocab_path, "rb") as f:
        vocab_dict = pickle.load(f)

    if not os.path.exists(scalers_path):
        raise FileNotFoundError(
            f"Scalers file not found at {scalers_path}. Model cannot be loaded without its scalers."
        )
    try:
        loaded_scalers = joblib.load(scalers_path)
        print(f"Scalers loaded from {scalers_path}")
    except Exception as e:
        raise IOError(f"Could not load scalers from {scalers_path}. Error: {e}")

    metadata = _load_model_metadata(model_dir)
    hyperparams = metadata.get("hyperparameters", {})
    model_arch = metadata.get("model_architecture", {})

    # Extract configuration parameters with defaults
    embedding_dim = hyperparams.get("embedding_dim", 128)
    seq_pooling = hyperparams.get("seq_pooling", "mean")
    hidden_dim = hyperparams.get(
        "hidden_dims", [128, 64]
    )  # Ensure this key matches NNHyperparams
    n_classes = hyperparams.get("n_classes", 2)
    dropout = hyperparams.get("dropout", 0.5)

    # Extract new structure parameters
    sequence_cols = model_arch.get("sequence_cols", [])
    char_cols = model_arch.get("char_cols", [])
    vector_cols = model_arch.get("vector_cols", [])
    vector_dims = model_arch.get("vector_dims", {})

    # Handle scalar_features metadata, which could be a list of names (new format) or a count (old format).
    scalar_features_meta = model_arch.get("scalar_features", [])
    if isinstance(scalar_features_meta, int):
        print(
            "Warning: Loading model with old metadata format (scalar feature count only)."
        )
        instantiation_scalar_cols = [f"scalar_{i}" for i in range(scalar_features_meta)]
    elif isinstance(scalar_features_meta, list):
        instantiation_scalar_cols = scalar_features_meta
    else:
        print(
            f"Warning: 'scalar_features' in metadata is of unexpected type: {type(scalar_features_meta)}. Assuming no scalar features."
        )
        instantiation_scalar_cols = []

    # Create model with the right architecture including new parameters
    model = APKAnalysisModel.from_config(
        vocab_dict=vocab_dict,
        sequence_cols=sequence_cols,
        scalar_cols=instantiation_scalar_cols,  # Pass the list of names
        char_cols=char_cols,
        vector_cols=vector_cols,
        vector_dims=vector_dims,
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
    return model, vocab_dict, loaded_scalers, metadata


def load_apk_feature_embedder_from_version(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/nn_models",
    device: torch.device | None = None,
) -> tuple[
    APKFeatureEmbedder, dict[str, Vocab], dict[str, StandardScaler], dict[str, Any]
]:
    """
    Load only the embedder component, vocab, scalers (if present), and metadata from a specific version directory.
    """
    if device is None:
        device = get_best_available_device()

    model_dir = _resolve_model_directory(version, base_dir)
    embedder_path = os.path.join(model_dir, "apk_feature_embedder.pt")
    vocab_path = os.path.join(model_dir, "vocab_dict.pkl")
    scalers_path = os.path.join(model_dir, "scalers.joblib")

    if not os.path.exists(embedder_path):
        raise FileNotFoundError(f"Embedder file not found: {embedder_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with open(vocab_path, "rb") as f:
        vocab_dict = pickle.load(f)

    if not os.path.exists(scalers_path):
        raise FileNotFoundError(
            f"Scalers file not found at {scalers_path}. Cannot proceed without scalers."
        )
    try:
        loaded_scalers = joblib.load(scalers_path)
        print(
            f"Scalers loaded from {scalers_path} (for context, though embedder doesn't use them directly)"
        )
    except Exception as e:
        raise IOError(f"Could not load scalers from {scalers_path}. Error: {e}")

    metadata = _load_model_metadata(model_dir)
    hyperparams = metadata.get("hyperparameters", {})
    model_arch = metadata.get("model_architecture", {})

    # Extract configuration parameters
    embedding_dim = hyperparams.get("embedding_dim", 128)
    seq_pooling = hyperparams.get("seq_pooling", "mean")

    # Extract new structure parameters
    sequence_cols = model_arch.get("sequence_cols", [])
    char_cols = model_arch.get("char_cols", [])
    vector_cols = model_arch.get("vector_cols", [])
    vector_dims = model_arch.get("vector_dims", {})

    # Create embedder with updated architecture
    embedder = APKFeatureEmbedder(
        vocab_dict=vocab_dict,
        sequence_cols=sequence_cols,
        embedding_dim=embedding_dim,
        seq_pooling=seq_pooling,
        char_cols=char_cols,
        vector_cols=vector_cols,
        vector_dims=vector_dims,
    )

    # Load embedder weights
    embedder.load_state_dict(torch.load(embedder_path, map_location=device))
    embedder = embedder.to(device)
    embedder.eval()

    print(f"Embedder loaded from {model_dir}")
    return embedder, vocab_dict, loaded_scalers, metadata
