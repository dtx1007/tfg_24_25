import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


def _convert_numpy_to_list(data: Any) -> Any:
    """
    Recursively convert numpy arrays to lists in data structures.

    This is necessary for JSON serialization of dictionaries or 
    lists containing numpy arrays.

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


def save_ml_models_with_metadata(
    models: dict[str, Any],
    results: dict[str, dict[str, Any]],
    save_dir: Path | str = "./model_artifacts/ml_models",
    save_best_only: bool = False,
    scoring_metric: Literal[
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ] = "f1",
) -> dict[str, str]:
    """
    Save trained machine learning models along with their evaluation metrics.

    Creates a timestamped directory containing all model artifacts for version control
    and reproducibility.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model objects
    results : dict
        Nested dictionary containing evaluation metrics from cross-validation
    save_dir : Path or str
        Directory to save the model artifacts
    save_best_only : bool
        When True, saves only the best performing model based on scoring_metric
        When False, saves all models in the models dictionary
    scoring_metric : str
        Metric used to determine the best model when save_best_only=True
        Options: "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"

    Returns
    -------
    dict
        Dictionary of saved file paths for all artifacts
    """
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped directory
    version_dir = os.path.join(save_dir, timestamp)
    os.makedirs(version_dir, exist_ok=True)

    # Determine which models to save based on save_best_only
    if save_best_only:
        # Find the best model according to the specified scoring metric
        best_model_name = None
        best_score = -float("inf")

        for model_name, model_results in results.items():
            metric_key = f"mean_{scoring_metric}"
            if metric_key in model_results and model_results[metric_key] > best_score:
                best_score = model_results[metric_key]
                best_model_name = model_name

        if best_model_name:
            models_to_save = {best_model_name: models[best_model_name]}
            results_to_save = {best_model_name: results[best_model_name]}
        else:
            # Fallback to saving all models if best model can't be determined
            models_to_save = models
            results_to_save = results
    else:
        # Save all models
        models_to_save = models
        results_to_save = results

    # Prepare path dictionary for return value
    saved_paths = {"version_dir": version_dir}
    
    # Initialize consolidated metadata
    all_models_metadata = {
        "timestamp": timestamp,
        "models": {},
        "save_best_only": save_best_only,
        "scoring_metric": scoring_metric if save_best_only else None
    }    # Process and save each model
    for model_name, model in models_to_save.items():
        # Create subdirectory for this specific model
        model_dir = os.path.join(version_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Define paths for all model artifacts
        model_path = os.path.join(model_dir, "model.pkl")
        summary_metrics_path = os.path.join(model_dir, "summary_metrics.json")
        all_metrics_path = os.path.join(model_dir, "all_metrics.json")
        hyperparams_path = os.path.join(model_dir, "hyperparams.json")

        # Save the model using pickle serialization
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Extract metrics for this model
        model_results = results_to_save[model_name]
        
        # Separate summary metrics (means and standard deviations) from all metrics
        summary_metrics = {}
        all_metrics = {}
        
        for metric_name, metric_value in model_results.items():
            # Add to summary metrics if it's a mean or std metric
            if metric_name.startswith("mean_") or metric_name.startswith("std_"):
                summary_metrics[metric_name] = metric_value
            
            # Copy all metrics except hyperparams (saved separately)
            if metric_name != "hyperparams":
                all_metrics[metric_name] = metric_value

        # Convert numpy arrays to lists for JSON serialization
        all_metrics = _convert_numpy_to_list(all_metrics)

        # Save metrics to JSON files
        with open(summary_metrics_path, "w") as f:
            json.dump(summary_metrics, f, indent=2)

        with open(all_metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Save hyperparameters separately if available
        hyperparams = model_results.get("hyperparams")
        if hyperparams:
            with open(hyperparams_path, "w") as f:
                json.dump(hyperparams, f, indent=2)

        # Track paths for this model
        saved_paths[model_name] = {
            "model": model_path,
            "summary_metrics": summary_metrics_path,
            "all_metrics": all_metrics_path,
            "hyperparams": hyperparams_path if hyperparams else None,
        }

        # Add to consolidated metadata
        all_models_metadata["models"][model_name] = {
            "paths": {
                "model": os.path.relpath(model_path, version_dir),
                "summary_metrics": os.path.relpath(summary_metrics_path, version_dir),
                "all_metrics": os.path.relpath(all_metrics_path, version_dir),
                "hyperparams": os.path.relpath(hyperparams_path, version_dir) if hyperparams else None,
            },
            "summary_metrics": {
                metric: model_results[f"mean_{metric}"]
                for metric in ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
                if f"mean_{metric}" in model_results
            },
            "model_type": type(model).__name__,
        }    # Save consolidated metadata with information about all models
    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_models_metadata, f, indent=2)

    # Add metadata path to results
    saved_paths["metadata"] = metadata_path

    # Create a version file to track the latest model version
    version_file = os.path.join(save_dir, "latest_version.txt")
    with open(version_file, "w") as f:
        f.write(timestamp)

    # Print summary of saved models
    print(f"Saved {len(models_to_save)} model(s) to {version_dir}")
    for model_name in models_to_save:
        print(f"  - {model_name}")

    return saved_paths


def _resolve_ml_model_directory(version: str | None, base_dir: Path | str) -> str:
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
        raise FileNotFoundError(f"No model versions found in {base_dir}")

    # Sort directories by name (timestamp format ensures chronological order)
    latest_version = sorted(timestamp_dirs)[-1]
    model_dir = os.path.join(base_dir, latest_version)
    print(f"Loading latest model version: {latest_version}")
    return model_dir


def _load_ml_model_metadata(
    version_dir: str, model_name: str | None = None
) -> dict[str, Any]:
    """
    Helper function to load metadata for ML models from a version directory.

    Parameters
    ----------
    version_dir : str
        Path to the version directory containing model artifacts
    model_name : str, optional
        If provided, returns metadata for only this specific model
        If None, returns metadata for all models in the version

    Returns
    -------
    dict
        Comprehensive metadata dictionary containing model information and performance metrics
    """
    # Load version metadata file
    metadata_path = os.path.join(version_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        version_metadata = json.load(f)

    # Add version directory info
    version_metadata["version_dir"] = str(version_dir)
    version_metadata["version"] = os.path.basename(version_dir)

    # If no specific model requested, return version metadata with all models
    if model_name is None:
        return version_metadata

    # Check if the requested model exists
    if model_name not in version_metadata["models"]:
        raise ValueError(
            f"Model '{model_name}' not found in version {os.path.basename(version_dir)}"
        )

    # Get model info
    model_info = version_metadata["models"][model_name]

    # Create model metadata object
    model_metadata = {
        "model_name": model_name,
        "model_type": model_info["model_type"],
        "summary_metrics": model_info["summary_metrics"],
        "version": os.path.basename(version_dir),
        "timestamp": version_metadata["timestamp"],
    }

    # Add all metrics if available
    metrics_path = os.path.join(version_dir, model_info["paths"]["all_metrics"])
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metadata["all_metrics"] = json.load(f)
    else:
        print(f"Warning: All metrics file not found: {metrics_path}")

    # Add hyperparameters if available
    if model_info["paths"]["hyperparams"]:
        hyperparams_path = os.path.join(version_dir, model_info["paths"]["hyperparams"])
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                model_metadata["hyperparams"] = json.load(f)
        else:
            print(f"Warning: Hyperparameters file not found: {hyperparams_path}")
            model_metadata["hyperparams"] = {}  # Ensure hyperparams key exists

    return model_metadata


def load_ml_models_from_version(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/ml_models",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load all ML models from a specific version directory.

    Parameters
    ----------
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories

    Returns
    -------
    tuple
        (loaded_models, metadata)
        - loaded_models: Dictionary mapping model names to loaded model objects
        - metadata: Dictionary containing version metadata
    """
    # Get model directory path
    version_dir = _resolve_ml_model_directory(version, base_dir)

    # Load version metadata
    metadata = _load_ml_model_metadata(version_dir)

    # Load all models in the version directory
    loaded_models = {}

    for model_name, model_info in metadata["models"].items():
        model_path = os.path.join(version_dir, model_info["paths"]["model"])

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found for {model_name}: {model_path}")
            continue

        with open(model_path, "rb") as f:
            loaded_models[model_name] = pickle.load(f)

    print(f"Loaded {len(loaded_models)} model(s) from {version_dir}")
    for name in loaded_models:
        print(f"  - {name}")

    return loaded_models, metadata


def load_ml_model_from_version(
    model_name: str,
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/ml_models",
) -> tuple[Any, dict[str, Any]]:
    """
    Load a specific ML model from a version directory.

    Parameters
    ----------
    model_name : str
        Name of the model to load
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories

    Returns
    -------
    tuple
        (model, model_metadata)
        - model: The loaded model object
        - model_metadata: Dictionary containing model metadata
    """
    # Get model directory path
    version_dir = _resolve_ml_model_directory(version, base_dir)

    # Load model metadata
    model_metadata = _load_ml_model_metadata(version_dir, model_name)

    # Get model info from version metadata
    with open(os.path.join(version_dir, "metadata.json"), "r") as f:
        version_metadata = json.load(f)

    model_info = version_metadata["models"][model_name]

    # Load model
    model_path = os.path.join(version_dir, model_info["paths"]["model"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded model '{model_name}' from {version_dir}")
    return model, model_metadata


def load_ml_version_metadata(
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/ml_models",
) -> dict[str, Any]:
    """
    Load metadata for all models in a version directory.

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
        Metadata dictionary containing information about all models in the version
    """
    # Get model directory path
    version_dir = _resolve_ml_model_directory(version, base_dir)

    # Load and return metadata for all models
    return _load_ml_model_metadata(version_dir)


def load_ml_model_metadata(
    model_name: str,
    version: str | None = None,
    base_dir: Path | str = "./model_artifacts/ml_models",
) -> dict[str, Any]:
    """
    Load metadata for a specific model in a version directory.

    Parameters
    ----------
    model_name : str
        Name of the model to load metadata for
    version : str, optional
        Timestamp version to load (e.g., '20250516_123456')
        If None, will load the latest version
    base_dir : str or Path
        Base directory containing model version directories

    Returns
    -------
    dict
        Metadata dictionary for the specific model
    """
    # Get model directory path
    version_dir = _resolve_ml_model_directory(version, base_dir)

    # Load and return metadata for specific model
    return _load_ml_model_metadata(version_dir, model_name)
