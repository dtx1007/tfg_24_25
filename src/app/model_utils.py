import streamlit as st
import numpy as np

from pathlib import Path

from src.prototypes.torch_apk_analysis_model import (
    extract_embeddings,
    get_best_available_device,
    predict,
)
from src.prototypes.torch_apk_analysis_model_io import (
    load_apk_analysis_model_from_version,
)
from src.prototypes.ml_model_io import load_ml_models_from_version
from src.utils.preprocessing_utils import (
    apply_scalers_to_dataframe,
    preprocess_data_for_nn,
)


@st.cache_resource
def load_nn_model_from_disk(version=None, base_dir="model_artifacts/nn_models"):
    with st.spinner("Loading Neural Network model and artifacts..."):
        device = get_best_available_device()
        model, vocab_dict, scalers, metadata = load_apk_analysis_model_from_version(
            version=version, base_dir=base_dir, device=device
        )
        model.to(device)
        model.eval()
        st.success("Neural Network model loaded.")
        return model, vocab_dict, scalers, metadata, device


@st.cache_resource
def load_ml_models_from_disk(version=None, base_dir="model_artifacts/ml_models"):
    with st.spinner("Loading classical ML models..."):
        ml_models, ml_metadata = load_ml_models_from_version(
            version=version, base_dir=base_dir
        )
        st.success("Classical ML models loaded.")
        return ml_models, ml_metadata


# Load umap reducer
@st.cache_resource
def load_umap_reducer_from_disk(base_dir: Path | str = "model_artifacts/umap"):
    with st.spinner("Loading UMAP reducer..."):
        import joblib

        reducer_path = f"{base_dir}/umap_reducer.pkl"
        reducer = joblib.load(reducer_path)
        st.success("UMAP reducer loaded.")
        return reducer


def full_preprocess_and_predict(
    df_raw,
    nn_model,
    vocab_dict,
    scalers,
    metadata,
    device,
    preprocess: bool = True,
):
    """Full pipeline: Preprocesses raw data, gets NN prediction, and extracts embeddings."""
    arch = metadata.get("model_architecture", {})
    sequence_cols = arch.get("sequence_cols", [])
    char_cols = arch.get("char_cols", [])
    vector_cols = arch.get("vector_cols", [])
    scalar_cols = arch.get("scalar_features", [])
    vector_dims = arch.get("vector_dims", {})
    max_lengths = metadata.get("max_lengths")

    if preprocess:
        df_pre_processed = df_raw.copy()
        for col in sequence_cols:
            if col in df_pre_processed.columns:
                df_pre_processed[col] = df_pre_processed[col].apply(
                    lambda x: str(list(x))
                    if isinstance(x, (list, np.ndarray))
                    else str(x)
                )

        df_tokenized, _ = preprocess_data_for_nn(
            df_pre_processed,
            sequence_cols,
            char_cols,
            vector_cols,
            scalar_cols,
            vector_dims,
            vocab_dict=vocab_dict,
            max_lengths=max_lengths,
        )

        df_scaled, _ = apply_scalers_to_dataframe(
            df_tokenized,
            scalar_cols=scalar_cols,
            vector_cols=vector_cols,
            scalers=scalers,
            fit_scalers=False,
        )
    else:
        df_scaled = df_raw

    embeddings, _ = extract_embeddings(
        model=nn_model,
        df=df_scaled,
        scalers=scalers,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        device=device,
        batch_size=df_scaled.shape[0],
    )

    predictions_nn, probabilities_nn = predict(
        model=nn_model,
        df=df_scaled,
        scalers=scalers,
        sequence_cols=sequence_cols,
        scalar_cols=scalar_cols,
        char_cols=char_cols,
        vector_cols=vector_cols,
        device=device,
        batch_size=df_scaled.shape[0],
    )

    return predictions_nn, probabilities_nn, embeddings, df_scaled


def get_ml_predictions(ml_models, embeddings):
    """Gets predictions from classical ML models."""
    predictions = {}
    for name, model in ml_models.items():
        try:
            pred = model.predict(embeddings)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(embeddings)
            else:  # Handle models like SVM without predict_proba
                proba = np.array([[1.0 - p, p] for p in pred])
            predictions[name] = (pred[0], proba[0])
        except Exception as e:
            st.warning(f"Could not get prediction for {name}: {e}")
    return predictions
