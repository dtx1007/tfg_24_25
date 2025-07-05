import numpy as np
import pandas as pd
import streamlit as st
import shap
import torch


def analyze_embeddings_with_shap(
    embeddings_to_explain: np.ndarray,
    background_embeddings: np.ndarray,
    classifier_prediction_fn,
) -> shap.Explanation:
    """Analyzes a classifier's behavior on an embedding space using SHAP."""
    explainer = shap.KernelExplainer(classifier_prediction_fn, background_embeddings)
    explainer_obj = explainer(embeddings_to_explain)
    return explainer_obj


def aggregate_embedding_shap(
    embedding_explanation: shap.Explanation,
    metadata: dict,
) -> shap.Explanation:
    """Aggregates SHAP values from the embedding space back to original features."""
    arch = metadata.get("model_architecture", {})
    seq_cols = arch.get("sequence_cols", [])
    char_cols = arch.get("char_cols", [])
    vector_cols = arch.get("vector_cols", [])
    scalar_cols = arch.get("scalar_features", [])
    embedding_dim = metadata.get("hyperparameters", {}).get("embedding_dim", 128)

    original_feature_names = seq_cols + char_cols + vector_cols + scalar_cols
    n_samples, n_features, n_classes = embedding_explanation.shape
    n_original_features = len(original_feature_names)

    aggregated_shap_matrix = np.zeros((n_samples, n_original_features, n_classes))
    aggregated_data_matrix = np.zeros((n_samples, n_original_features))

    current_index = 0
    feature_map = (
        [(name, embedding_dim) for name in seq_cols]
        + [(name, embedding_dim) for name in char_cols]
        + [(name, embedding_dim) for name in vector_cols]
        + [(name, 1) for name in scalar_cols]
    )

    for i, (name, width) in enumerate(feature_map):
        feature_slice = slice(current_index, current_index + width)
        for c in range(n_classes):
            aggregated_shap_matrix[:, i, c] = embedding_explanation.values[
                :, feature_slice, c
            ].sum(axis=1)
        aggregated_data_matrix[:, i] = embedding_explanation.data[
            :, feature_slice
        ].mean(axis=1)
        current_index += width

    return shap.Explanation(
        values=aggregated_shap_matrix,
        base_values=embedding_explanation.base_values,
        data=aggregated_data_matrix,
        feature_names=original_feature_names,
    )


@st.cache_resource
def get_background_explainer_and_explanations(
    model_name: str,
    _model_to_explain,
    _background_embeddings: np.ndarray,
    _nn_metadata: dict,
    _device,
    _df_background_is_malware: pd.Series,
):
    """Creates a SHAP explainer and pre-calculates explanations for the background dataset."""
    if model_name == "Neural Network":

        def nn_prediction_function(embeddings_numpy):
            embeddings_tensor = torch.from_numpy(embeddings_numpy).to(
                _device, dtype=torch.float32
            )
            with torch.no_grad():
                logits = _model_to_explain(embeddings_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

        prediction_fn = nn_prediction_function
    else:
        prediction_fn = _model_to_explain.predict_proba

    sampled_background_embeddings = shap.sample(_background_embeddings, 100)
    explainer = shap.KernelExplainer(prediction_fn, sampled_background_embeddings)
    embedding_explanation = explainer(_background_embeddings)
    agg_explanation = aggregate_embedding_shap(embedding_explanation, _nn_metadata)
    expl_global = agg_explanation[:, :, 1]

    return explainer, expl_global


def explain_single_instance(explainer, instance_embedding, nn_metadata):
    """Calculates the SHAP explanation for a single new instance using a pre-built explainer."""
    embedding_explanation = explainer(instance_embedding)
    agg_explanation = aggregate_embedding_shap(embedding_explanation, nn_metadata)
    expl_instance = agg_explanation[:, :, 1]
    return expl_instance[0]
