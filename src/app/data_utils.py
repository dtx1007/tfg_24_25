import streamlit as st
import pandas as pd
from pathlib import Path

from src.app.model_utils import full_preprocess_and_predict


@st.cache_data
def load_background_df(path: Path) -> pd.DataFrame | None:
    """Loads the pre-computed background dataframe for SHAP analysis."""
    if not path.exists():
        st.error(f"Background data file not found at: {path}")
        st.warning(
            "Please run the `model_interpretability.ipynb` notebook to generate `df_background.pic`."
        )
        return None

    df = pd.read_pickle(path)

    for col in df.columns:
        if df[col].apply(type).eq(list).any():
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    return df


@st.cache_data
def get_background_embeddings(
    _nn_model, _vocab_dict, _scalers, _metadata, _device, df_background
):
    """Processes the background dataframe and extracts embeddings."""
    with st.spinner("Processing background data for explainer context..."):
        _, _, background_embeddings, df_background_processed = (
            full_preprocess_and_predict(
                df_background,
                _nn_model,
                _vocab_dict,
                _scalers,
                _metadata,
                _device,
                preprocess=False,
            )
        )
        return background_embeddings, df_background_processed
