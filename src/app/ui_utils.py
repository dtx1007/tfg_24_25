import streamlit as st
import pandas as pd
import shap
import umap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def st_shap(plot, height=None):
    """Helper function to display SHAP plots in Streamlit."""
    shap.initjs()
    shap_html = f'<head>{shap.getjs()}</head><body><div style="background-color: white">{plot.html()}</div></body>'
    st.components.v1.html(shap_html, height=height)


def display_list_feature(feature_name, features_dict, container):
    with container:
        title = feature_name.replace("_", " ").title()
        st.markdown(f"**{title}**")
        value = features_dict.get(feature_name, [])
        if len(value) > 0:
            st.dataframe(pd.DataFrame(value, columns=[title]), height=300)
        else:
            st.info("No data found.")


def plot_umap_projection(
    current_embedding: np.ndarray,
    background_embeddings: np.ndarray,
    background_labels: np.ndarray,
):
    """Visualizes embedding space with UMAP, highlighting the current sample."""
    with st.spinner("Reducing dimensionality with UMAP..."):
        try:
            all_data = np.vstack([current_embedding, background_embeddings])
            reducer = umap.UMAP(
                n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
            )
            embedded = reducer.fit_transform(all_data)

            fig, ax = plt.subplots(figsize=(12, 8))

            colors = [
                "#1a9850" if label == 0 else "#d73027" for label in background_labels
            ]
            ax.scatter(embedded[1:, 0], embedded[1:, 1], c=colors, alpha=0.6, s=50)

            ax.scatter(
                embedded[0, 0],
                embedded[0, 1],
                c="#3366FF",
                s=250,
                marker="*",
                edgecolor="black",
                linewidth=1.5,
                label="Current APK",
                zorder=10,
            )

            legend_elements = [
                Patch(facecolor="#1a9850", label="Benign (Background)"),
                Patch(facecolor="#d73027", label="Malware (Background)"),
                Patch(color="#3366FF", label="Current APK"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")
            ax.set_title("UMAP Projection of APK Embeddings", fontsize=16)
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Failed to generate UMAP plot: {e}")
