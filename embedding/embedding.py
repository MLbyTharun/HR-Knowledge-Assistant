import streamlit as st
from langchain_huggingface import HuggingFaceEndpointEmbeddings


@st.cache_resource
def load_embeddings(hf_token: str):
    """Cloud-hosted embeddings (HuggingFace Inference API) — no local torch/GPU needed."""
    if not hf_token or not hf_token.strip():
        raise ValueError("HuggingFace token is missing. Add it in the sidebar (free at huggingface.co/settings/tokens).")
    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-base-en-v1.5",  # Same model as before, now served remotely
        huggingfacehub_api_token=hf_token.strip(),
    )
