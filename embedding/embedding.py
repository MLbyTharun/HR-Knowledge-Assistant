import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Required for BGE models
    )
