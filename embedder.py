from __future__ import annotations

from typing import List
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Small, fast, CPU-friendly
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = load_embedder(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)
