import os
import pickle

import faiss
import numpy as np

from config.config import settings

INDEX_PATH = f"{settings.VECTOR_DB_PATH}/index.faiss"
META_PATH = f"{settings.VECTOR_DB_PATH}/metadata.pkl"

def load_index():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError("Vector index not found. Run data_loader first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_similar(query_embedding: np.ndarray, top_k: int = 3):
    index, metadata = load_index()
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results = [metadata[i] for i in indices[0]]
    return results
