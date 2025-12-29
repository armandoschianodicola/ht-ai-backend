import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from config.config import settings
from factories.embedder_factory import EmbedderFactory

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_documents():
    found_docs = []
    for filename in os.listdir(os.path.join(ROOT_DIR, settings.DOCS_PATH)):
        if filename.endswith(".txt"):
            with open(os.path.join(settings.DOCS_PATH, filename), "r") as f:
                found_docs.append({"text": f.read(), "filename": filename})
    return found_docs


def build_index(embedder_type="openai"):
    """Build FAISS index using the selected embedder."""
    selected_embedder = EmbedderFactory.create(embedder_type)
    found_docs = load_documents()

    texts = [d["text"] for d in found_docs]
    embeddings = np.array([selected_embedder.embed(t) for t in texts]).astype("float32")

    dim = embeddings.shape[1]
    current_index = faiss.IndexFlatL2(dim)
    current_index.add(embeddings)

    print(f"✅ Indexed {len(found_docs)} documents using {embedder_type} embedder.")

    # --- NEW: SAVE TO DISK ---
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    faiss.write_index(current_index, f"{settings.VECTOR_DB_PATH}/index.faiss")

    metadata = [{"filename": d["filename"], "text": d["text"]} for d in found_docs]
    with open(f"{settings.VECTOR_DB_PATH}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return current_index, found_docs, selected_embedder


def search(query, index, docs, embedder, k=3):
    """Search the most similar documents."""
    query_emb = np.array([embedder.embed(query)]).astype("float32")
    distances, indices = index.search(query_emb, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        doc = docs[idx]
        results.append({"filename": doc["filename"], "text": doc["text"], "distance": float(dist)})
    return results


if __name__ == "__main__":
    # switch between "openai" and "sentence"
    index, docs, embedder = build_index(embedder_type="openai")

    q = "What is this project about?"
    res = search(q, index, docs, embedder)
    print(f"🔎 Search results for query: {q}")
    for r in res:
        print(f"→ {r['filename']} ({r['distance']:.4f})")
    q = "How weather impacts the match?"
    res = search(q, index, docs, embedder)
    print(f"🔎 Search results for query: {q}")
    for r in res:
        print(f"→ {r['filename']} ({r['distance']:.4f})")
