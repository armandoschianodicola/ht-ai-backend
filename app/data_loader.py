import os
from pathlib import Path

import faiss
import numpy as np

from app.factories.embedder_factory import EmbedderFactory
from app.utils.config import settings

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_documents():
    found_docs = []
    for filename in os.listdir(os.path.join(ROOT_DIR, settings.DOCS_PATH)):
        if filename.endswith(".txt"):
            with open(os.path.join(settings.DOCS_PATH, filename), "r") as f:
                found_docs.append({"text": f.read(), "source": filename})
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
    index, docs, embedder = build_index(embedder_type="sentence")

    q = "What is this project about?"
    res = search(q, index, docs, embedder)
    print("🔎 Search results:")
    for r in res:
        print(f"→ {r['filename']} ({r['distance']:.4f})")
