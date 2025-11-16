from services.data_loader import build_index, search

index = None
docs = None
embedder = None

async def preload_rag():
    global index, docs, embedder
    if index is None:
        index, docs, embedder = build_index(embedder_type="openai")
        print("📦 RAG index preloaded.")

async def run_rag(query: str):
    global index, docs, embedder
    results = search(query, index, docs, embedder)
    return results[0]["text"] if results else "No results."
