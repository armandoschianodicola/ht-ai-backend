from services.data_loader import build_index, search
from services.prompter.prompter import generate_answer

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
    if not results:
        return "No results."
    return await generate_answer(query, results)

