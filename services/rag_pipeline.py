from factories.embedder_factory import EmbedderFactory
from services.llm import generate_answer
from services.retriever import search_similar


def run_rag(question: str) -> str:
    selected_embedder = EmbedderFactory.create("openai")
    query_emb = selected_embedder.embed(question)
    results = search_similar(query_emb, top_k=3)

    context = "\n\n".join([r["text"] for r in results])
    return generate_answer(question, context)
