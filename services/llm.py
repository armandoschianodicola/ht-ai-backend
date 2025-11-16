from openai import OpenAI

from config.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant.
    Use the following context to answer the question:

    Context:
    {context}

    Question:
    {query}

    Answer:"""
    completion = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()
