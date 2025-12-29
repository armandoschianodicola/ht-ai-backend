from openai import AsyncOpenAI

from config.config import settings

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY
)

def build_context(results):
    context_chunks = []
    for i, r in enumerate(results, start=1):
        context_chunks.append(
            f"Source {i} ({r['filename']}):\n{r['text']}"
        )
    return "\n\n".join(context_chunks)

async def generate_answer(query: str, results: list):
    context = build_context(results)

    prompt = f"""
You are an expert assistant.

Answer the user's question using ONLY the information from the sources below.
If the sources are insufficient, say so explicitly.

Question:
{query}

Sources:
{context}

Answer:
"""

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are a precise and concise expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
