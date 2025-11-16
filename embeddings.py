from openai import OpenAI
import numpy as np

from utils.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_text(text: str):
    resp = client.embeddings.create(
        input=text,
        model=settings.EMBEDDING_MODEL
    )
    return np.array(resp.data[0].embedding)
