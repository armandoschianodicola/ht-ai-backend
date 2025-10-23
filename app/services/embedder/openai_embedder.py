import numpy as np
from openai import OpenAI

from app.interfaces.embedder import EmbedderInterface
from app.utils.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

class OpenaiEmbedder(EmbedderInterface):

    def embed(self, text: str):
        resp = client.embeddings.create(
            input=text,
            model=settings.EMBEDDING_MODEL
        )
        return np.array(resp.data[0].embedding)
