# import numpy as np
# from sentence_transformers import SentenceTransformer
# from app.interfaces.embedder import EmbedderInterface
#
#
# class SentenceTransformerEmbedder(EmbedderInterface):
#     """Local embedder using SentenceTransformers."""
#
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#
#     def embed(self, text: str):
#         embedding = self.model.encode(text, convert_to_numpy=True)
#         return np.array(embedding, dtype="float32")
