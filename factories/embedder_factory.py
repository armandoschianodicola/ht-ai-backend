from services.embedder.openai_embedder import OpenaiEmbedder


class EmbedderFactory:

    @staticmethod
    def create(embedder_type: str):

        if embedder_type == 'openai':

            return OpenaiEmbedder()

        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
