from abc import ABC, abstractmethod


class EmbedderInterface(ABC):

    @abstractmethod
    def embed(self, text: str):
        pass