from abc import ABC, abstractmethod


class BaseEmbeddingsFunc(ABC):
    @abstractmethod
    def embeddingsFunc(self):
        pass

    @abstractmethod
    def encode(self, sentences):
        pass