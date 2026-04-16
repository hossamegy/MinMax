from abc import ABC, abstractmethod


class BaseVectorDb(ABC):

    def build(self, chunks):
        pass
    @abstractmethod
    def query(self, query):
        pass

    @abstractmethod
    def add_chunks(self, embeddings, chunks):
        pass