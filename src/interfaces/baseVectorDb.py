from abc import ABC, abstractmethod


class BaseVectorDb(ABC):

    @abstractmethod
    def query(self, query):
        pass

    @abstractmethod
    def add_chunks(self,chunks):
        pass