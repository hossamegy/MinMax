from interfaces.baseEmbeddingsFunc import BaseEmbeddingsFunc
from retrieval.local_embeddings import LocalEmbeddingsFunc
from retrieval.gemini_embeddings import GeminiEmbeddingsFunc
from retrieval.loader import JsonLoader
from retrieval.chunker import MaxMinChunker
from config import Config
from vectorstore.vectoreDb import ChromaDb

class Container:
    """Dependency Injection Container for the VectorStore application."""
    
    def __init__(self):
        self._config = Config()
        self._embeddings = None
        self._loader = None
        self._chunker = None
        self._vectorstore = None

    def get_config(self) -> Config:
        return self._config

    def get_embeddings(self) -> BaseEmbeddingsFunc:
        if self._embeddings is None:
            if self._config.embedding_type == "gemini":
                self._embeddings = GeminiEmbeddingsFunc(self._config)
            else:
                self._embeddings = LocalEmbeddingsFunc(self._config)
        return self._embeddings

    def get_loader(self) -> JsonLoader:
        if self._loader is None:
            self._loader = JsonLoader(self._config.data_path)
        return self._loader

    def get_chunker(self) -> MaxMinChunker:
        if self._chunker is None:
            self._chunker = MaxMinChunker(
                hard_thr=self._config.hard_thr, 
                c=self._config.c, 
                init_const=self._config.init_const, 
                model=self.get_embeddings()
            )
        return self._chunker

    def get_vectorstore(self) -> ChromaDb:
        if self._vectorstore is None:
            self._vectorstore = ChromaDb(self._config, self.get_embeddings())
        return self._vectorstore

def bootstrap():
    """Wires up the dependencies and returns the ready-to-use components."""
    container = Container()
    return {
        "config": container.get_config(),
        "loader": container.get_loader(),
        "chunker": container.get_chunker(),
        "vectorstore": container.get_vectorstore()
    }
