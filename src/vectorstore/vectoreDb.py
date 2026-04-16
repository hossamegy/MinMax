from src.interfaces.baseVectorDb import BaseVectorDb
from langchain_chroma import Chroma


class ChromaDb(BaseVectorDb):

    def __init__(self, config, embeddings):
        self.config = config
        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=self.config.persist_directory,
        )   

    def add_chunks(self, chunks):
        self.vector_store.add_documents(chunks)

    def query(self, query):
        return self.vector_store.similarity_search(query)

