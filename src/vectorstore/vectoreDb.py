from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from interfaces.baseVectorDb import BaseVectorDb

class ChromaDb(BaseVectorDb):

    def __init__(self, config, embeddings_wrapper):
        self.config = config
        
        class EmbeddingWrapper:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            def embed_documents(self, texts):
                return self.wrapper.encode(texts).tolist()
            def embed_query(self, text):
                return self.wrapper.encode([text])[0].tolist()

        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=EmbeddingWrapper(embeddings_wrapper),
            persist_directory=self.config.persist_directory,
        )   

    def add_chunks(self, chunks):
        documents = []
        for chunk in chunks:
            document = Document(
                page_content=" ".join(chunk),
                metadata={},
                id=str(uuid4()),
            )
            documents.append(document)

        self.vector_store.add_documents(documents)

    def query(self, query):
        return self.vector_store.similarity_search(query)
