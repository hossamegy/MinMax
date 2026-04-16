from interfaces.baseEmbeddingsFunc import BaseEmbeddingsFunc
from sentence_transformers import SentenceTransformer


class LocalEmbeddingsFunc(BaseEmbeddingsFunc):
    def __init__(self, config):
        self.config = config
        self._embeddings = SentenceTransformer(self.config.model_name)

    def embeddingsFunc(self):
        return self._embeddings

    def encode(self, sentences):
        embeddings = self._embeddings.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return embeddings
