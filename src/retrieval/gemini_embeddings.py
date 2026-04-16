from interfaces.baseEmbeddingsFunc import BaseEmbeddingsFunc
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import time

class GeminiEmbeddingsFunc(BaseEmbeddingsFunc):
    def __init__(self, config):
        self.config = config
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.gemini_model_name,
            google_api_key=self.config.gemini_api_key,
        )

    def embeddingsFunc(self):
        return self._embeddings

    def encode(self, sentences):
        print(f"DEBUG: Atomicly embedding {len(sentences)} sentences using {self.config.gemini_model_name}...")
        
        all_vectors = []
        for i, sentence in enumerate(sentences):
            try:
                if i > 0:
                    time.sleep(0.6)
                
                vector = self._embeddings.embed_query(sentence)
                all_vectors.append(vector)
            except Exception as e:
                if "429" in str(e):
                    print(f"DEBUG: Rate limit hit at sentence {i}, waiting 60s as requested by API...")
                    time.sleep(60)
                    vector = self._embeddings.embed_query(sentence)
                    all_vectors.append(vector)
                else:
                    print(f"ERROR at sentence {i}: {e}")
                    raise e
                    
            if (i + 1) % 20 == 0:
                print(f"DEBUG: Progress: {i + 1}/{len(sentences)} sentences encoded.")
        return np.array(all_vectors)