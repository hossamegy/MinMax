import numpy as np
from scipy.special import expit
from typing import List



HARD_THR = 0.6
C = 0.9
INIT_CONST = 1.5


class MaxMinChunker:
    def __init__(
        self,
        hard_thr: float = HARD_THR,
        c: float = C,
        init_const: float = INIT_CONST,
        normalize: bool = True,
    ):
        self.model
        self.hard_thr = hard_thr
        self.c = c
        self.init_const = init_const
        self.normalize = normalize

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fast cosine similarity (assumes normalized vectors if enabled)."""
        return float(np.dot(a, b))

    def threshold(self, min_sim: float, size: int) -> float:
        """
        thr(C) = max( c * min_sim(C) * sigmoid(|C|), hard_thr )
        """
        return max(self.c * min_sim * float(expit(size)), self.hard_thr)

    def chunk(self, sentences: List[str]) -> List[List[str]]:
        if not sentences:
            return []

        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        chunks = []

        chunk_sents = []
        chunk_embs = []

        current_min_sim = 1.0

        for i in range(len(sentences)):
            emb = embeddings[i]
            sent = sentences[i]

            if i == 0:
                chunk_sents = [sent]
                chunk_embs = [emb]
                current_min_sim = 1.0
                continue

            if len(chunk_embs) == 1:
                sim = self.cosine_sim(chunk_embs[0], emb)

                if self.init_const * sim > self.hard_thr:
                    chunk_sents.append(sent)
                    chunk_embs.append(emb)
                    current_min_sim = sim
                else:
                    chunks.append(chunk_sents)
                    chunk_sents = [sent]
                    chunk_embs = [emb]
                    current_min_sim = 1.0
                continue

            sims = [self.cosine_sim(emb, e) for e in chunk_embs]
            max_s = max(sims)

            thr = self.threshold(current_min_sim, len(chunk_embs))

            if max_s > current_min_sim and max_s > thr:
                chunk_sents.append(sent)
                chunk_embs.append(emb)

                for s in sims:
                    if s < current_min_sim:
                        current_min_sim = s
            else:
                chunks.append(chunk_sents)
                chunk_sents = [sent]
                chunk_embs = [emb]
                current_min_sim = 1.0

        if chunk_sents:
            chunks.append(chunk_sents)

        return chunks
