"""Embeddings utilities using Sentence-Transformers (Offline, Fast, Reliable)."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class MyEmbedder:
    """Offline local embedder using all-MiniLM-L6-v2 (384 dimensions)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        print("   Loading local embedding model (all-MiniLM-L6-v2)...", flush=True)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()  # 384

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts locally in batches on CPU."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        # batch_size 64 on CPU is extremely fast and light on RAM
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string -> 1-D vector."""
        return self.model.encode(query, convert_to_numpy=True)
