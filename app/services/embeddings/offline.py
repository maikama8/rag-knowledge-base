"""Deterministic hashing embeddings for offline retrieval.

This is a bag-of-words feature hash, not a semantic model. Passages that share
words with the question rank higher. It needs no API key and no download.
"""

import hashlib
import math

from app.services.text import content_tokens


class OfflineEmbeddingProvider:
    """Hash content tokens into a fixed vector and L2-normalize it."""

    name = "offline"

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 32:
            raise ValueError("dimension must be at least 32.")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in content_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
