"""Embedding provider interface."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Map texts to equal-length vectors."""

    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` in order. An empty input returns an empty list."""
