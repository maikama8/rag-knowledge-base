"""OpenAI-compatible embeddings.

Selected with EMBEDDING_PROVIDER=openai. The request goes to
`{OPENAI_BASE_URL}/embeddings`.
"""

import httpx

from app.services.errors import ProviderError

_BATCH_SIZE = 64


class OpenAIEmbeddingProvider:
    """Embed texts with an OpenAI-compatible embeddings endpoint."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        dimension: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            for index, vector in self._embed_batch(batch):
                vectors[start + index] = vector
        if any(vector is None for vector in vectors):
            raise ProviderError("OpenAI embeddings response did not cover every input.")
        return [vector for vector in vectors if vector is not None]

    def _embed_batch(self, texts: list[str]) -> list[tuple[int, list[float]]]:
        body = {
            "model": self._model,
            "input": texts,
            "dimensions": self.dimension,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise ProviderError(
                f"OpenAI embeddings failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("OpenAI embeddings request failed.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("OpenAI embeddings returned a non-JSON response.") from exc

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ProviderError("OpenAI embeddings returned an unexpected response.")

        parsed: list[tuple[int, list[float]]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ProviderError("OpenAI embeddings returned an unexpected response.")
            try:
                index = int(row["index"])
                raw_vector = row["embedding"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("OpenAI embeddings returned an unexpected response.") from exc
            if not isinstance(raw_vector, list) or len(raw_vector) != self.dimension:
                raise ProviderError("OpenAI embeddings returned a vector with the wrong dimension.")
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as exc:
                raise ProviderError("OpenAI embeddings returned a non-numeric vector.") from exc
            parsed.append((index, vector))
        return parsed
