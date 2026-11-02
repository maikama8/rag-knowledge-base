"""Environment-backed settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProviderName = Literal["offline", "openai"]
AnswerProviderName = Literal["offline", "openai"]
VectorStoreName = Literal["local"]


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment and an optional `.env` file."""

    app_name: str = "RAG Knowledge Base"
    embedding_provider: EmbeddingProviderName = "offline"
    llm_provider: AnswerProviderName = "offline"
    vector_store: VectorStoreName = "local"
    data_dir: Path = Path("./data")
    chunk_size: int = Field(default=800, ge=200, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    offline_embedding_dimensions: int = Field(default=512, ge=32, le=4096)
    openai_api_key: SecretStr | None = None
    openai_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    openai_embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    openai_embedding_dimensions: int = Field(default=1536, ge=32, le=8192)
    openai_chat_model: str = Field(default="gpt-4o-mini", min_length=1)
    openai_timeout_seconds: float = Field(default=60.0, gt=0)
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("embedding_provider", "llm_provider", "vector_store", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self

    def openai_api_key_value(self) -> str | None:
        """Return the OpenAI key, or None when it is unset or blank."""
        if self.openai_api_key is None:
            return None
        value = self.openai_api_key.get_secret_value().strip()
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
