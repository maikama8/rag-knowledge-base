"""Build the providers selected by settings."""

from app.config import Settings
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.offline import OfflineEmbeddingProvider
from app.services.embeddings.openai import OpenAIEmbeddingProvider
from app.services.errors import ProviderConfigError
from app.services.llm.base import AnswerProvider
from app.services.llm.offline import OfflineAnswerProvider
from app.services.llm.openai import OpenAIAnswerProvider
from app.services.pipeline import KnowledgeBase
from app.services.vectorstore.base import VectorStore
from app.services.vectorstore.local import LocalVectorStore


def _require_openai_api_key(settings: Settings) -> str:
    value = settings.openai_api_key_value()
    if not value:
        raise ProviderConfigError("OPENAI_API_KEY is required when a provider is set to 'openai'.")
    return value


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider
    if provider == "offline":
        return OfflineEmbeddingProvider(dimension=settings.offline_embedding_dimensions)
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=_require_openai_api_key(settings),
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
            timeout=settings.openai_timeout_seconds,
            dimension=settings.openai_embedding_dimensions,
        )
    raise ProviderConfigError(f"Unsupported embedding provider: {provider}")


def build_answer_provider(settings: Settings) -> AnswerProvider:
    provider = settings.llm_provider
    if provider == "offline":
        return OfflineAnswerProvider()
    if provider == "openai":
        return OpenAIAnswerProvider(
            api_key=_require_openai_api_key(settings),
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
            timeout=settings.openai_timeout_seconds,
        )
    raise ProviderConfigError(f"Unsupported LLM provider: {provider}")


def build_vector_store(settings: Settings, embeddings: EmbeddingProvider) -> VectorStore:
    provider = settings.vector_store
    if provider == "local":
        return LocalVectorStore(
            settings.data_dir,
            embedding_provider=embeddings.name,
            dimension=embeddings.dimension,
        )
    raise ProviderConfigError(f"Unsupported vector store: {provider}")


def build_knowledge_base(settings: Settings) -> KnowledgeBase:
    """Assemble embeddings, the vector store, and the answerer."""
    embeddings = build_embedding_provider(settings)
    return KnowledgeBase(
        embeddings=embeddings,
        store=build_vector_store(settings, embeddings),
        answerer=build_answer_provider(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )
