"""FastAPI application factory."""

from fastapi import FastAPI

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.config import Settings, get_settings
from app.services.factory import build_knowledge_base


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API and the knowledge base selected by `settings`."""
    resolved = settings or get_settings()
    app = FastAPI(
        title="RAG Knowledge Base",
        version=__version__,
        summary="Index documents and answer questions with a citation on every claim.",
        description=(
            "Upload PDF, Markdown, text, or DOCX files. The service chunks them, embeds them, "
            "and answers questions from those chunks. Every claim in an answer cites the source "
            "file and the chunk or page it came from. The default embedding and answer providers "
            "run offline. OpenAI-compatible endpoints are selected with environment variables."
        ),
        contact={"name": "Abdulrasheed Oseni", "url": "https://github.com/maikama8"},
        license_info={"name": "MIT"},
    )
    app.state.settings = resolved
    app.state.knowledge_base = build_knowledge_base(resolved)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
