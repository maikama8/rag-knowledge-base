"""Request dependencies."""

from fastapi import Request

from app.config import Settings
from app.services.pipeline import KnowledgeBase


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_knowledge_base(request: Request) -> KnowledgeBase:
    return request.app.state.knowledge_base
