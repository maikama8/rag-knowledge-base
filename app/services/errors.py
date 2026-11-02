"""Errors raised while indexing documents or answering a question."""


class ProviderConfigError(Exception):
    """The selected provider is missing a dependency or credential."""


class ProviderError(Exception):
    """The selected provider failed while handling a request."""


class UnsupportedDocumentError(Exception):
    """The upload is not a PDF, Markdown, text, or DOCX file."""


class InvalidDocumentError(Exception):
    """The file could not be read."""


class EmptyDocumentError(Exception):
    """The file has no extractable text."""


class VectorStoreError(Exception):
    """The local index is missing, incompatible, or failed to save."""


class UncitedClaimError(ProviderError):
    """An answer claim was produced without a source citation."""
