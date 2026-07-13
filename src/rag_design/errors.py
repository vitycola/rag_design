class RagError(Exception):
    """Base exception for all rag_design errors."""


class IngestionError(RagError):
    """Raised when a document cannot be loaded or parsed."""


class EmbeddingProviderError(RagError):
    """Raised when the configured embedder fails (missing key, quota, etc.)."""


class VectorStoreError(RagError):
    """Raised when ChromaDB operations fail (corruption, lock, etc.)."""


class GenerationError(RagError):
    """Raised when the Claude generation call fails hard (non-retryable)."""
