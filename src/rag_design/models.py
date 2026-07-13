from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Raw document loaded from a source before chunking."""

    text: str
    source: str
    source_type: str  # pdf | markdown | url
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Text chunk produced by a Chunker, ready for embedding."""

    chunk_id: str
    text: str
    source: str
    page: int  # -1 sentinel when page is not applicable
    section: str
    seq: int

    @classmethod
    def make_id(cls, source: str, seq: int) -> str:
        """Stable chunk ID: sha1(source+seq)[:16]."""
        raw = f"{source}{seq}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]


@dataclass
class RetrievedChunk:
    """A Chunk returned from a VectorStore query, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A single source reference extracted from a generated answer."""

    source: str
    page: int  # -1 when not applicable
    section: str
    chunk_id: str


@dataclass
class Answer:
    """Final output of the RAG pipeline for a single query."""

    text: str
    citations: list[Citation]
    contexts: list[str]  # raw chunk texts, needed by RAGAS
