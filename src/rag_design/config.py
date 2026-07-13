from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderProvider(str, Enum):
    voyage = "voyage"
    local = "local"


class ChunkStrategy(str, Enum):
    semantic = "semantic"
    fixed = "fixed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None

    # Embedder
    embedder: EmbedderProvider = EmbedderProvider.voyage
    embedder_fallback: bool = True

    # Chunking
    chunk_strategy: ChunkStrategy = ChunkStrategy.semantic
    min_tokens: int = 256
    max_tokens: int = 512
    breakpoint_percentile: int = 95
    chunk_embed_model: str = "all-MiniLM-L6-v2"

    # Embedding models
    st_model: str = "all-mpnet-base-v2"
    voyage_model: str = "voyage-3-lite"

    # Generation
    generation_model: str = "claude-sonnet-4-5"
    judge_model: str = "claude-haiku-3-5"

    # Retrieval
    top_k: int = 5
    retrieval_threshold: float = 0.0

    # Storage
    chroma_path: Path = Path("data/processed/chroma")

    @model_validator(mode="after")
    def validate_voyage_key(self) -> "Settings":
        if self.embedder == EmbedderProvider.voyage and not self.embedder_fallback:
            if not self.voyage_api_key:
                raise ValueError(
                    "VOYAGE_API_KEY must be set when EMBEDDER=voyage and EMBEDDER_FALLBACK=false"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
