"""
Centralized configuration management using Pydantic Settings.
Supports environment variables via .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project Paths ─────────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def schema_path(self) -> Path:
        return self.data_dir / "schema_docs" / "dvdrental_schema.json"

    @property
    def examples_path(self) -> Path:
        return self.data_dir / "examples" / "examples.jsonl"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    # ── Database Configuration ────────────────────────────────────────────────
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_user: str = Field(default="divyanshpandey", description="Database user")
    db_password: str = Field(default="", description="Database password")
    db_name: str = Field(default="dvdrental", description="Database name")
    db_timeout: int = Field(default=10, description="Query timeout in seconds")

    @property
    def db_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # ── Embedding Configuration ───────────────────────────────────────────────
    embedding_model: str = Field(
        default="nomic-embed-text", description="Embedding model name"
    )
    embedding_url: str = Field(
        default="http://localhost:11434/api/embeddings",
        description="Ollama embedding URL",
    )
    embedding_dimension: int = Field(
        default=768, description="Embedding vector dimension"
    )
    embedding_batch_size: int = Field(
        default=32, description="Batch size for embedding operations"
    )

    # ── Retrieval Configuration ───────────────────────────────────────────────
    n_schema_results: int = Field(
        default=5, description="Number of schema tables to retrieve"
    )
    n_example_results: int = Field(
        default=3, description="Number of example queries to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score for retrieval"
    )

    # ── LLM Configuration ─────────────────────────────────────────────────────
    default_llm_model: Literal["qwen2.5-coder", "codellama"] = Field(
        default="qwen2.5-coder", description="Default LLM model for SQL generation"
    )
    llm_temperature: float = Field(
        default=0.0, description="LLM temperature (0 = deterministic)"
    )
    llm_timeout: int = Field(default=60, description="LLM request timeout in seconds")
    llm_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    llm_context_window: int = Field(
        default=4096, description="Maximum context window size"
    )

    # ── ChromaDB Collections ──────────────────────────────────────────────────
    schema_collection_name: str = Field(
        default="schema_collection", description="ChromaDB schema collection name"
    )
    examples_collection_name: str = Field(
        default="examples_collection", description="ChromaDB examples collection name"
    )

    # ── UI Configuration ──────────────────────────────────────────────────────
    chainlit_port: int = Field(default=8000, description="Chainlit server port")
    chainlit_host: str = Field(default="0.0.0.0", description="Chainlit server host")

    # ── Logging Configuration ─────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: str = Field(
        default="logs/app.log", description="Log file path (relative to project root)"
    )
    log_rotation: str = Field(default="10 MB", description="Log rotation size or time")
    log_retention: str = Field(default="1 week", description="Log retention period")

    @field_validator("db_port", "chainlit_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port numbers."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate LLM temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity(cls, v: float) -> float:
        """Validate similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Similarity threshold must be between 0.0 and 1.0, got {v}"
            )
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses @lru_cache to ensure singleton pattern.
    """
    return Settings()
