"""
Custom exception classes for the Text-to-SQL application.
Provides specific error types for better error handling and debugging.
"""


class TextToSQLError(Exception):
    """Base exception for all Text-to-SQL errors."""

    pass


# ── Database Errors ───────────────────────────────────────────────────────────
class DatabaseError(TextToSQLError):
    """Base exception for database-related errors."""

    pass


class ConnectionError(DatabaseError):
    """Failed to connect to the database."""

    pass


class SQLExecutionError(DatabaseError):
    """Failed to execute SQL query."""

    pass


class SQLTimeoutError(DatabaseError):
    """SQL query execution timed out."""

    pass


# ── Retrieval Errors ──────────────────────────────────────────────────────────
class RetrievalError(TextToSQLError):
    """Base exception for retrieval-related errors."""

    pass


class EmptyRetrievalError(RetrievalError):
    """No relevant context found during retrieval."""

    pass


class ChromaDBError(RetrievalError):
    """ChromaDB operation failed."""

    pass


# ── LLM Errors ────────────────────────────────────────────────────────────────
class LLMError(TextToSQLError):
    """Base exception for LLM-related errors."""

    pass


class ModelNotAvailableError(LLMError):
    """Requested LLM model is not available."""

    pass


class InvalidSQLError(LLMError):
    """Generated SQL is invalid or malformed."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    pass


class LLMConnectionError(LLMError):
    """Failed to connect to LLM service (Ollama)."""

    pass


# ── Validation Errors ─────────────────────────────────────────────────────────
class ValidationError(TextToSQLError):
    """Base exception for validation errors."""

    pass


class InputValidationError(ValidationError):
    """User input failed validation."""

    pass


class ConfigurationError(ValidationError):
    """Configuration is invalid or missing."""

    pass


# ── Security Errors ───────────────────────────────────────────────────────────
class SecurityError(TextToSQLError):
    """Base exception for security-related errors."""

    pass


class SQLInjectionError(SecurityError):
    """Potential SQL injection detected."""

    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded."""

    pass
