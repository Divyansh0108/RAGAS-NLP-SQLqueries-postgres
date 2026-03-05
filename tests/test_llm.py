"""
Unit tests for LLM module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

from src.exceptions import (
    InvalidSQLError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    ModelNotAvailableError,
)
from src.models.llm import extract_sql, generate_sql


class TestSQLExtraction:
    """Tests for SQL extraction from LLM output."""

    def test_extract_clean_sql(self):
        """Test extracting clean SQL without any formatting."""
        raw = "SELECT * FROM film LIMIT 5;"
        result = extract_sql(raw)
        assert result == "SELECT * FROM film LIMIT 5;"

    def test_extract_sql_with_markdown(self):
        """Test extracting SQL from markdown code blocks."""
        raw = "```sql\nSELECT * FROM film;\n```"
        result = extract_sql(raw)
        assert result == "SELECT * FROM film;"

    def test_extract_sql_with_plain_markdown(self):
        """Test extracting SQL from plain markdown blocks."""
        raw = "```\nSELECT * FROM film;\n```"
        result = extract_sql(raw)
        assert result == "SELECT * FROM film;"

    def test_extract_sql_with_whitespace(self):
        """Test extracting SQL with extra whitespace."""
        raw = "  \n  SELECT * FROM film;  \n  "
        result = extract_sql(raw)
        assert result == "SELECT * FROM film;"

    def test_extract_empty_response(self):
        """Test that empty responses raise InvalidSQLError."""
        with pytest.raises(InvalidSQLError):
            extract_sql("")

        with pytest.raises(InvalidSQLError):
            extract_sql("   ")

    def test_extract_only_markdown(self):
        """Test that markdown-only responses raise InvalidSQLError."""
        with pytest.raises(InvalidSQLError):
            extract_sql("```sql\n```")


class TestGenerateSQL:
    """Tests for SQL generation functionality."""

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_success(self, mock_settings, mock_llm_class, mock_retrieve):
        """Test successful SQL generation."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film\nExamples: SELECT * FROM film;"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "SELECT title FROM film LIMIT 5;"
        mock_llm_class.return_value = mock_llm

        result = generate_sql("What are the top 5 films?")

        assert result["question"] == "What are the top 5 films?"
        assert result["model"] == "qwen2.5-coder"
        assert "SELECT" in result["sql"]
        assert result["error"] is None

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    def test_generate_sql_empty_question(self, mock_llm_class, mock_retrieve):
        """Test that empty questions raise LLMError."""
        with pytest.raises(LLMError):
            generate_sql("")

        with pytest.raises(LLMError):
            generate_sql("   ")

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_connection_error(
        self, mock_settings, mock_llm_class, mock_retrieve
    ):
        """Test handling of Ollama connection errors."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Cannot connect")
        mock_llm_class.return_value = mock_llm

        with pytest.raises(LLMConnectionError):
            generate_sql("What are the films?")

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_timeout(self, mock_settings, mock_llm_class, mock_retrieve):
        """Test handling of LLM timeout."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Timeout("Request timed out")
        mock_llm_class.return_value = mock_llm

        with pytest.raises(LLMTimeoutError):
            generate_sql("What are the films?")

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_model_not_found(
        self, mock_settings, mock_llm_class, mock_retrieve
    ):
        """Test handling of model not found errors."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("model 'nonexistent' not found")
        mock_llm_class.return_value = mock_llm

        with pytest.raises(ModelNotAvailableError):
            generate_sql("What are the films?")

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_empty_response(
        self, mock_settings, mock_llm_class, mock_retrieve
    ):
        """Test handling of empty LLM responses."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""
        mock_llm_class.return_value = mock_llm

        with pytest.raises(InvalidSQLError):
            generate_sql("What are the films?")

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_custom_model(
        self, mock_settings, mock_llm_class, mock_retrieve
    ):
        """Test SQL generation with custom model."""
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "SELECT * FROM film;"
        mock_llm_class.return_value = mock_llm

        result = generate_sql("What are the films?", model="codellama")

        assert result["model"] == "codellama"

    @patch("src.models.llm.retrieve_context")
    @patch("src.models.llm.OllamaLLM")
    @patch("src.models.llm.settings")
    def test_generate_sql_retry_success(
        self, mock_settings, mock_llm_class, mock_retrieve
    ):
        """Test that retry logic works on transient failures."""
        mock_settings.default_llm_model = "qwen2.5-coder"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_timeout = 60

        mock_retrieve.return_value = "Schema: film"

        mock_llm = MagicMock()
        # Fail once, then succeed
        mock_llm.invoke.side_effect = [
            ConnectionError("Temporary failure"),
            "SELECT * FROM film;",
        ]
        mock_llm_class.return_value = mock_llm

        result = generate_sql("What are the films?", max_retries=2)

        assert result["sql"] == "SELECT * FROM film;"
        assert mock_llm.invoke.call_count == 2
