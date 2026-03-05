"""
Unit tests for retriever module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.exceptions import RetrievalError, EmptyRetrievalError, ChromaDBError
from src.rag.retriever import retrieve_context


class TestRetriever:
    """Tests for context retrieval functionality."""

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    @patch("src.rag.retriever.settings")
    def test_retrieve_context_success(
        self, mock_settings, mock_examples_col, mock_schema_col
    ):
        """Test successful context retrieval."""
        mock_settings.n_schema_results = 2
        mock_settings.n_example_results = 2

        # Mock schema collection
        mock_schema_col.query.return_value = {
            "documents": [
                [
                    "Table: film\nColumns: film_id, title\nForeign Keys: None",
                    "Table: rental\nColumns: rental_id\nForeign Keys: None",
                ]
            ]
        }

        # Mock examples collection
        mock_examples_col.query.return_value = {
            "documents": [
                [
                    "Question: Show films\nSQL: SELECT * FROM film;",
                    "Question: Count rentals\nSQL: SELECT COUNT(*) FROM rental;",
                ]
            ]
        }

        result = retrieve_context("What are the top films?")

        assert "Relevant Tables:" in result
        assert "Relevant Examples:" in result
        assert "film" in result
        assert "rental" in result
        assert "SELECT" in result

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    def test_retrieve_context_empty_question(self, mock_examples_col, mock_schema_col):
        """Test that empty questions are rejected."""
        with pytest.raises(RetrievalError):
            retrieve_context("")

        with pytest.raises(RetrievalError):
            retrieve_context("   ")

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    @patch("src.rag.retriever.settings")
    def test_retrieve_context_no_results(
        self, mock_settings, mock_examples_col, mock_schema_col
    ):
        """Test handling when no results are found."""
        mock_settings.n_schema_results = 2
        mock_settings.n_example_results = 2

        # Empty results
        mock_schema_col.query.return_value = {"documents": [[]]}
        mock_examples_col.query.return_value = {"documents": [[]]}

        with pytest.raises(EmptyRetrievalError):
            retrieve_context("What are the top films?")

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    @patch("src.rag.retriever.settings")
    def test_retrieve_context_schema_error(
        self, mock_settings, mock_examples_col, mock_schema_col
    ):
        """Test handling of schema retrieval errors."""
        mock_settings.n_schema_results = 2
        mock_settings.n_example_results = 2

        # Schema query fails
        mock_schema_col.query.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(ChromaDBError):
            retrieve_context("What are the top films?")

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    @patch("src.rag.retriever.settings")
    def test_retrieve_context_examples_error(
        self, mock_settings, mock_examples_col, mock_schema_col
    ):
        """Test handling of examples retrieval errors."""
        mock_settings.n_schema_results = 2
        mock_settings.n_example_results = 2

        # Schema succeeds
        mock_schema_col.query.return_value = {
            "documents": [["Table: film\nColumns: film_id\nForeign Keys: None"]]
        }

        # Examples query fails
        mock_examples_col.query.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(ChromaDBError):
            retrieve_context("What are the top films?")

    @patch("src.rag.retriever.schema_col")
    @patch("src.rag.retriever.examples_col")
    @patch("src.rag.retriever.settings")
    def test_retrieve_context_custom_limits(
        self, mock_settings, mock_examples_col, mock_schema_col
    ):
        """Test custom retrieval limits."""
        mock_settings.n_schema_results = 5
        mock_settings.n_example_results = 3

        mock_schema_col.query.return_value = {
            "documents": [["Table: film\nColumns: film_id\nForeign Keys: None"]]
        }
        mock_examples_col.query.return_value = {
            "documents": [["Question: Show films\nSQL: SELECT * FROM film;"]]
        }

        retrieve_context("What are the top films?", n_schema=10, n_examples=5)

        # Check that custom limits were used
        mock_schema_col.query.assert_called_once()
        call_kwargs = mock_schema_col.query.call_args[1]
        assert call_kwargs["n_results"] == 10

        mock_examples_col.query.assert_called_once()
        call_kwargs = mock_examples_col.query.call_args[1]
        assert call_kwargs["n_results"] == 5
