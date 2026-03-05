"""
Unit tests for database executor module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import psycopg2

from src.exceptions import (
    ConnectionError as DBConnectionError,
    SQLExecutionError,
    SQLTimeoutError,
    ValidationError,
)
from src.db.executor import execute_sql


class TestExecuteSQL:
    """Tests for SQL execution functionality."""

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_success(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test successful SQL execution."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"film_id": 1, "title": "Film 1"},
            {"film_id": 2, "title": "Film 2"},
        ]
        mock_cursor.description = [("film_id",), ("title",)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = execute_sql("SELECT film_id, title FROM film LIMIT 2;")

        assert result["success"] is True
        assert result["error"] is None
        assert len(result["rows"]) == 2
        assert result["columns"] == ["film_id", "title"]
        assert result["row_count"] == 2

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    def test_execute_sql_validation_error(
        self, mock_validate, mock_complexity, mock_connect
    ):
        """Test handling of SQL validation errors."""
        mock_validate.return_value = (False, "Dangerous SQL detected: DROP TABLE")

        result = execute_sql("DROP TABLE film;")

        assert result["success"] is False
        assert "validation" in result["error_type"]
        assert "Dangerous SQL detected" in result["error"]

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_connection_error(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test handling of database connection errors."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)
        mock_connect.side_effect = psycopg2.OperationalError("Cannot connect")

        result = execute_sql("SELECT * FROM film;")

        assert result["success"] is False
        assert result["error_type"] == "connection"
        assert "database" in result["error"].lower()

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_syntax_error(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test handling of SQL syntax errors."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = psycopg2.ProgrammingError(
            "Syntax error at 'FORM'"
        )

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = execute_sql("SELECT * FORM film;")  # Typo in FROM

        assert result["success"] is False
        assert result["error_type"] == "syntax"

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_timeout(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test handling of query timeout."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)
        mock_connect.side_effect = psycopg2.OperationalError("timeout expired")

        result = execute_sql("SELECT COUNT(*) FROM large_table;")

        assert result["success"] is False
        assert result["error_type"] == "timeout"
        assert "timeout" in result["error"].lower()

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_no_results(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test successful query with no results."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("count",)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = execute_sql("SELECT * FROM film WHERE film_id = -1;")

        assert result["success"] is True
        assert result["row_count"] == 0
        assert len(result["rows"]) == 0

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_with_null_values(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test handling of NULL values in results."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"film_id": 1, "title": "Film 1", "description": None},
            {"film_id": 2, "title": None, "description": "Description 2"},
        ]
        mock_cursor.description = [("film_id",), ("title",), ("description",)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = execute_sql("SELECT film_id, title, description FROM film;")

        assert result["success"] is True
        assert result["rows"][0]["description"] is None
        assert result["rows"][1]["title"] is None

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    @patch("src.db.executor.db_config")
    @patch("src.db.executor.settings")
    def test_execute_sql_large_resultset(
        self,
        mock_settings,
        mock_db_config,
        mock_validate,
        mock_complexity,
        mock_connect,
    ):
        """Test handling of large result sets."""
        mock_settings.db_timeout = 30
        mock_db_config.connection_string = "postgresql://user:pass@localhost/db"
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (True, None)

        # Simulate 1000 rows
        mock_results = [{"film_id": i, "title": f"Film {i}"} for i in range(1000)]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_results
        mock_cursor.description = [("film_id",), ("title",)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = execute_sql("SELECT film_id, title FROM film;")

        assert result["success"] is True
        assert result["row_count"] == 1000
        assert len(result["rows"]) == 1000

    @patch("src.db.executor.psycopg2.connect")
    @patch("src.db.executor.check_query_complexity")
    @patch("src.db.executor.validate_sql")
    def test_execute_sql_complexity_check_fails(
        self, mock_validate, mock_complexity, mock_connect
    ):
        """Test handling of query complexity check failure."""
        mock_validate.return_value = (True, None)
        mock_complexity.return_value = (
            False,
            "Too many joins (15). Maximum allowed: 10",
        )

        result = execute_sql(
            "SELECT * FROM t1 JOIN t2 JOIN t3 JOIN t4 JOIN t5 JOIN t6 JOIN t7 JOIN t8 JOIN t9 JOIN t10 JOIN t11 JOIN t12 JOIN t13 JOIN t14 JOIN t15;"
        )

        assert result["success"] is False
        assert result["error_type"] == "complexity"
        assert (
            "too complex" in result["error"].lower()
            or "complexity" in result["error"].lower()
        )
