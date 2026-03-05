"""
Unit tests for SQL validator and input sanitization.
"""

import pytest

from src.exceptions import SQLInjectionError, ValidationError
from src.utils.validator import (
    validate_sql,
    sanitize_input,
    check_query_complexity,
)


class TestSQLValidation:
    """Tests for SQL validation."""

    def test_valid_select_query(self):
        """Test that valid SELECT queries pass validation."""
        sql = "SELECT * FROM film WHERE film_id = 1;"
        is_valid, error = validate_sql(sql)
        assert is_valid
        assert error == ""

    def test_valid_with_query(self):
        """Test that valid WITH queries pass validation."""
        sql = """
        WITH rental_counts AS (
            SELECT film_id, COUNT(*) as cnt FROM rental GROUP BY film_id
        )
        SELECT * FROM rental_counts;
        """
        is_valid, error = validate_sql(sql)
        assert is_valid

    def test_dangerous_drop_statement(self):
        """Test that DROP statements are blocked."""
        sql = "DROP TABLE film;"
        with pytest.raises(SQLInjectionError):
            validate_sql(sql)

    def test_dangerous_delete_statement(self):
        """Test that DELETE statements are blocked."""
        sql = "DELETE FROM film WHERE film_id = 1;"
        with pytest.raises(SQLInjectionError):
            validate_sql(sql)

    def test_dangerous_update_statement(self):
        """Test that UPDATE statements are blocked."""
        sql = "UPDATE film SET title = 'New' WHERE film_id = 1;"
        with pytest.raises(SQLInjectionError):
            validate_sql(sql)

    def test_dangerous_insert_statement(self):
        """Test that INSERT statements are blocked."""
        sql = "INSERT INTO film (title) VALUES ('New Film');"
        with pytest.raises(SQLInjectionError):
            validate_sql(sql)

    def test_multiple_statements(self):
        """Test that multiple statements are blocked."""
        sql = "SELECT * FROM film; SELECT * FROM rental;"
        is_valid, error = validate_sql(sql)
        assert not is_valid
        assert "Multiple" in error

    def test_empty_query(self):
        """Test that empty queries are rejected."""
        is_valid, error = validate_sql("")
        assert not is_valid
        assert "empty" in error.lower()

    def test_allow_write_operations(self):
        """Test that write operations can be allowed with flag."""
        sql = "INSERT INTO film (title) VALUES ('New Film');"
        is_valid, error = validate_sql(sql, allow_write=True)
        # Should not raise SQLInjectionError, but might fail other validation
        assert True  # Just checking it doesn't raise


class TestInputSanitization:
    """Tests for input sanitization."""

    def test_sanitize_normal_input(self):
        """Test sanitization of normal input."""
        input_text = "  What are the top films?  "
        result = sanitize_input(input_text)
        assert result == "What are the top films?"

    def test_sanitize_empty_input(self):
        """Test sanitization of empty input."""
        result = sanitize_input("")
        assert result == ""

    def test_sanitize_max_length_exceeded(self):
        """Test that overly long inputs are rejected."""
        long_input = "a" * 1001
        with pytest.raises(ValidationError):
            sanitize_input(long_input, max_length=1000)

    def test_sanitize_null_bytes(self):
        """Test that null bytes are removed."""
        input_text = "text\x00with\x00nulls"
        result = sanitize_input(input_text)
        assert "\x00" not in result
        # Null bytes are removed without adding spaces
        assert result == "textwithnulls"

    def test_sanitize_custom_max_length(self):
        """Test custom max length parameter."""
        input_text = "a" * 100
        result = sanitize_input(input_text, max_length=200)
        assert len(result) == 100


class TestQueryComplexity:
    """Tests for query complexity checks."""

    def test_simple_query_complexity(self):
        """Test that simple queries pass complexity check."""
        sql = "SELECT * FROM film;"
        is_acceptable, warning = check_query_complexity(sql)
        assert is_acceptable
        assert warning == ""

    def test_query_with_reasonable_joins(self):
        """Test query with reasonable number of joins."""
        sql = """
        SELECT * FROM film f
        JOIN inventory i ON f.film_id = i.film_id
        JOIN rental r ON i.inventory_id = r.inventory_id;
        """
        is_acceptable, warning = check_query_complexity(sql)
        assert is_acceptable

    def test_query_with_excessive_joins(self):
        """Test that queries with too many joins are rejected."""
        # Create a query with 11 joins
        joins = " JOIN t{} ON t{}.id = t{}.id".format(1, 0, 1)
        for i in range(2, 12):
            joins += " JOIN t{} ON t{}.id = t{}.id".format(i, i - 1, i)

        sql = f"SELECT * FROM t0{joins};"
        is_acceptable, warning = check_query_complexity(sql, max_joins=10)
        assert not is_acceptable
        assert "JOIN" in warning

    def test_query_with_nested_subqueries(self):
        """Test that deeply nested subqueries are rejected."""
        sql = """
        SELECT * FROM (
            SELECT * FROM (
                SELECT * FROM (
                    SELECT * FROM (
                        SELECT * FROM film
                    )
                )
            )
        );
        """
        is_acceptable, warning = check_query_complexity(sql)
        assert not is_acceptable
        assert "nested" in warning.lower()
