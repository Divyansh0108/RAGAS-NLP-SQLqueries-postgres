"""
SQL validation and security checks.
Prevents SQL injection and dangerous operations.
"""

import re
from typing import Tuple

import sqlparse
from sqlparse import sql
from sqlparse.tokens import Keyword, DML

from src.config import get_settings
from src.exceptions import SQLInjectionError, ValidationError
from src.utils import get_logger

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()
logger = get_logger(__name__)

# Dangerous SQL keywords that should be blocked
DANGEROUS_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "GRANT",
    "REVOKE",
    "EXECUTE",
    "CALL",
    "COPY",
}

# Allowed SQL operations (read-only)
ALLOWED_KEYWORDS = {"SELECT", "WITH"}


def validate_sql(sql_query: str, allow_write: bool = False) -> Tuple[bool, str]:
    """
    Validate SQL query for security and syntax.

    Args:
        sql_query: SQL query to validate
        allow_write: Whether to allow write operations (default: False)

    Returns:
        Tuple of (is_valid, error_message)

    Raises:
        SQLInjectionError: If potential SQL injection detected
        ValidationError: If validation fails
    """
    if not sql_query or not sql_query.strip():
        return False, "SQL query is empty"

    sql_query = sql_query.strip()

    try:
        # 1. Parse SQL
        parsed = sqlparse.parse(sql_query)

        if not parsed:
            return False, "Failed to parse SQL query"

        if len(parsed) > 1:
            return (
                False,
                "Multiple SQL statements detected. Only one statement allowed.",
            )

        statement = parsed[0]

        # 2. Check for dangerous keywords (unless write operations allowed)
        if not allow_write:
            sql_upper = sql_query.upper()
            found_dangerous = [kw for kw in DANGEROUS_KEYWORDS if kw in sql_upper]

            if found_dangerous:
                logger.warning(
                    f"Dangerous SQL keywords detected: {', '.join(found_dangerous)}"
                )
                raise SQLInjectionError(
                    f"Dangerous operations not allowed: {', '.join(found_dangerous)}"
                )

        # 3. Validate it's a SELECT or WITH statement (if read-only)
        if not allow_write:
            first_token = statement.token_first(skip_ws=True, skip_cm=True)

            if first_token:
                token_value = first_token.value.upper()
                if token_value not in ALLOWED_KEYWORDS:
                    return (
                        False,
                        f"Only SELECT and WITH queries are allowed. Found: {token_value}",
                    )
            else:
                return False, "Unable to identify query type"

        # 4. Check for suspicious patterns
        suspicious_patterns = [
            (r"--", "SQL comment detected"),
            (r"/\*.*\*/", "Block comment detected"),
            (r";\s*\w+", "Multiple statements detected"),
            (r"xp_\w+", "System stored procedure call detected"),
            (r"sp_\w+", "System stored procedure call detected"),
        ]

        for pattern, error_msg in suspicious_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                logger.warning(f"Suspicious SQL pattern: {error_msg}")
                # For comments, just log warning but don't block
                if "comment" in error_msg.lower():
                    logger.info("Allowing SQL with comments")
                else:
                    raise SQLInjectionError(error_msg)

        # 5. Validate syntax (basic check)
        try:
            # Check if statement has proper structure
            if not statement.tokens:
                return False, "Invalid SQL structure"
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"

        logger.debug(f"SQL validation passed: {sql_query[:100]}...")
        return True, ""

    except SQLInjectionError:
        # Re-raise security errors
        raise
    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        raise ValidationError(f"Failed to validate SQL: {str(e)}")


def sanitize_input(user_input: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        user_input: Raw user input
        max_length: Maximum allowed input length

    Returns:
        Sanitized input string

    Raises:
        ValidationError: If input fails validation
    """
    if not user_input:
        return ""

    # Trim whitespace
    sanitized = user_input.strip()

    # Check length
    if len(sanitized) > max_length:
        raise ValidationError(
            f"Input too long. Maximum {max_length} characters allowed."
        )

    # Remove null bytes
    sanitized = sanitized.replace("\x00", "")

    # Check for excessive special characters (potential obfuscation)
    special_char_count = len(
        [c for c in sanitized if not c.isalnum() and not c.isspace()]
    )
    if special_char_count > len(sanitized) * 0.3:
        logger.warning(
            f"High special character ratio in input: {special_char_count}/{len(sanitized)}"
        )

    logger.debug(f"Input sanitized: {len(sanitized)} chars")
    return sanitized


def check_query_complexity(sql_query: str, max_joins: int = 10) -> Tuple[bool, str]:
    """
    Check if SQL query complexity is within acceptable limits.

    Args:
        sql_query: SQL query to check
        max_joins: Maximum number of joins allowed

    Returns:
        Tuple of (is_acceptable, warning_message)
    """
    try:
        sql_upper = sql_query.upper()

        # Count JOINs
        join_count = sql_upper.count(" JOIN ")
        if join_count > max_joins:
            return False, f"Too many JOINs: {join_count} (max: {max_joins})"

        # Check for nested subqueries depth
        subquery_depth = sql_upper.count("SELECT") - 1
        if subquery_depth > 3:
            return False, f"Too many nested subqueries: {subquery_depth} (max: 3)"

        return True, ""

    except Exception as e:
        logger.warning(f"Error checking query complexity: {e}")
        return True, ""  # Allow on error
