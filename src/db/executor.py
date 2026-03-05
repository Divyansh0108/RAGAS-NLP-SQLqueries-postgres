import traceback
import time
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError, ProgrammingError, errors

from src.config import get_settings
from src.db.config import db_config
from src.exceptions import (
    ConnectionError,
    SQLExecutionError,
    SQLInjectionError,
    SQLTimeoutError,
    ValidationError,
)
from src.utils import get_logger
from src.utils.validator import check_query_complexity, validate_sql

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()
logger = get_logger(__name__)


# ── Execute SQL ───────────────────────────────────────────────────────────────
def execute_sql(sql: str, timeout: int | None = None) -> dict[str, Any]:
    """
    Execute a SQL query against PostgreSQL and return results.

    Args:
        sql: SQL query to execute
        timeout: Query timeout in seconds (defaults to settings.db_timeout)

    Returns a dict with:
        - success: bool
        - rows: list of dicts (column_name -> value)
        - row_count: number of rows returned
        - columns: list of column names
        - error: error message if failed (None if success)
        - error_type: type of error (connection, syntax, timeout, etc.)
    """
    if timeout is None:
        timeout = settings.db_timeout

    conn = None
    logger.debug(f"Executing SQL: {sql[:100]}...")
    start_time = time.time()

    try:
        # 1. Validate SQL before execution
        try:
            is_valid, error_msg = validate_sql(sql, allow_write=False)
            if not is_valid:
                logger.error(f"SQL validation failed: {error_msg}")
                return {
                    "success": False,
                    "rows": [],
                    "row_count": 0,
                    "columns": [],
                    "error": f"SQL validation failed: {error_msg}",
                    "error_type": "validation",
                }
        except SQLInjectionError as e:
            logger.error(f"SQL injection attempt detected: {e}")
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": str(e),
                "error_type": "security",
            }
        except ValidationError as e:
            logger.error(f"SQL validation error: {e}")
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": str(e),
                "error_type": "validation",
            }

        # 2. Check query complexity
        is_acceptable, warning = check_query_complexity(sql)
        if not is_acceptable:
            logger.warning(f"Query complexity check failed: {warning}")
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": f"Query too complex: {warning}",
                "error_type": "complexity",
            }

        # 3. Connect to database
        conn = psycopg2.connect(
            db_config.connection_string,
            connect_timeout=timeout,
            options=f"-c statement_timeout={timeout * 1000}",  # Convert to ms
        )

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            latency = time.time() - start_time
            logger.info(
                f"SQL executed successfully: {len(rows)} rows in {latency:.2f}s"
            )
            return {
                "success": True,
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
                "columns": columns,
                "error": None,
                "error_type": None,
            }

    except OperationalError as e:
        latency = time.time() - start_time
        error_msg = str(e)

        # Check if it's a timeout
        if "timeout" in error_msg.lower() or "canceling statement" in error_msg.lower():
            logger.error(f"SQL timeout after {latency:.2f}s: {error_msg}")
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": f"Query timeout after {timeout}s. Try simplifying your query.",
                "error_type": "timeout",
                "traceback": traceback.format_exc(),
            }
        # Connection error
        else:
            logger.error(f"Database connection error: {error_msg}")
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": "Failed to connect to database. Please check if the database is running.",
                "error_type": "connection",
                "traceback": traceback.format_exc(),
            }

    except ProgrammingError as e:
        latency = time.time() - start_time
        error_msg = str(e)
        logger.error(f"SQL syntax error after {latency:.2f}s: {error_msg}")
        return {
            "success": False,
            "rows": [],
            "row_count": 0,
            "columns": [],
            "error": f"SQL syntax error: {error_msg}",
            "error_type": "syntax",
            "traceback": traceback.format_exc(),
        }

    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        logger.error(f"SQL execution failed after {latency:.2f}s: {error_msg}")
        return {
            "success": False,
            "rows": [],
            "row_count": 0,
            "columns": [],
            "error": f"Unexpected error: {error_msg}",
            "error_type": "unknown",
            "traceback": traceback.format_exc(),
        }

    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")


# ── Format Results ────────────────────────────────────────────────────────────
def format_results(result: dict[str, Any], max_rows: int = 20) -> str:
    """
    Format execution results into a readable string for display.
    """
    if not result["success"]:
        return f"❌ SQL Error:\n{result['error']}"

    if result["row_count"] == 0:
        return "✅ Query executed successfully. No rows returned."

    rows = result["rows"][:max_rows]
    columns = result["columns"]

    # Build table header
    header = " | ".join(columns)
    separator = "-+-".join("-" * len(col) for col in columns)
    lines = [header, separator]

    # Build rows
    for row in rows:
        line = " | ".join(str(row.get(col, "")) for col in columns)
        lines.append(line)

    output = "\n".join(lines)

    if result["row_count"] > max_rows:
        output += f"\n\n... showing {max_rows} of {result['row_count']} rows"

    return f"✅ {result['row_count']} row(s) returned:\n\n{output}"
