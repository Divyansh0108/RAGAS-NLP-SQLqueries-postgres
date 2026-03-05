"""
Pytest configuration and shared fixtures for tests.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import MagicMock, Mock


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.project_root = Path("/tmp/test_project")
    settings.data_dir = Path("/tmp/test_project/data")
    settings.chroma_path = Path("/tmp/test_project/data/chroma_db")
    settings.schema_path = Path(
        "/tmp/test_project/data/schema_docs/dvdrental_schema.json"
    )
    settings.examples_path = Path("/tmp/test_project/data/examples/examples.jsonl")
    settings.evaluation_dir = Path("/tmp/test_project/data/evaluation")

    # Database config
    settings.db_host = "localhost"
    settings.db_port = 5432
    settings.db_user = "testuser"
    settings.db_password = "testpass"
    settings.db_name = "testdb"
    settings.db_timeout = 10
    settings.db_connection_string = (
        "postgresql://testuser:testpass@localhost:5432/testdb"
    )

    # Embedding config
    settings.embedding_model = "nomic-embed-text"
    settings.embedding_url = "http://localhost:11434/api/embeddings"
    settings.embedding_dimension = 768

    # Retrieval config
    settings.n_schema_results = 5
    settings.n_example_results = 3
    settings.similarity_threshold = 0.7

    # LLM config
    settings.default_llm_model = "qwen2.5-coder"
    settings.llm_temperature = 0.0
    settings.llm_timeout = 60
    settings.llm_base_url = "http://localhost:11434"
    settings.llm_context_window = 4096

    # ChromaDB collections
    settings.schema_collection_name = "schema_collection"
    settings.examples_collection_name = "examples_collection"

    # Logging
    settings.log_level = "INFO"
    settings.log_file = "logs/test.log"
    settings.log_rotation = "10 MB"
    settings.log_retention = "1 week"

    return settings


@pytest.fixture
def sample_question():
    """Sample natural language question."""
    return "What are the top 5 most rented movies?"


@pytest.fixture
def sample_sql():
    """Sample SQL query."""
    return """
    SELECT f.title, COUNT(r.rental_id) AS rental_count
    FROM film f
    JOIN inventory i ON f.film_id = i.film_id
    JOIN rental r ON i.inventory_id = r.inventory_id
    GROUP BY f.title
    ORDER BY rental_count DESC
    LIMIT 5;
    """


@pytest.fixture
def sample_schema_docs():
    """Sample schema documents."""
    return [
        "Table: film\nColumns: film_id (integer), title (character varying)\nForeign Keys: None",
        "Table: rental\nColumns: rental_id (integer), inventory_id (integer)\nForeign Keys: inventory_id → inventory.inventory_id",
        "Table: inventory\nColumns: inventory_id (integer), film_id (integer)\nForeign Keys: film_id → film.film_id",
    ]


@pytest.fixture
def sample_example_docs():
    """Sample example documents."""
    return [
        "Question: Show me all films\nSQL: SELECT * FROM film;",
        "Question: Count total rentals\nSQL: SELECT COUNT(*) FROM rental;",
    ]


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection."""
    mock_collection = MagicMock()

    def mock_query(query_texts, n_results, **kwargs):
        if "schema" in str(mock_collection.name):
            docs = [
                "Table: film\nColumns: film_id, title\nForeign Keys: None",
                "Table: rental\nColumns: rental_id, inventory_id\nForeign Keys: inventory_id → inventory.inventory_id",
            ][:n_results]
        else:
            docs = [
                "Question: Show films\nSQL: SELECT * FROM film;",
                "Question: Count rentals\nSQL: SELECT COUNT(*) FROM rental;",
            ][:n_results]

        return {
            "documents": [docs],
            "metadatas": [[{"source": "test"}] * len(docs)],
            "distances": [[0.1] * len(docs)],
        }

    mock_collection.query = mock_query
    return mock_collection


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Setup cursor mock
    mock_cursor.description = [("title",), ("rental_count",)]
    mock_cursor.fetchall.return_value = [
        {"title": "Film A", "rental_count": 100},
        {"title": "Film B", "rental_count": 90},
    ]

    mock_conn.cursor.return_value.__enter__ = lambda x: mock_cursor
    mock_conn.cursor.return_value.__exit__ = lambda x, *args: None

    return mock_conn


@pytest.fixture
def mock_llm():
    """Mock LLM instance."""
    mock = MagicMock()
    mock.invoke.return_value = "SELECT * FROM film LIMIT 5;"
    return mock


@pytest.fixture
def sample_execution_result():
    """Sample SQL execution result."""
    return {
        "success": True,
        "rows": [
            {"title": "Film A", "rental_count": 100},
            {"title": "Film B", "rental_count": 90},
        ],
        "row_count": 2,
        "columns": ["title", "rental_count"],
        "error": None,
        "error_type": None,
    }


@pytest.fixture
def sample_error_result():
    """Sample SQL error result."""
    return {
        "success": False,
        "rows": [],
        "row_count": 0,
        "columns": [],
        "error": "syntax error at or near 'SELCT'",
        "error_type": "syntax",
        "traceback": "...",
    }
