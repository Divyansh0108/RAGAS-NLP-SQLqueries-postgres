import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from src.config import get_settings
from src.exceptions import ChromaDBError, EmptyRetrievalError, RetrievalError
from src.utils import get_logger

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()
logger = get_logger(__name__)

# ── Embedding Function ────────────────────────────────────────────────────────
embedding_fn = OllamaEmbeddingFunction(
    url=settings.embedding_url,
    model_name=settings.embedding_model,
)

# ── ChromaDB Client ───────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(settings.chroma_path))

schema_col = client.get_or_create_collection(
    name=settings.schema_collection_name,
    embedding_function=embedding_fn,
)

examples_col = client.get_or_create_collection(
    name=settings.examples_collection_name,
    embedding_function=embedding_fn,
)


# ── Retriever ─────────────────────────────────────────────────────────────────
def retrieve_context(
    question: str, n_schema: int | None = None, n_examples: int | None = None
) -> str:
    """
    Given a natural language question, retrieve relevant schema tables
    and NL-SQL examples from ChromaDB and return a combined context string.

    Args:
        question: Natural language question
        n_schema: Number of schema tables to retrieve (defaults to settings)
        n_examples: Number of examples to retrieve (defaults to settings)

    Returns:
        Combined context string with schema and examples

    Raises:
        RetrievalError: If retrieval fails
        EmptyRetrievalError: If no relevant context is found
    """
    # Validate input
    if not question or not question.strip():
        logger.warning("Empty question provided to retriever")
        raise RetrievalError("Question cannot be empty")

    # Use config defaults if not specified
    if n_schema is None:
        n_schema = settings.n_schema_results
    if n_examples is None:
        n_examples = settings.n_example_results

    logger.debug(f"Retrieving context for question: {question[:50]}...")

    try:
        # 1. Retrieve relevant tables
        try:
            schema_results = schema_col.query(
                query_texts=[question],
                n_results=n_schema,
            )
            schema_docs = schema_results["documents"][0]
            logger.debug(f"Retrieved {len(schema_docs)} schema tables")
        except Exception as e:
            logger.error(f"Schema retrieval failed: {e}")
            raise ChromaDBError(f"Failed to retrieve schema: {str(e)}")

        # 2. Retrieve relevant examples
        try:
            example_results = examples_col.query(
                query_texts=[question],
                n_results=n_examples,
            )
            example_docs = example_results["documents"][0]
            logger.debug(f"Retrieved {len(example_docs)} examples")
        except Exception as e:
            logger.error(f"Example retrieval failed: {e}")
            raise ChromaDBError(f"Failed to retrieve examples: {str(e)}")

        # 3. Validate we have some context
        if not schema_docs and not example_docs:
            logger.warning("No context retrieved for question")
            raise EmptyRetrievalError(
                "No relevant schema or examples found. The database may be empty."
            )

        # 4. Build context string
        context = ""

        if schema_docs:
            context += "### Relevant Tables:\n"
            for doc in schema_docs:
                context += f"{doc}\n\n"

        if example_docs:
            context += "### Relevant Examples:\n"
            for doc in example_docs:
                context += f"{doc}\n\n"

        logger.info(
            f"Context built: {len(schema_docs)} tables, {len(example_docs)} examples"
        )
        return context.strip()

    except (ChromaDBError, EmptyRetrievalError):
        # Re-raise known errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error during retrieval: {e}")
        raise RetrievalError(f"Retrieval failed: {str(e)}")
