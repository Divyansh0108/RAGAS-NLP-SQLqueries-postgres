import re
import time
from typing import Any

from langchain_ollama import OllamaLLM
from requests.exceptions import ConnectionError, Timeout

from src.config import get_settings
from src.exceptions import (
    InvalidSQLError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    ModelNotAvailableError,
)
from src.rag.retriever import retrieve_context
from src.utils import get_logger

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()
logger = get_logger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────
QWEN = "qwen2.5-coder"
CODELLAMA = "codellama"


# ── Prompt Template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
    You are an expert PostgreSQL assistant. Your job is to write a single, valid \
    PostgreSQL query that answers the user's question.
    
    Rules:
    - Only output the SQL query. No explanation, no markdown, no comments.
    - Use only the tables and columns provided in the schema below to you.
    - Use JOINs wherever necessary based on foreign key relationships.
    - Always use table aliases for clarity and avoiding ambiguity.
    - Ensure the SQL is syntactically correct and can be executed without modification.
    - End the query with a semicolon.
    
    {context}
    
    ### Question:
    {question}
    
    ### SQL:
"""


# ── SQL Extractor ─────────────────────────────────────────────────────────────
def extract_sql(raw: str) -> str:
    """
    Strip markdown code fences and whitespace from LLM output,
    returning only the raw SQL string, ready for execution.
    """
    if not raw or not raw.strip():
        raise InvalidSQLError("LLM returned empty response")

    # Remove ```sql ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    if not cleaned:
        raise InvalidSQLError("Failed to extract SQL from LLM response")

    # Basic validation: should contain SQL keywords
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
    if not any(keyword in cleaned.upper() for keyword in sql_keywords):
        logger.warning(f"Generated text doesn't look like SQL: {cleaned[:100]}")

    return cleaned


# ── Retry Logic ───────────────────────────────────────────────────────────────
def retry_on_failure(func, max_retries: int = 3, backoff: float = 1.0):
    """
    Retry a function with exponential backoff on failure.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff: Initial backoff time in seconds

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (ConnectionError, Timeout, LLMConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        except Exception as e:
            # Don't retry on other errors
            raise


# ── Core Generation Function ──────────────────────────────────────────────────
def generate_sql(
    question: str,
    model: str | None = None,
    n_schema: int | None = None,
    n_examples: int | None = None,
    max_retries: int = 2,
) -> dict:
    """
    Given a natural language question:
    1. Retrieve relevant schema + examples from ChromaDB
    2. Build a prompt
    3. Call the specified Ollama model
    4. Extract and return the SQL

    Args:
        question: Natural language question
        model: LLM model to use (defaults to settings.default_llm_model)
        n_schema: Number of schema tables to retrieve (defaults to settings)
        n_examples: Number of examples to retrieve (defaults to settings)
        max_retries: Maximum number of retry attempts on failure

    Returns a dict with:
        - question: original question
        - model: model used
        - context: retrieved context
        - prompt: full prompt sent to LLM
        - raw_response: raw LLM output
        - sql: extracted SQL query
        - error: error message if failed (None if success)

    Raises:
        LLMError: If LLM invocation fails
        InvalidSQLError: If generated SQL is invalid
    """
    # Validate input
    if not question or not question.strip():
        raise LLMError("Question cannot be empty")

    # Use config defaults if not specified
    if model is None:
        model = settings.default_llm_model

    logger.info(f"Generating SQL for question: {question[:50]}... (model={model})")
    start_time = time.time()

    try:
        # 1. Retrieve context (with error handling in retriever)
        context = retrieve_context(question, n_schema=n_schema, n_examples=n_examples)

        # 2. Build prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # 3. Call LLM with retry logic
        def call_llm():
            try:
                llm = OllamaLLM(
                    model=model,
                    temperature=settings.llm_temperature,
                    base_url=settings.llm_base_url,
                    timeout=settings.llm_timeout,
                )
                return llm.invoke(prompt)
            except ConnectionError as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                raise LLMConnectionError(
                    f"Cannot connect to Ollama at {settings.llm_base_url}. Is Ollama running?"
                )
            except Timeout as e:
                logger.error(f"LLM request timed out: {e}")
                raise LLMTimeoutError(
                    f"LLM request timed out after {settings.llm_timeout}s"
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "model" in error_msg and "not found" in error_msg:
                    raise ModelNotAvailableError(
                        f"Model '{model}' not found. Pull it with: ollama pull {model}"
                    )
                raise LLMError(f"LLM invocation failed: {str(e)}")

        raw_response = retry_on_failure(call_llm, max_retries=max_retries)
        latency = time.time() - start_time
        logger.info(f"SQL generated in {latency:.2f}s")

        # 4. Extract and validate SQL
        sql = extract_sql(raw_response)

        return {
            "question": question,
            "model": model,
            "context": context,
            "prompt": prompt,
            "raw_response": raw_response,
            "sql": sql,
            "error": None,
        }

    except (LLMError, InvalidSQLError) as e:
        # Re-raise known errors
        logger.error(f"SQL generation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during SQL generation: {e}")
        raise LLMError(f"Failed to generate SQL: {str(e)}")


# ── Compare Both Models ───────────────────────────────────────────────────────
def compare_models(question: str) -> dict:
    """
    Run the same question through both Qwen2.5-Coder and CodeLlama
    and return both results for comparison.
    """
    print(f"\n🤖 Running: {QWEN}")
    qwen_result = generate_sql(question, model=QWEN)

    print(f"\n🤖 Running: {CODELLAMA}")
    codellama_result = generate_sql(question, model=CODELLAMA)

    return {
        "question": question,
        QWEN: qwen_result["sql"],
        CODELLAMA: codellama_result["sql"],
    }
