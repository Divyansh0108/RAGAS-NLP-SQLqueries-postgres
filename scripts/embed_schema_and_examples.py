import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from src.config import get_settings

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()

# ── ChromaDB + Embedding Function ────────────────────────────────────────────
embedding_fn = OllamaEmbeddingFunction(
    url=settings.embedding_url,
    model_name=settings.embedding_model,
)

client = chromadb.PersistentClient(path=str(settings.chroma_path))

# ── Collections ──────────────────────────────────────────────────────────────
schema_col = client.get_or_create_collection(
    name=settings.schema_collection_name,
    embedding_function=embedding_fn,
)

examples_col = client.get_or_create_collection(
    name=settings.examples_collection_name,
    embedding_function=embedding_fn,
)


# ── 1. Embed Schema ───────────────────────────────────────────────────────────
def embed_schema():
    print("📂 Loading schema...")
    with open(settings.schema_path) as f:
        schema = json.load(f)

    documents, metadatas, ids = [], [], []

    for table in schema.values():
        table_name = table["table_name"]
        columns = ", ".join(
            f"{col['column_name']} ({col['data_type']})" for col in table["columns"]
        )
        foreign_keys = "; ".join(
            f"{fk['column_name']} → {fk['foreign_table_name']}.{fk['foreign_column_name']}"
            for fk in table.get("foreign_keys", [])
        )

        doc = (
            f"Table: {table_name}\n"
            f"Columns: {columns}\n"
            f"Foreign Keys: {foreign_keys if foreign_keys else 'None'}"
        )

        documents.append(doc)
        metadatas.append({"table_name": table_name})
        ids.append(f"schema_{table_name}")

    schema_col.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Embedded {len(documents)} tables into schema_collection")


# ── 2. Embed Examples ─────────────────────────────────────────────────────────
def embed_examples():
    print("📂 Loading examples...")
    documents, metadatas, ids = [], [], []

    with open(settings.examples_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            doc = f"Question: {ex['question']}\nSQL: {ex['sql']}"
            documents.append(doc)
            metadatas.append(
                {
                    "id": ex["id"],
                    "difficulty": ex["difficulty"],
                    "tables": ", ".join(ex.get("tables", [])),
                    "sql_patterns": ", ".join(ex.get("sql_patterns", [])),
                }
            )
            ids.append(f"example_{ex['id']}")

    examples_col.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Embedded {len(documents)} examples into examples_collection")


# ── 3. Smoke Test ─────────────────────────────────────────────────────────────
def smoke_test():
    print("\n🔍 Smoke test — querying: 'top rented movies'")

    schema_results = schema_col.query(
        query_texts=["top rented movies"],
        n_results=3,
    )
    print("\n📌 Relevant tables:")
    for doc in schema_results["documents"][0]:
        print(f"  - {doc.split(chr(10))[0]}")

    example_results = examples_col.query(
        query_texts=["top rented movies"],
        n_results=3,
    )
    print("\n📌 Relevant examples:")
    for doc in example_results["documents"][0]:
        print(f"  - {doc.split(chr(10))[0]}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"💾 ChromaDB will persist to: {settings.chroma_path}\n")
    embed_schema()
    embed_examples()
    smoke_test()
    print("\n✅ Done. ChromaDB is ready.")
