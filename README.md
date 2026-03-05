<div align="center">

# 🤖 AI-Powered Text-to-SQL RAG Chatbot

### *Natural Language → SQL Queries → Database Insights*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18.3-336791.svg)](https://www.postgresql.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000.svg)](https://ollama.ai/)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-FF6B6B.svg)](https://chainlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF4438.svg)](https://www.trychroma.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-00C853.svg)](https://github.com/explodinggradients/ragas)
[![Tests](https://img.shields.io/badge/Tests-64%20Passed-success.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A production-ready RAG system that democratizes database access through natural language, built entirely with open-source technologies and designed to run locally.*

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Evaluation](#-evaluation) • [Design Journey](#-design-journey-original-plan-vs-implementation)

---

</div>

## 📖 About the Author

**👨‍💻 Connect With Me**

For in-depth technical blogs, tutorials, and detailed explanations of this project:

- 📝 **Blog/Medium**: [Read my technical articles](https://medium.com/@divyanshpandey0108)
- 💼 **LinkedIn**: [Connect with me](https://www.linkedin.com/in/divyansh-pandey-ds/)
- 🐦 **Twitter/X**: [Follow for updates](https://x.com/metadatahere)

> 📚 **Project Deep Dives**: Check out my Medium blog for detailed technical breakdowns on RAG architecture, LLM integration, and production-grade text-to-SQL systems!

---

## 🎯 Executive Summary

This project implements an **end-to-end Text-to-SQL chatbot** using Retrieval-Augmented Generation (RAG) that enables non-technical users to query PostgreSQL databases using natural language. The system runs **entirely locally** on consumer hardware (tested on M-series Mac with 24GB RAM), requiring no cloud APIs or external dependencies.

**Key Achievements:**
- 🎯 **84% SQL accuracy** (80% end-to-end) with ~2.5s latency
- 🔒 **Production-ready**: Security hardening, 64 tests, RAGAS evaluation
- 🚀 **Developer-friendly**: Modular architecture, full documentation

## 🌟 Features

- **🗣️ Natural Language to SQL**: English → PostgreSQL queries using specialized code LLMs
- **🧠 RAG-Enhanced**: ChromaDB retrieval of schema docs and query examples
- **🔄 Multi-Model**: Switch between Qwen2.5-Coder (84%) and CodeLlama (78%)
- **💬 Interactive UI**: Chainlit interface with examples, help commands, schema browser
- **🛡️ Security**: SQL injection prevention, rate limiting, query complexity checks
- **📈 Evaluation**: RAGAS metrics, 64 unit tests, 97% coverage
- **⚙️ Production-Ready**: Structured logging, Pydantic settings, error recovery

---

## 🏗️ Architecture

### System Overview

<div align="center">

![Architecture Diagram](assets/diagram-export-2-25-2026-2_01_15-PM.png)

*Complete system architecture showing data flow from user input to database results*

</div>

### Component Interaction

<div align="center">

![Component Diagram](assets/image.png)

*Detailed component interaction and RAG pipeline visualization*

</div>

### Pipeline Flow

**Key Steps:**

1. **Input Processing** → Validate, sanitize (remove null bytes, check length), check rate limits
2. **Context Retrieval** → Query ChromaDB for top-3 relevant schema docs + top-2 example queries
3. **SQL Generation** → Assemble prompt with context, call Ollama LLM (Qwen2.5-Coder/CodeLlama)
4. **SQL Validation** → Block dangerous keywords (DROP, DELETE, etc.), check query complexity
5. **Execution** → Run against PostgreSQL with timeout protection (30s default)
6. **Results** → Format and display with user-friendly error handling

---

## 📦 Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **PostgreSQL 18+** ([Download](https://www.postgresql.org/download/))
- **Ollama** ([Download](https://ollama.ai/download))
- **uv** (Python package manager) ([Install](https://github.com/astral-sh/uv))

### Quick Start (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/Divyansh0108/RAGAS-NLP-SQLqueries-postgres.git
cd RAGAS-NLP-SQLqueries-postgres

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 4. Setup database
createdb dvdrental
psql -U your_user -d dvdrental -f dvdrental/restore.sql

# 5. Install Ollama models
ollama pull qwen2.5-coder
ollama pull codellama
ollama pull nomic-embed-text

# 6. Generate embeddings
uv run python scripts/embed_schema_and_examples.py

# 7. Launch the chatbot 🚀
uv run chainlit run src/ui/app.py
```

Open `http://localhost:8000` in your browser!

### Detailed Installation

<details>
<summary><b>📦 Step 1: Install Python 3.11+</b></summary>

**macOS (Homebrew):**
```bash
brew install python@3.11
python3.11 --version  # Verify
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and install (check "Add to PATH")

</details>

<details>
<summary><b>🐘 Step 2: Install & Setup PostgreSQL 18+</b></summary>

**macOS (Homebrew):**
```bash
brew install postgresql@18
brew services start postgresql@18

# Create user (if needed)
createuser -s your_username
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-18 postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create user
sudo -u postgres createuser --interactive
```

**Windows:**
Download installer from [postgresql.org](https://www.postgresql.org/download/windows/) and follow wizard.

**Load DVD Rental Database:**
```bash
# Download DVD Rental database (if not included)
# wget https://www.postgresqltutorial.com/wp-content/uploads/2019/05/dvdrental.zip
# unzip dvdrental.zip

# Create database and restore
createdb dvdrental
psql -U your_username -d dvdrental -f dvdrental/restore.sql

# Verify tables loaded
psql -U your_username -d dvdrental -c "\dt"
# Should show: actor, address, category, city, country, customer, film, ...
```

</details>

<details>
<summary><b>🦙 Step 3: Install Ollama & Models</b></summary>

**Download Ollama:**
- **macOS/Linux**: Visit [ollama.ai/download](https://ollama.ai/download) and install
- **Verify**: `ollama --version`

**Pull Required Models:**
```bash
ollama pull qwen2.5-coder:7b  # Primary model (~4.7GB)
ollama pull codellama:7b       # Alternative (~3.8GB)
ollama pull nomic-embed-text   # Embeddings (~274MB)
ollama list                     # Verify installation
```

</details>

<details>
<summary><b>📦 Step 4: Install uv Package Manager</b></summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify:**
```bash
uv --version  # Should show version 0.1.x or higher
```

</details>

<details>
<summary><b>⚙️ Step 5: Clone Repository & Install Dependencies</b></summary>

```bash
# Clone repository
git clone https://github.com/Divyansh0108/RAGAS-NLP-SQLqueries-postgres.git
cd RAGAS-NLP-SQLqueries-postgres

# Install dependencies
uv sync  # Creates venv and installs packages
```

</details>

<details>
<summary><b>🔐 Step 6: Environment Configuration</b></summary>

**Create .env file:**
```bash
cp .env.example .env
```

**Edit .env with your settings:**
```bash
# Database Connection
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/dvdrental

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=qwen2.5-coder:7b
EMBEDDING_MODEL=nomic-embed-text

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=dvdrental_collection

# Security Settings
RATE_LIMIT_MAX_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60
MAX_QUERY_COMPLEXITY=10  # Max JOIN operations
MAX_NESTED_SUBQUERIES=3

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log
LOG_ROTATION=10 MB
LOG_RETENTION=7 days

# UI Configuration
CHAINLIT_PORT=8000
CHAINLIT_HOST=localhost
```

**Key Variables:** `DATABASE_URL` (PostgreSQL connection), `OLLAMA_BASE_URL`, `DEFAULT_LLM_MODEL`, `RATE_LIMIT_MAX_REQUESTS` (10 req/min), `MAX_QUERY_COMPLEXITY` (max 10 JOINs)

</details>

<details>
<summary><b>🧠 Step 7: Generate Embeddings</b></summary>

```bash
# Generate embeddings for schema + examples
uv run python scripts/embed_schema_and_examples.py
# Creates ChromaDB collection with 15 schema docs + 50 examples
```

</details>

<details>
<summary><b>✅ Step 8: Run Tests (Optional but Recommended)</b></summary>

Verify everything works before first run:

```bash
# Run all tests
uv run pytest tests/ -v  # Should pass all 64 tests
```

</details>

<details>
<summary><b>🚀 Step 9: Launch the Chatbot</b></summary>

```bash
uv run chainlit run src/ui/app.py
# Opens browser at http://localhost:8000
```

</details>

---



## 💬 Usage

### Starting the Chatbot

```bash
# Activate environment and run
uv run chainlit run src/ui/app.py

# Custom host/port
uv run chainlit run src/ui/app.py --host 0.0.0.0 --port 8080
```

The chatbot web interface will open automatically at `http://localhost:8000`.

### Example Queries

**Simple:** "Show me all customer names" • "How many films are in the database?"

**Complex:** "Top 10 customers by spending" • "Films never rented" • "Average rental duration by category"

**Advanced:** "Actor pairs appearing together most" • "Revenue by store per month" • "Rental patterns by day of week"

### Special Commands

`/help` - Show commands • `/schema` - Display DB schema • `/examples` - Sample queries • `/model <name>` - Switch LLM • `/reset` - Clear history • `/stats` - Session statistics

### UI Features

- **Message history** with timestamps
- **Collapsible steps** showing retrieval → generation → execution
- **Syntax-highlighted SQL** with result tables
- **Friendly error messages** with actionable solutions
- **Action buttons** for quick access to examples and schema

### Model Selection

- **qwen2.5-coder:7b** (Default): 84% accuracy, ~2.5s latency - best balance
- **codellama:7b**: 78% accuracy, ~2.1s latency - faster for simple queries
- **qwen2.5-coder:14b**: 87% accuracy, ~6.8s latency - complex analytical queries

Switch with `/model <name>` command

---

### Performance Metrics

| Metric | Value |
|--------|-------|
| **SQL Accuracy** | 84% (Qwen2.5-Coder) / 78% (CodeLlama) |
| **End-to-End Accuracy** | 80% |
| **Average Latency** | ~2.5s |
| **Retrieval Precision** | 0.91 |
| **Test Coverage** | 97% (64 tests) |

## 🎨 Design Journey: Original Plan vs. Implementation

<div align="center">

### **Evolution of the Project**

</div>

When this project began, the original plan outlined a comprehensive text-to-SQL system. However, through iterative development and real-world testing, several key decisions diverged from the initial blueprint. Here's why:

### 📋 Original Plan

**From:** `Plan/Chatbot_Plan.md` (Initial Design Document)

| Component | Original Plan | Rationale |
|-----------|---------------|-----------|
| **Database** | MySQL with `classicmodels` schema | Industry-standard e-commerce dataset |
| **UI Framework** | Streamlit | Rapid prototyping, Python-native |
| **Backend** | FastAPI | RESTful API, async support |
| **LLM Models** | Generic LLaMA variants | Open-source, local inference |
| **Vector Store** | Qdrant or ChromaDB | High-performance vector search |
| **Evaluation** | RAGAS with custom metrics | SQL semantic equivalence |

### 🎯 Final Implementation

**What We Built:** Production-ready system with strategic improvements

| Component | Implementation | **Why We Changed** |
|-----------|----------------|-------------------|
| **Database** | **PostgreSQL 18.3** with `dvdrental` schema | ✅ **Superior features**: Better JSON support, advanced indexing (BRIN, GiST), window functions<br>✅ **Ecosystem**: Richer sample databases (dvdrental is film-rental vs retail)<br>✅ **Performance**: MVCC for better concurrency, materialized views<br>✅ **Community**: More extensive documentation and tutorials |
| **UI Framework** | **Chain lit** (instead of Streamlit) | ✅ **Async-first**: Native async support for real-time streaming<br>✅ **Chat-optimized**: Built specifically for conversational AI (message history, typing indicators)<br>✅ **UX**: Collapsible sections, execution steps, better error displays<br>✅ **Developer Experience**: Simpler decorator-based routing vs Streamlit's reruns |
| **Backend** | **Integrated Chainlit** (no separate FastAPI) | ✅ **Simplicity**: Eliminated unnecessary layer (Chainlit has built-in server)<br>✅ **Maintenance**: Fewer moving parts, easier deployment<br>✅ **Performance**: Direct function calls vs HTTP overhead<br>✅ **Development Speed**: Faster iteration without API versioning concerns |
| **LLM Models** | **Qwen2.5-Coder + CodeLlama** (specific code models) | ✅ **Specialization**: Purpose-built for code generation (SQL is code)<br>✅ **Benchmark Performance**: Qwen2.5-Coder: 84% accuracy vs ~70% with generic models<br>✅ **Context Understanding**: Better schema comprehension, foreign key relationships<br>✅ **Prompt Efficiency**: Requires fewer examples due to SQL-specific pre-training |
| **Vector Store** | **ChromaDB** (definitive choice) | ✅ **File-based Persistence**: SQLite backend, no server management<br>✅ **Simplicity**: Single Python library, zero configuration<br>✅ **Embeddings**: Built-in support for multiple embedding providers<br>✅ **Performance**: HNSW algorithm, <0.5s retrieval latency<br>✅ **Developer-Friendly**: Great docs, active community |
| **Evaluation** | **RAGAS + 64 unit tests + integration tests** | ✅ **Production Focus**: Added comprehensive testing (unit + integration)<br>✅ **Security**: Validation tests for SQL injection, rate limiting<br>✅ **Reliability**: Error handling tests, timeout scenarios<br>✅ **Metrics**: Added retrieval precision (0.91), execution success (0.80) |

### 🔑 Key Design Decisions & Rationale

<details>
<summary><b>1. PostgreSQL over MySQL</b></summary>

**Decision**: Switch from MySQL (classicmodels) to PostgreSQL (dvdrental)

**Reasoning**:
- **Feature Richness**: PostgreSQL's JSON/JSONB types, array operations, and CTE support enable more complex queries
- **Sample Database Quality**: `dvdrental` has 15 tables with realistic relationships (customers, rentals, films, actors) vs `classicmodels` (8 tables, simpler schema)
- **Evaluation Diversity**: Film rental domain offers richer query patterns (time-based rentals, many-to-many actor-film relationships)
- **Performance**: MVCC eliminates read-write blocking, crucial for concurrent chatbot users

**Impact**: Enabled testing of more diverse SQL patterns (temporal queries, many-to-many joins, subqueries)

</details>

<details>
<summary><b>2. Chainlit over Streamlit + FastAPI</b></summary>

**Decision**: Use Chainlit as unified UI + backend instead of separate Streamlit + FastAPI

**Reasoning**:
- **Async Architecture**: Chainlit's event-driven model (`@cl.on_message`) naturally handles streaming LLM responses
- **Chat-Native UX**: Built-in message history, typing indicators, collapsible steps (retrieval → generation → execution)
- **Development Velocity**: Eliminated API contract design, reduced codebase by ~30%
- **User Experience**: Action buttons for examples, `/help` and `/schema` commands, model switching

**Trade-off**: Less flexible for non-chat interfaces, but our use case is purely conversational

**Impact**: Faster iteration, cleaner codebase, better UX (transparent pipeline stages)

</details>

<details>
<summary><b>3. Specialized Code LLMs (Qwen2.5-Coder, CodeLlama)</b></summary>

**Decision**: Use code-specialized models instead of general-purpose LLaMA variants

**Reasoning**:
- **Performance Gap**: Qwen2.5-Coder achieved **84% SQL accuracy** vs **~70%** with base LLaMA-7B
- **Context Efficiency**: Code models require **fewer examples** (2-3 vs 5-7) to generalize
- **Schema Understanding**: Better handling of foreign keys, data types, constraints
- **Error Recovery**: More accurate self-correction when given syntax errors

**Benchmark Example**:
```sql
# Query: "Show top 5 customers by total spending"

# Generic LLaMA (Incorrect):
SELECT customer_id, SUM(amount) FROM payment GROUP BY customer_id LIMIT 5;
# ❌ Missing customer names, no ORDER BY

# Qwen2.5-Coder (Correct):
SELECT c.customer_id, c.first_name, c.last_name, SUM(p.amount) AS total
FROM customer c JOIN payment p ON c.customer_id = p.customer_id
GROUP BY c.customer_id ORDER BY total DESC LIMIT 5;
# ✅ Proper join, sorting, aliasing
```

**Impact**: Reduced prompt engineering effort, improved accuracy by 14%

</details>

<details>
<summary><b>4. Production-Grade Security & Testing</b></summary>

**Decision**: Add comprehensive security hardening and 64 unit tests (not in original plan)

**Additions**:
1. **Security Features**:
   - SQL injection prevention (blocks DROP, DELETE, UPDATE, INSERT)
   - Input sanitization (null bytes, length limits)
   - Rate limiting (10 req/min per session)
   - Query complexity checks (max 10 joins, max 3 nested subqueries)

2. **Testing Infrastructure**:
   - 64 unit tests (configuration, validation, rate limiting, retrieval, LLM, execution)
   - Integration tests (end-to-end pipeline verification)
   - Mocked dependencies (ChromaDB, Ollama, PostgreSQL)

**Reasoning**: Original plan was research-focused; production deployment demanded robustness

**Impact**: System can handle malicious queries, high load, and edge cases gracefully

</details>

<details>
<summary><b>5. Configuration Management & Observability</b></summary>

**Additions** (not in original plan):
- **Pydantic Settings**: Type-safe configuration with validation
- **Structured Logging**: Loguru with rotation (10MB), retention (1 week)
- **Error Categorization**: Connection, syntax, timeout, validation errors
- **User-Friendly Messages**: Actionable solutions for every error type

**Example**:
```python
# Before (original plan):
print(f"Error: {str(e)}")

# After (implementation):
logger.error(f"Database connection failed: {e}", exc_info=True)
await cl.Message(content=\
    "❌ **Cannot Connect to Database**\n\n"
    f"{str(e)}\n\n"
    "**Solution**: Check if PostgreSQL is running:\n"
    "```bash\npg_ctl status\n```"
).send()
```

**Impact**: Easier debugging, better user experience, production monitoring

</details>

### 📊 Design Evolution Summary

| Aspect | Original Plan | Implementation | **Gain** |
|--------|---------------|----------------|----------|
| **Accuracy** | 70-75% (estimated) | 84% (Qwen2.5-Coder) | +14% improvement |
| **Latency** | 3-5s (estimated) | ~2.5s | 20-50% faster |
| **Codebase** | FastAPI + Streamlit (split) | Chainlit (unified) | 30% fewer lines |
| **Testing** | Basic evaluation | 64 unit + integration tests | Production-grade |
| **Security** | Not specified | Full hardening | Enterprise-ready |
| **UX** | Basic chat | Interactive (examples, help, steps) | Delightful |

---

## 🧠 Technical Deep Dive

### RAG Pipeline

1. **Schema Extraction** → Table metadata, columns, foreign keys, descriptions
2. **Embedding Generation** → 768-dim vectors using nomic-embed-text
3. **Retrieval** → Top-3 schema docs + top-2 examples via ChromaDB similarity search
4. **Prompt Construction** → Assemble context: schema + examples + user question
5. **LLM Generation** → Ollama (qwen2.5-coder, temperature=0.1 for deterministic SQL)
6. **Validation & Execution** → Block unsafe keywords, check complexity, execute on PostgreSQL

### Security Features

| Feature | Implementation | Purpose |
|---------|----------------|---------|
| **SQL Injection Prevention** | Block DROP, DELETE, UPDATE, INSERT, ALTER | Read-only access |
| **Input Sanitization** | Remove null bytes, 500 char limit | Prevent malformed input |
| **Rate Limiting** | 10 requests/min per session | DoS prevention |
| **Query Complexity** | Max 10 JOINs, max 3 nested subqueries | Resource protection |

### Error Handling

Graceful degradation with retry logic (exponential backoff), specific error messages with solutions, and full error logging for debugging

---

## 🔬 Key Findings & Results

**RAG Dramatically Improves Accuracy:** Zero-shot (42%) → Schema in prompt (68%) → RAG with examples (84%) - retrieving 2-3 similar examples boosted accuracy by 16%

**Query Complexity Impact:** Single table (94%) → Single JOIN (87%) → Multi-JOIN (76%) → Subqueries (73%). Adding FK descriptions improved JOIN accuracy by 9%

**Model Sweet Spot:** Qwen2.5-Coder 7B (84%, 2.5s) best for production. 14B too slow (6.8s), CodeLlama faster but less accurate (78%)

**Temperature Matters:** 0.1 temperature optimal for SQL (84%). Higher temps introduce randomness and errors

**Retrieval Quality = Higher ROI:** Better retrieval (precision 0.91) more impactful than larger models. Improving embeddings/filtering beats scaling

**Surprising Discoveries:** Natural language schema descriptions > column names only • Negative examples reduced errors by 12% • Execution time display increased user trust

---

## 🚀 Future Work & Extensibility

### Planned Enhancements

**Intelligence:** Query result summarization, multi-turn conversations, SQL explanation mode

**Visualization:** Auto-generate charts (Plotly), CSV/Excel export

**Security:** OAuth2 authentication, query audit logs, dynamic rate limiting

**Performance:** Result caching (Redis), async DB connections with pooling

**Integrations:** Multi-database support (MySQL, SQLite), REST API

### Research Directions

**Hybrid Retrieval** - Combine semantic + keyword search • **RLHF** - Fine-tune on user corrections • **Agent-Based Execution** - Auto-debug failed queries

### How to Extend

**New Database:** Add schema JSON → `python src/db/schema_extractor.py --database your_db` → generate embeddings

**New LLM:** `ollama pull model_name` → update `.env` → test with `/model model_name`

**Custom Retrieval:** Edit [src/rag/retriever.py](src/rag/retriever.py) and add filtering logic

---

## 🧪 Testing

### Test Suite Overview

**Total Tests**: 64 (all passing ✅)

| Module | Tests | Coverage | Focus Areas |
|--------|-------|----------|-------------|
| **Configuration** | 8 | 100% | Settings validation, env vars |
| **Validator** | 18 | 100% | SQL injection, input sanitization, complexity checks |
| **Rate Limiter** | 9 | 100% | Request throttling, window sliding |
| **Retriever** | 6 | 95% | ChromaDB queries, embedding generation |
| **LLM** | 14 | 98% | Ollama integration, prompt formatting |
| **Executor** | 9 | 97% | PostgreSQL execution, error handling |

### Running Tests

```bash
uv run pytest tests/ -v                    # All 64 tests
uv run pytest tests/test_validator.py -v   # Specific module
uv run pytest tests/ --cov=src --cov-report=html  # With coverage
```

---

## 🛠️ Technologies

**Core Stack:** Python 3.11+ • PostgreSQL 18.3 • Ollama (Qwen2.5-Coder) • ChromaDB • Chainlit

**Why These Choices:** Code-specialized LLMs (84% vs 70% generic) • PostgreSQL advanced features (JSON, CTEs) • ChromaDB file-based simplicity • Chainlit async-first chat UI • Zero external APIs (privacy-first)

**Testing & Config:** pytest (64 tests, 97% coverage) • Pydantic (type-safe settings) • Loguru (structured logging) • RAGAS (LLM evaluation) • uv (10-100x faster than pip)

---

## 📚 Documentation

**Project Docs:** [Project Report](docs/project_report.md) • [Original Plan](Plan/Chatbot_Plan.md) • [Schema JSON](data/schema_docs/dvdrental_schema.json)

**Author's Blog:** [Medium](https://medium.com/@divyanshpandey0108) - RAG systems, Ollama deployment, prompt engineering, evaluation frameworks

**External:** [Ollama](https://ollama.ai/docs) • [Chainlit](https://docs.chainlit.io/) • [ChromaDB](https://docs.trychroma.com/) • [RAGAS](https://docs.ragas.io/) • [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)

---

## 🐛 Troubleshooting

**PostgreSQL connection error**: Check `pg_ctl status`, start with `brew services start postgresql@18`

**Ollama model not found**: Run `ollama pull qwen2.5-coder:7b` and verify with `ollama list`

**ChromaDB collection error**: Regenerate embeddings with `uv run python main.py`

**Slow generation (>10s)**: Switch to smaller model (`/model codellama:7b`) or check RAM usage

**Low accuracy (<70%)**: Use recommended model (`qwen2.5-coder:7b`), add more examples to `data/examples/`

**Rate limit exceeded**: Edit `.env` to increase `RATE_LIMIT_MAX_REQUESTS` (default 10/min)

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/YOUR_USERNAME/RAGAS-NLP-SQLqueries-postgres.git
cd RAGAS-NLP-SQLqueries-postgres
```

*Replace `YOUR_USERNAME` with your GitHub username after forking*

2. **Create Branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make Changes** - Follow PEP 8, add type hints, write tests, update docs

4. **Run Tests** - `uv run pytest tests/ -v` (ensure all pass)

5. **Commit and Push**
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

6. **Open Pull Request**
- Describe changes clearly
- Link related issues
- Add screenshots for UI changes

### Contribution Ideas

**Good First Issues:** Add example queries, fix typos, add tests, improve error messages

**Advanced:** Query result caching (Redis), data visualization (Plotly), multi-turn conversations, auth system

### Code Style

Use type hints, docstrings with Args/Returns/Raises sections, and structured logging. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TL;DR**: You can use, modify, and distribute this project freely, even for commercial purposes. Just keep the original copyright notice.

---

## 🙏 Acknowledgments

**Built with:** Ollama • Chainlit • Qwen2.5-Coder • ChromaDB • PostgreSQL • RAGAS • dvdrental sample DB

Inspired by the need for secure, privacy-preserving enterprise data access. Thanks to the open-source community!

---

## 📞 Contact

**Questions & Collaboration:**
- 📝 Technical: [Medium](https://medium.com/@divyanshpandey0108)
- 💼 Professional: [LinkedIn](https://www.linkedin.com/in/divyansh-pandey-ds/)
- 🐦 Updates: [Twitter/X](https://x.com/metadatahere)

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ using local AI • No API keys required • Privacy-first**

</div>

---

## 🗺️ Project Roadmap

### ✅ Phase 1: Core Functionality (Completed)
- [x] RAG pipeline (retrieval + generation)
- [x] PostgreSQL integration
- [x] Chainlit UI with chat interface
- [x] Basic error handling
- [x] SQL validation and security

### ✅ Phase 2: Production Hardening (Completed)
- [x] Comprehensive testing (64 unit tests)
- [x] Rate limiting and query complexity checks
- [x] Structured logging and monitoring
- [x] Configuration management (Pydantic)
- [x] RAGAS evaluation framework

### 🚧 Phase 3: Enhanced Intelligence (In Progress)
- [ ] Query result summarization
- [ ] Multi-turn conversations
- [ ] Query explanation mode

### 📋 Phase 4: Visualization & UX (Planned)
- [ ] Auto-generate charts from results
- [ ] Export to CSV/Excel
- [ ] Query history with search
- [ ] Saved queries/favorites

### 📋 Phase 5: Enterprise Features (Planned)
- [ ] User authentication (OAuth2)
- [ ] Query audit logs
- [ ] Multiple database support
- [ ] REST API endpoint

---

## 📈 Project Stats

<div align="center">

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~3,500 |
| **Test Coverage** | 97% |
| **Documentation** | 100% functions documented |
| **CI/CD** | GitHub Actions (planned) |
| **Performance** | <3s average response |
| **Accuracy** | 84% SQL correctness |

</div>

---

**[⬆ Back to Top](#-ai-powered-text-to-sql-rag-chatbot)**

*Made with ❤️ using local AI • No API keys required • Privacy-first*

</div>
