import chainlit as cl

from src.db.executor import execute_sql, format_results
from src.exceptions import (
    ChromaDBError,
    EmptyRetrievalError,
    InvalidSQLError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    ModelNotAvailableError,
    RateLimitError,
    RetrievalError,
    ValidationError,
)
from src.models.llm import CODELLAMA, QWEN, generate_sql
from src.utils import get_logger
from src.utils.rate_limiter import rate_limiter
from src.utils.validator import sanitize_input

# ── Configuration ─────────────────────────────────────────────────────────────
logger = get_logger(__name__)
MODELS = [QWEN, CODELLAMA]


@cl.on_chat_start
async def on_chat_start():
    logger.info("New chat session started")
    # Default model
    cl.user_session.set("model", QWEN)

    # Model selector
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model",
                label="SQL Model",
                values=MODELS,
                initial_value=QWEN,
            )
        ]
    ).send()

    welcome_message = """👋 **Welcome to the Text-to-SQL Chatbot!**

Ask me any question about the **dvdrental** database in plain English, and I'll generate SQL and run it for you.

🤖 **Current model:** `{model}`

---

### 📚 Example Queries

Try asking:
- "**Show me the top 5 films by rental count**"
- "**List all customers from California**"
- "**What's the total revenue by store?**"
- "**Find actors who appeared in more than 20 films**"
- "**Show rentals from the last 7 days**"

---

### 💡 Tips

- **Be specific**: Include details like limits, filters, or sorting preferences
- **Ask about relationships**: I understand foreign keys and table relationships
- **Use natural language**: No need for SQL keywords
- **Switch models**: Use ⚙️ settings above to try different models

---

### 🏗️ Database Schema

The **dvdrental** database contains:
- **film**: Movies available for rent (title, description, rating, length, etc.)
- **actor**: Actors in the database
- **customer**: Customer information
- **rental**: Rental transactions
- **payment**: Payment records
- **store**: Store locations
- **staff**: Staff members
- **inventory**: Film inventory at each store
- **category**: Film categories

---

### ❓ Need Help?

Type **"/help"** for troubleshooting tips or **"/schema"** for detailed schema information.
"""

    # Send welcome message with suggested actions
    actions = [
        cl.Action(
            name="example_1",
            value="Show me the top 5 films by rental count",
            label="📽️ Top Films",
            description="Popular films query",
        ),
        cl.Action(
            name="example_2",
            value="List all customers from California",
            label="👥 Customers CA",
            description="Filter customers",
        ),
        cl.Action(
            name="example_3",
            value="What's the total revenue by store?",
            label="💰 Revenue",
            description="Aggregate query",
        ),
        cl.Action(
            name="help",
            value="/help",
            label="🆘 Help",
            description="Troubleshooting",
        ),
        cl.Action(
            name="schema",
            value="/schema",
            label="🗄️ Schema",
            description="Database structure",
        ),
    ]

    await cl.Message(content=welcome_message.format(model=QWEN), actions=actions).send()


@cl.on_settings_update
async def on_settings_update(settings):
    model = settings["model"]
    cl.user_session.set("model", model)
    await cl.Message(content=f"✅ Model switched to `{model}`").send()


@cl.action_callback("example_1")
async def on_example_1(action):
    """Handle example query 1."""
    await on_message(cl.Message(content=action.value))


@cl.action_callback("example_2")
async def on_example_2(action):
    """Handle example query 2."""
    await on_message(cl.Message(content=action.value))


@cl.action_callback("example_3")
async def on_example_3(action):
    """Handle example query 3."""
    await on_message(cl.Message(content=action.value))


@cl.action_callback("help")
async def on_help_action(action):
    """Handle help command."""
    await on_message(cl.Message(content=action.value))


@cl.action_callback("schema")
async def on_schema_action(action):
    """Handle schema command."""
    await on_message(cl.Message(content=action.value))


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    model = cl.user_session.get("model", QWEN)
    session_id = cl.user_session.get("id", "unknown")

    # Validate input
    if not question:
        await cl.Message(content="❌ Please provide a question.").send()
        return

    # Handle special commands
    if question.lower() == "/help":
        help_message = """### 🆘 Troubleshooting Guide

**1. Empty Results**
- The query might be too specific
- Try using broader terms
- Example: Instead of "Find John Doe", try "Find customers named John"

**2. Connection Errors**
- Check if PostgreSQL is running: `pg_ctl status`
- Verify connection settings in `.env` file
- Test connection: `psql -U [username] -d dvdrental`

**3. Model Not Available**
- Pull the model: `ollama pull qwen2.5-coder`
- List available models: `ollama list`
- Start Ollama service: `ollama serve`

**4. Timeout Errors**
- Simplify your question
- Reduce the scope (add more filters or limits)
- Try asking for aggregated data instead of raw rows

**5. Rate Limit Exceeded**
- Wait 60 seconds before making more requests
- Current limit: 10 requests per minute

**6. Invalid SQL Generated**
- Rephrase your question more clearly
- Be specific about what data you want
- Try switching to a different model in settings

---

### 📝 Best Practices

- Start with simple queries, then add complexity
- Use specific table/column names when known
- Ask for limited results (e.g., "top 10", "first 5")
- Include sorting preferences (e.g., "sorted by date")
- Specify time ranges explicitly

---

Type **"/schema"** to see detailed database schema information.
"""
        await cl.Message(content=help_message).send()
        return

    if question.lower() == "/schema":
        schema_message = """### 🗄️ Database Schema Details

**Film Table**
- `film_id`: Primary key
- `title`: Film title
- `description`: Film description
- `release_year`: Year of release
- `language_id`: Foreign key to language
- `rental_duration`: Rental period in days
- `rental_rate`: Cost per rental
- `length`: Duration in minutes
- `replacement_cost`: Cost to replace
- `rating`: MPAA rating (G, PG, PG-13, R, NC-17)
- `special_features`: Array of special features

**Actor Table**
- `actor_id`: Primary key
- `first_name`: Actor's first name
- `last_name`: Actor's last name

**Customer Table**
- `customer_id`: Primary key
- `store_id`: Foreign key to store
- `first_name`: Customer's first name
- `last_name`: Customer's last name
- `email`: Email address
- `address_id`: Foreign key to address
- `active`: Active status
- `create_date`: Registration date

**Rental Table**
- `rental_id`: Primary key
- `rental_date`: Date/time of rental
- `inventory_id`: Foreign key to inventory
- `customer_id`: Foreign key to customer
- `return_date`: Date/time of return
- `staff_id`: Foreign key to staff

**Payment Table**
- `payment_id`: Primary key
- `customer_id`: Foreign key to customer
- `staff_id`: Foreign key to staff
- `rental_id`: Foreign key to rental
- `amount`: Payment amount
- `payment_date`: Date/time of payment

**Store Table**
- `store_id`: Primary key
- `manager_staff_id`: Foreign key to staff
- `address_id`: Foreign key to address

**Staff Table**
- `staff_id`: Primary key
- `first_name`: Staff member's first name
- `last_name`: Staff member's last name
- `address_id`: Foreign key to address
- `email`: Email address
- `store_id`: Foreign key to store
- `active`: Active status
- `username`: Login username

**Inventory Table**
- `inventory_id`: Primary key
- `film_id`: Foreign key to film
- `store_id`: Foreign key to store

**Category Table**
- `category_id`: Primary key
- `name`: Category name (Action, Comedy, Drama, etc.)

**Relationships**
- Films ↔ Actors (via film_actor)
- Films ↔ Categories (via film_category)
- Films ↔ Inventory ↔ Rentals
- Customers ↔ Rentals ↔ Payments
- Stores ↔ Staff, Inventory, Customers

---

Type **"/help"** for troubleshooting tips.
"""
        await cl.Message(content=schema_message).send()
        return

    try:
        # 1. Rate limiting
        try:
            rate_limiter.check_rate_limit(session_id)
            remaining = rate_limiter.get_remaining(session_id)
            logger.debug(f"Rate limit check passed. {remaining} requests remaining.")
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            await cl.Message(
                content=(
                    f"⚠️ **Rate Limit Exceeded**\n\n"
                    f"{str(e)}\n\n"
                    f"You can make up to 10 requests per minute."
                )
            ).send()
            return

        # 2. Sanitize input
        try:
            question = sanitize_input(question)
            logger.info(f"Processing question: {question[:50]}... (model={model})")
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            await cl.Message(
                content=(
                    f"❌ **Invalid Input**\n\n"
                    f"{str(e)}\n\n"
                    f"Please provide a valid question."
                )
            ).send()
            return

    except Exception as e:
        logger.error(f"Pre-processing error: {e}")
        await cl.Message(content=f"❌ Error processing request: {str(e)}").send()
        return

    try:
        # ── Step 1: Retrieve context + generate SQL ──────────────────────────
        async with cl.Step(name="🔍 Retrieving context + generating SQL") as step:
            step.input = question

            try:
                # Generate SQL
                result = generate_sql(question, model=model)
                sql = result["sql"]
                step.output = f"```sql\n{sql}\n```"
            except EmptyRetrievalError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **No Relevant Context Found**\n\n"
                        "The retrieval system couldn't find relevant schema or examples. "
                        "This might mean the vector store is empty.\n\n"
                        "**Solution:** Run the embedding script:\n"
                        "```bash\n"
                        "uv run python scripts/embed_schema_and_examples.py\n"
                        "```"
                    )
                ).send()
                return
            except ChromaDBError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **Vector Store Error**\n\n"
                        f"Failed to retrieve context: {str(e)}\n\n"
                        "**Solution:** Check if ChromaDB is accessible and data is embedded."
                    )
                ).send()
                return
            except ModelNotAvailableError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        f"❌ **Model Not Available**\n\n"
                        f"The model `{model}` is not available.\n\n"
                        f"**Solution:** Pull the model:\n"
                        f"```bash\n"
                        f"ollama pull {model}\n"
                        f"```"
                    )
                ).send()
                return
            except LLMConnectionError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **Cannot Connect to Ollama**\n\n"
                        f"{str(e)}\n\n"
                        "**Solution:** Start Ollama:\n"
                        "```bash\n"
                        "ollama serve\n"
                        "```"
                    )
                ).send()
                return
            except LLMTimeoutError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **LLM Request Timed Out**\n\n"
                        f"{str(e)}\n\n"
                        "**Solution:** Try a simpler question or increase the timeout in settings."
                    )
                ).send()
                return
            except InvalidSQLError as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **Invalid SQL Generated**\n\n"
                        f"{str(e)}\n\n"
                        "**Solution:** Try rephrasing your question or provide more details."
                    )
                ).send()
                return
            except (LLMError, RetrievalError) as e:
                step.output = f"❌ {str(e)}"
                step.is_error = True
                await cl.Message(
                    content=(
                        "❌ **Error Generating SQL**\n\n"
                        f"{str(e)}\n\n"
                        "Please try again or rephrase your question."
                    )
                ).send()
                return

        # ── Step 2: Execute SQL ───────────────────────────────────────────────
        async with cl.Step(name="⚙️ Executing SQL against PostgreSQL") as step:
            step.input = sql
            execution = execute_sql(sql)

            if execution["success"]:
                step.output = f"✅ {execution['row_count']} rows returned"
            else:
                error_type = execution.get("error_type", "unknown")
                step.output = f"❌ {error_type}: {execution['error']}"
                step.is_error = True

        # ── Step 3: Display Results ───────────────────────────────────────────
        formatted = format_results(execution)

        if execution["success"]:
            await cl.Message(
                content=(
                    f"### 🧠 Generated SQL\n"
                    f"```sql\n{sql}\n```\n\n"
                    f"### 📊 Results\n"
                    f"```\n{formatted}\n```\n\n"
                    f"*Model: `{model}`*"
                )
            ).send()
        else:
            # Handle SQL execution errors
            error_type = execution.get("error_type", "unknown")
            error_msg = execution.get("error", "Unknown error")

            error_guidance = {
                "connection": "Failed to connect to the database. Check if PostgreSQL is running.",
                "timeout": "Query took too long. Try simplifying your question.",
                "syntax": "The generated SQL has syntax errors. Try rephrasing your question.",
            }

            guidance = error_guidance.get(
                error_type, "An unexpected error occurred. Please try again."
            )

            await cl.Message(
                content=(
                    f"### 🧠 Generated SQL\n"
                    f"```sql\n{sql}\n```\n\n"
                    f"### ❌ Execution Failed\n"
                    f"**Error:** {error_msg}\n\n"
                    f"**Guidance:** {guidance}\n\n"
                    f"*Model: `{model}`*"
                )
            ).send()

    except Exception as e:
        logger.error(f"Unexpected error in message handler: {e}", exc_info=True)
        await cl.Message(
            content=(
                "❌ **Unexpected Error**\n\n"
                f"An unexpected error occurred: {str(e)}\n\n"
                "Please try again or contact support if the issue persists."
            )
        ).send()
