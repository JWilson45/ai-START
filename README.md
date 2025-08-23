# ai-stack project
-e 
Arrgh, may yer code be as treasure!
May yer tests be as plentiful as sea monsters!
Fair winds and following seas!

# ai-stack

A minimal-but-powerful AI app scaffold built around **LangGraph** and a **CLI chat** experience. It showcases:

- Tool-enabled LLM chats (OpenAI-compatible models)
- A clean interrupt/approval flow for risky tools (e.g., `run_bash`)
- Memory management with message trimming + optional summarization
- Persistent checkpoints using SQLite
- Simple, testable structure with Poetry

---

## Features

- **CLI chat** with a friendly prompt and streaming output.
- **LangGraph state machine** that routes between the model and tools.
- **Tool calling** with explicit approval:
  - `run_bash` — executes shell commands **after user approval**.
  - `human_assistance` — pauses the graph and prompts for human input.
- **Message hygiene**: clamps large tool outputs, keeps the latest assistant-with-tools + its tool replies, drops orphans, and trims to fit model context.
- **Checkpointing** via `SqliteSaver` so you can resume sessions.
- **Configurable** LLM/model and limits through environment variables.

---

## Requirements

- **Python** 3.11+ (tested with 3.13)
- **Poetry** 1.6+
- An **OpenAI-compatible** API key (e.g., OpenAI)

> If you use `pyenv`, ensure your local Python matches the one Poetry will use.

---

## Quick Start (Poetry)

1) **Install Poetry** (skip if you already have it):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2) **Select a Python version** (optional, if using pyenv):

```bash
pyenv install -s 3.13.6
pyenv local 3.13.6
poetry env use python3.13
```

3) **Install dependencies**:

```bash
poetry install --no-root
```

4) **Configure environment**: create a `.env` file in the project root and set at least:

```dotenv
# Required
OPENAI_API_KEY=sk-...

# Optional
MODEL=gpt-4o-mini            # or gpt-4.1, etc.
MAX_MODEL_TOKENS=32768       # hard cap for context preparation
MAX_STORED_MESSAGES=40       # how many messages to retain before summarizing
DB_PATH=.checkpoints/sqlite.db
```

> If you don’t want a `.env`, export variables in your shell before running.

5) **Run the CLI chat**:

```bash
poetry run python cli_chat.py
```

You should see a banner and be able to type messages. When a tool call is proposed, you’ll be asked to **approve** it.

---

## How It Works

### Architecture at a glance

```
User ↔ CLI ↔ LangGraph (StateGraph)
                 ↙         ↘
             LLM Node    Tool Node
                 ↘       ↙
               Checkpointer (SQLite)
```

- **LLM Node**: Calls your configured model. It may request a tool.
- **Tool Node**: Executes tools. For `run_bash`, the graph interrupts for approval.
- **Message Prep**: Before each model call, messages are compacted: last tool cycle kept, tool outputs clamped, orphans removed.
- **Checkpointing**: Conversation state is stored in SQLite so you can resume later.

### Tools

- `run_bash(command: str)` — Executes a shell command **only after you confirm**. Outputs are captured and trimmed.
- `human_assistance(query: str)` — Pauses the graph and returns control to the CLI so a human can answer.

---

## Common Commands

```bash
# Start the chat
poetry run python cli_chat.py

# Run tests (if present)
poetry run pytest -q

# Lint/format (if configured)
poetry run ruff check .
poetry run ruff format .
```

---

## Troubleshooting

- **Poetry can’t find Python**: Ensure the Python version is installed and on PATH. With pyenv, run `pyenv which python` and then `poetry env use /path/to/python`.
- **`NameError: name 'tool' is not defined`**: Ensure `from langchain_core.tools import tool` is imported before using the decorator.
- **Model too chatty with tools**: Adjust your system prompt and/or tool docstrings to clarify when tools should be used.
- **Context trimming issues**: Verify your message preparation keeps the last assistant-with-tools **and** its tool outputs together.

---

## Configuration Reference

Environment variables you can use:

- `OPENAI_API_KEY` *(required)* — API key for your LLM provider.
- `MODEL` — model name; default may be set in code.
- `MAX_MODEL_TOKENS` — max tokens targeted during trimming.
- `MAX_STORED_MESSAGES` — how many messages to keep before summarizing.
- `DB_PATH` — SQLite path for the checkpointer.

---

## Contributing

PRs are welcome. Keep changes small and well-tested. Aim for minimalism and clear logs.

---

## License

MIT (see `LICENSE` if present)