import subprocess
import json
import os
import sqlite3
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
# --- Custom state typing ---
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from memory_manager import MemoryManager, messages_stats

import glob
import hashlib
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Custom State definition ---
class State(TypedDict):
    # Append-only channel used for within-turn tool routing (Human → AI → Tool → AI).
    messages: Annotated[list, add_messages]
    # Our authoritative, compacted conversation memory that we fully replace each turn.
    history: list


# Quiet noisy third-party logs while keeping our own DEBUG/INFO as configured
for name, level in {"httpx": logging.WARNING, "openai": logging.WARNING, "langchain": logging.INFO}.items():
    logging.getLogger(name).setLevel(level)

load_dotenv()

# --- Tool: run_bash (human approval required) ---
@tool
def run_bash(command: str) -> str:
    """
    Run ONE safe shell command on the user's machine when live output is required or explicitly requested.

    Rules:
    - Use only for local inspection tasks (files here, top process, disk usage, hostname/IP, env vars, version checks).
    - Do NOT use for explanations or examples—answer in text instead.
    - Single command only; no chaining, pipes, redirects, subshells.
    - If ambiguous, ask for clarification instead of guessing.
    - Always pauses for approval before running.
    """
    decision = interrupt({"tool": "run_bash", "command": command})
    approved = bool(decision.get("approve")) if isinstance(decision, dict) else False

    if approved:
        # Safety: block compound or redirection operators; require a single, simple command
        forbidden_tokens = ["&&", ";", "|", "`", "$(", ">>", ">", "<", "\n"]
        if any(tok in command for tok in forbidden_tokens):
            return json.dumps({
                "status": "denied",
                "command": command,
                "reason": "compound_or_redirection_operators_not_allowed",
            })
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout or result.stderr or "(no output)"
            return json.dumps({
                "status": "executed",
                "command": command,
                "returncode": result.returncode,
                "output": output,
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "command": command,
                "error": str(e),
            })

    reason = (decision.get("reason") or "").strip() if isinstance(decision, dict) else ""
    return json.dumps({
        "status": "denied",
        "command": command,
        "reason": reason or "no reason provided",
    })

# --- Vector store (Chroma) + RAG helpers ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_COLLECTION = "kb_main"

import re
COLL_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]{1,510})[a-zA-Z0-9]$")

def _sanitize_collection_name(name: str) -> str:
    """Coerce a user-provided collection name to Chroma's constraints.
    Rules: 3-512 chars, [a-zA-Z0-9._-], start/end alnum.
    """
    if not isinstance(name, str):
        name = str(name or "kb_main")
    # Replace disallowed chars with '-'
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    # Ensure starts/ends with alnum
    if not name or not name[0].isalnum():
        name = f"k{name}"
    if not name[-1].isalnum():
        name = f"{name}0"
    # Enforce min length 3
    while len(name) < 3:
        name += "0"
    # Enforce max length 512
    name = name[:512]
    # Final check; if still invalid, fall back
    if not COLL_RE.match(name):
        return "kb_main"
    return name


def _ensure_persist_dir(subdir: str) -> str:
    base = os.path.join(os.path.dirname(__file__), ".state", subdir)
    os.makedirs(base, exist_ok=True)
    return base


def _vectorstore(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """
    Return a persistent Chroma vector store located under ./.state/chroma
    configured with OpenAI embeddings (text-embedding-3-small).
    """
    original = collection_name
    collection_name = _sanitize_collection_name(collection_name)
    if original != collection_name:
        logger.info("Sanitized collection name: '%s' -> '%s'", original, collection_name)
    persist_directory = _ensure_persist_dir("chroma")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        logger.info("Chroma vectorstore ready: collection=%s dir=%s", collection_name, persist_directory)
        return vs
    except Exception as e:
        logger.exception("Vectorstore init failed: %s", e)
        raise


def _expand_paths(path_or_glob: str) -> list[str]:
    # Accept file, directory, or glob. Return concrete file list.
    p = os.path.expanduser(path_or_glob)
    if os.path.isdir(p):
        # Recurse common doc types inside a directory
        files = []
        for ext in ("**/*.pdf", "**/*.txt", "**/*.md"):
            files.extend(glob.glob(os.path.join(p, ext), recursive=True))
        return sorted(files)
    # Glob or single file
    matches = glob.glob(p, recursive=True)
    return sorted([m for m in matches if os.path.isfile(m)])


def _load_documents(paths: list[str]):
    docs = []
    for fp in paths:
        try:
            if fp.lower().endswith(".pdf"):
                loader = PyPDFLoader(fp)
                docs.extend(loader.load())
            elif fp.lower().endswith((".txt", ".md")):
                loader = TextLoader(fp, encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            logger.warning("Skipping %s (%s)", fp, e)
    return docs


def _chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _ids_for_docs(docs) -> list[str]:
    ids = []
    for d in docs:
        src = str(d.metadata.get("source", ""))
        page = str(d.metadata.get("page", ""))
        h = hashlib.sha1((d.page_content + "|" + src + "|" + page).encode("utf-8", errors="ignore")).hexdigest()
        ids.append(h)
    return ids


@tool
def ingest_documents(path: str, collection: str = DEFAULT_COLLECTION) -> str:
    """
    Ingest local files into the persistent Chroma vector store.

    Args:
    - path: A file path, directory, or glob (supports .pdf, .txt, .md)
    - collection: Optional Chroma collection name (default: "kb")

    Behavior:
    - Loads PDFs via PyPDFLoader and text/Markdown via TextLoader
    - Splits into ~1000-char chunks with 150 overlap
    - Embeds with OpenAI `text-embedding-3-small` and upserts into Chroma
    - Deduplicates by content hash-based IDs
    """
    logger.info("Ingest starting: path=%s collection=%s", path, collection)
    files = _expand_paths(path)
    logger.info("Expand paths -> %d files: %s", len(files), files)
    if not files:
        return json.dumps({
            "status": "no_files_found",
            "path": path,
        })

    raw_docs = _load_documents(files)
    logger.info("Loaded %d docs", len(raw_docs))
    if not raw_docs:
        return json.dumps({
            "status": "load_failed",
            "path": path,
            "files": files,
        })

    chunks = _chunk_documents(raw_docs)
    logger.info("Chunked into %d chunks", len(chunks))
    if not chunks:
        return json.dumps({
            "status": "no_chunks",
            "path": path,
        })

    try:
        vs = _vectorstore(collection)
        ids = _ids_for_docs(chunks)
        logger.info("Generated %d ids", len(ids))
        vs.add_documents(documents=chunks, ids=ids)
        logger.info("Upserted %d chunks to Chroma collection=%s", len(chunks), collection)
    except Exception as e:
        # Common causes: missing OPENAI_API_KEY, network error, incompatible chroma/langchain versions
        api_key = os.getenv("OPENAI_API_KEY", "")
        masked = (api_key[:6] + "…") if api_key else "(missing)"
        logger.exception("Ingest failed during vectorstore/add_documents: %s", e)
        return json.dumps({
            "status": "error",
            "stage": "vectorstore_or_upsert",
            "reason": str(e),
            "openai_api_key": masked,
            "collection": collection,
        })
    # Chroma persists automatically when persist_directory is set

    return json.dumps({
        "status": "ingested",
        "path": path,
        "files_count": len(files),
        "chunks_upserted": len(chunks),
        "collection": collection,
        "persist_dir": _ensure_persist_dir("chroma"),
    })


@tool
def search_corpus(query: str, k: int = 5, collection: str = DEFAULT_COLLECTION) -> str:
    """
    Retrieve top-k chunks from the Chroma vector store and return structured results.

    Args:
    - query: The search query
    - k: number of results (default 5)
    - collection: Chroma collection name (default "kb")

    Returns JSON with fields: query, k, results=[{score, snippet, source, page}]
    """
    if not query or not query.strip():
        return json.dumps({"status": "error", "reason": "empty_query"})

    vs = _vectorstore(collection)
    try:
        docs_scores = vs.similarity_search_with_score(query, k=k)
        logger.info("Search: query='%s' k=%d -> %d hits", query, k, len(docs_scores))
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return json.dumps({"status": "error", "reason": str(e)})

    results = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        results.append({
            "score": float(score),
            "snippet": doc.page_content[:500],
            "source": meta.get("source"),
            "page": meta.get("page"),
        })

    return json.dumps({
        "status": "ok",
        "query": query,
        "k": k,
        "collection": collection,
        "results": results,
    })

# --- Model + tools (native tool calling) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="llama3.2:3b", temperature=0)
# llm = ChatOllama(model="gpt-oss:20b", temperature=0)

tools = [run_bash, ingest_documents, search_corpus]
llm_with_tools = llm.bind_tools(tools)

# --- Memory management helpers (trim, summarize, clamp tool outputs) ---
MAX_MODEL_TOKENS = 3000        # ~60–70% of context window
MAX_STORED_MESSAGES = 40       # keep recent conversational detail compact
SUMMARY_CHUNK_SIZE = 20        # how many early messages to compress at a time
TOOL_OUTPUT_CHAR_LIMIT = 4000  # clamp tool outputs stored in history

mm = MemoryManager(
    max_model_tokens=MAX_MODEL_TOKENS,
    max_stored_messages=MAX_STORED_MESSAGES,
    summary_chunk_size=SUMMARY_CHUNK_SIZE,
    tool_output_char_limit=TOOL_OUTPUT_CHAR_LIMIT,
    llm=llm,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You talk like a pirate. Keep replies concise.

TOOL RULES:
- Default to chat answers. Only call `run_bash` if live machine output is required (list files, top process, disk, hostname/IP, env, version check) or if user explicitly asks.
- Pass exactly ONE safe command; no chaining/pipes/redirects/subshells.
- At most one tool call per turn; if more is needed, ask first.
- If ambiguous or risky, ask for clarification.

RAG RULES:
- To **ingest** user documents into the knowledge base, call `ingest_documents(path, collection?)` where `path` may be a file, directory, or glob (supports .pdf/.txt/.md). Use chunk_size≈1000 and overlap≈150 (handled by the tool).
- To **answer questions about the user's docs**, first call `search_corpus(query, k=5, collection?)`. Then write the answer **using only retrieved snippets**. Add a short **Sources** list mapping to each snippet's `source` (and `page` if present).
- If retrieval returns nothing useful, say so and proceed with a normal chat answer (no hallucinations).
- If ingestion/search errors persist, you may call `diag_vectorstore()` once to report environment issues (API key, Chroma init).
"""),
    MessagesPlaceholder("messages"),
])

# --- Graph nodes ---
def call_model_node(state: State):
    # Build prompt input with accumulated messages

    prior_history = state.get("history", []) or []
    full_msgs = state.get("messages", []) or []
    current_turn = mm.slice_current_turn(full_msgs)

    # Working set for this turn = compacted history + current turn
    working_messages = prior_history + current_turn
    logger.info("Phase: start; state=%s", messages_stats(working_messages))

    # Summarize history if needed (roll-up old context)
    compact_messages = mm.summarize_history(working_messages)
    logger.info("Phase: after summarize; state=%s", messages_stats(compact_messages))

    # Build the model-facing view
    model_view = mm.prepare_for_model(compact_messages)
    logger.info("Phase: model_view ready; view=%s", messages_stats(model_view))

    prompt_input = {"messages": model_view}

    # Enforce a ONE-tool-call budget per user turn by switching to a no-tools model
    # if a ToolMessage already exists after the latest HumanMessage.
    # Count ToolMessages since the last HumanMessage.
    tool_calls_after_last_human = 0
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            tool_calls_after_last_human += 1
    tools_budget_exhausted = tool_calls_after_last_human >= 1

    model_to_use = llm if tools_budget_exhausted else llm_with_tools

    # Let the model respond with or without tools; preserve tool_calls for routing
    # Always wrap with OpenAI callback; if no OpenAI calls occur, totals will be zero
    with get_openai_callback() as cb:
        response = model_to_use.invoke(prompt.invoke(prompt_input))

    # Normalize to AIMessage
    if isinstance(response, AIMessage):
        msg = response
    else:
        msg = AIMessage(content=getattr(response, "content", str(response)))

    # Ensure we never return an empty reply unless a tool call is present
    if (not getattr(msg, "content", "").strip()) and not getattr(msg, "tool_calls", None):
        msg = AIMessage(content="Aye! How be ye?")

    logger.debug("call_model_node invoked, total messages: %d", len(state["messages"]))
    logger.debug("AI response: %s", msg.content)

    if getattr(cb, "total_tokens", 0) > 0:
        logger.info(
            "Token usage - prompt: %d, completion: %d, total: %d, cost: $%.4f",
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_tokens,
            cb.total_cost,
        )
    else:
        logger.info("Token usage - no OpenAI tokens recorded (non-OpenAI model or zero-token call)")

    # Replace authoritative history; only emit AI delta to messages for tool routing
    new_history = compact_messages + [msg]
    logger.info("Phase: persist; new_history=%s", messages_stats(new_history))
    return {"history": new_history, "messages": [msg]}

def build_app():
    builder = StateGraph(State)
    builder.add_node("call_model", call_model_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {
            "tools": "tools",
            "action": "tools",
            "__end__": END,
            "continue": END,
        },
    )
    builder.add_edge("tools", "call_model")
    # Ensure a persistent state directory exists and use SQLite checkpointer (Option A)
    state_dir = os.path.join(os.path.dirname(__file__), ".state")
    os.makedirs(state_dir, exist_ok=True)
    db_path = os.path.join(state_dir, "langgraph.sqlite")

    cp = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    logger.info("SQLite checkpointer initialized at %s", db_path)
    return builder.compile(checkpointer=cp)

# --- Interrupt helpers (clean, testable) ---

def _latest_ai_message(messages):
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def _print_ai(out) -> bool:
    last_ai = _latest_ai_message(out.get("messages", []))
    if last_ai and str(getattr(last_ai, "content", "")).strip():
        print("AI:", str(last_ai.content), "\n")
        return True
    return False


def _resume(app, config, payload):
    logger.debug("Resuming with payload: %s", payload)
    return app.invoke(Command(resume=payload), config)


def handle_interrupts(app, config, out):
    """Process any pending interrupts in a single place.

    Returns (out, printed_ai) where `out` is the updated graph output and
    `printed_ai` indicates whether an AI reply was printed as part of handling.
    """
    printed_ai = False
    while "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        logger.debug("Handling interrupt: %s", payload)
        if isinstance(payload, dict) and payload.get("tool") == "run_bash":
            cmd = payload.get("command", "")
            print(f"Tool request: run_bash\n$ {cmd}")
            yn = input("Approve this single command? (y/n): ").strip().lower()
            if yn == "y":
                out = _resume(app, config, {"approve": True})
                # Print tool output (latest ToolMessage from run_bash)
                for m in reversed(out.get("messages", [])):
                    if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                        print("\n―――― TOOL OUTPUT ――――")
                        print(m.content)
                        print("―――― END OUTPUT ――――\n")
                        break
                printed_ai = _print_ai(out)
            else:
                reason = input("Why deny? ").strip()
                out = _resume(app, config, {"approve": False, "reason": reason})
                # Print tool response (denied message from ToolMessage)
                for m in reversed(out.get("messages", [])):
                    if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                        print("\n―――― TOOL RESPONSE ――――")
                        print(m.content)
                        print("―――― END RESPONSE ――――\n")
                        break
                printed_ai = _print_ai(out)
        else:
            # Non-run_bash interrupts: resume once and exit loop
            out = _resume(app, config, {"approve": False, "reason": "Non-run_bash interrupt"})
            break
    return out, printed_ai

# --- CLI ---
def main():
    print("Welcome — pirate CLI. Type 'exit' to quit.")
    app = build_app()
    config = {"configurable": {"thread_id": "default"}}

    while True:
        user_input = input("You: ")
        logger.debug("User input: %s", user_input)
        if user_input.lower() in {"exit", "quit"}:
            break

        out = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        logger.debug("Graph output: %s", out)
        printed_ai = False

        # Handle any pending interrupts (tool approvals) in one place
        out, printed_ai = handle_interrupts(app, config, out)

        if not printed_ai:
            if not _print_ai(out):
                print("AI:", "Ahoy! How can I assist ye today?", "\n")

if __name__ == "__main__":
    main()