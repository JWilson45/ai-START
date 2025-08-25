"""Graph definition for the AI chat application."""
import os
import sqlite3
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from .memory_manager import MemoryManager, messages_stats
from .tools import run_bash, ingest_documents, search_corpus
from .config import MAX_MODEL_TOKENS, MAX_STORED_MESSAGES, SUMMARY_CHUNK_SIZE, TOOL_OUTPUT_CHAR_LIMIT

logger = logging.getLogger(__name__)

# --- Custom State definition ---
class State(TypedDict):
    # Append-only channel used for within-turn tool routing (Human → AI → Tool → AI).
    messages: Annotated[list, add_messages]
    # Our authoritative, compacted conversation memory that we fully replace each turn.
    history: list

# --- Model + tools (native tool calling) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [run_bash, ingest_documents, search_corpus]
llm_with_tools = llm.bind_tools(tools)

# --- Memory management helpers (trim, summarize, clamp tool outputs) ---
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
    state_dir = os.path.join(os.path.dirname(__file__), "..", ".state")
    os.makedirs(state_dir, exist_ok=True)
    db_path = os.path.join(state_dir, "langgraph.sqlite")

    cp = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    logger.info("SQLite checkpointer initialized at %s", db_path)
    return builder.compile(checkpointer=cp)
