import subprocess
import json
import os
import sqlite3
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
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

from memory_manager import MemoryManager

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Custom State definition ---
class State(TypedDict):
    # Append-only channel used for within-turn tool routing (Human → AI → Tool → AI).
    messages: Annotated[list, add_messages]
    # Our authoritative, compacted conversation memory that we fully replace each turn.
    history: list

def _messages_stats(messages):
    total = len(messages)
    chars = 0
    tool_msgs = 0
    ai_with_tools = 0
    for m in messages:
        content = getattr(m, "content", "")
        try:
            chars += len(content) if isinstance(content, str) else 0
        except Exception:
            pass
        from langchain_core.messages import ToolMessage, AIMessage
        if isinstance(m, ToolMessage):
            tool_msgs += 1
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            ai_with_tools += 1
    return {"count": total, "chars": chars, "tool_msgs": tool_msgs, "ai_with_tools": ai_with_tools}

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

# --- Model + tools (native tool calling) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="llama3.2:3b", temperature=0)
# llm = ChatOllama(model="gpt-oss:20b", temperature=0)

tools = [run_bash]
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
    logger.info("Phase: start; state=%s", _messages_stats(working_messages))

    # Summarize history if needed (roll-up old context)
    compact_messages = mm.summarize_history(working_messages)
    logger.info("Phase: after summarize; state=%s", _messages_stats(compact_messages))

    # Build the model-facing view
    model_view = mm.prepare_for_model(compact_messages)
    logger.info("Phase: model_view ready; view=%s", _messages_stats(model_view))

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
    logger.info("Phase: persist; new_history=%s", _messages_stats(new_history))
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