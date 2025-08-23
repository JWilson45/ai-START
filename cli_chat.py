import subprocess
import json
import os
import sqlite3
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
# --- Custom state typing ---
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.INFO)

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

# --- Memory management helpers (trim, summarize, clamp tool outputs) ---
MAX_MODEL_TOKENS = 3000        # ~60–70% of context window
MAX_STORED_MESSAGES = 40       # keep recent conversational detail compact
SUMMARY_CHUNK_SIZE = 20        # how many early messages to compress at a time
TOOL_OUTPUT_CHAR_LIMIT = 4000  # clamp tool outputs stored in history

SUMMARY_PREFIX = "[SUMMARY]"

def _is_summary_message(m):
  return isinstance(m, SystemMessage) and isinstance(m.content, str) and m.content.startswith(SUMMARY_PREFIX)

def _collapse_tool_cycles(messages):
  """Keep only the most recent assistant-with-tools group (and its tool results)
  since the last HumanMessage; drop earlier assistant-with-tools groups and their
  tool messages within the same turn. This prevents multiple cycles from bloating
  the prompt and avoids dangling tool_calls.
  """
  # Find last HumanMessage (start of current turn)
  last_human_idx = None
  for i in range(len(messages) - 1, -1, -1):
    if isinstance(messages[i], HumanMessage):
      last_human_idx = i
      break

  seg_start = 0 if last_human_idx is None else last_human_idx + 1
  prefix = messages[:seg_start]
  segment = messages[seg_start:]
  if not segment:
    return messages

  # Identify the LAST assistant-with-tools in the segment
  last_ai_idx = None
  last_ids = []
  for j in range(len(segment) - 1, -1, -1):
    m = segment[j]
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
      ids = []
      try:
        for tc in m.tool_calls:
          tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
          if tc_id:
            ids.append(tc_id)
      except Exception:
        pass
      last_ai_idx = j
      last_ids = ids
      break

  if last_ai_idx is None:
    return messages

  need = set(last_ids)
  new_segment = []
  # Pre-section: drop earlier assistant-with-tools and any ToolMessages
  for k in range(0, last_ai_idx):
    m = segment[k]
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
      continue
    if isinstance(m, ToolMessage):
      continue
    new_segment.append(m)

  # Keep the chosen assistant-with-tools
  new_segment.append(segment[last_ai_idx])

  # Collect its matching tool outputs that appear after it
  chosen_tools = []
  chosen_ids = set()
  tail_other = []
  for k in range(last_ai_idx + 1, len(segment)):
    m = segment[k]
    if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in need:
      chosen_tools.append(m)
      chosen_ids.add(getattr(m, "tool_call_id", None))
    else:
      tail_other.append(m)

  # Append tools then tail (excluding duplicates)
  new_segment.extend(chosen_tools)
  for m in tail_other:
    if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in chosen_ids:
      continue
    new_segment.append(m)

  collapsed = prefix + new_segment
  if len(collapsed) != len(messages):
    logger.info("Collapse tool cycles: before=%s -> after=%s", _messages_stats(messages), _messages_stats(collapsed))
  return collapsed

def _truncate_tool_outputs(messages):
  """Clamp ToolMessage content so history doesn't explode."""
  trimmed = []
  for m in messages:
    if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > TOOL_OUTPUT_CHAR_LIMIT:
      logger.info("Clamp ToolMessage: name=%s id=%s original_len=%d -> %d", getattr(m, "name", None), getattr(m, "tool_call_id", None), len(m.content), TOOL_OUTPUT_CHAR_LIMIT)
      trimmed.append(ToolMessage(
        content=m.content[:TOOL_OUTPUT_CHAR_LIMIT] + "\n…(truncated)",
        name=getattr(m, "name", None),
        tool_call_id=getattr(m, "tool_call_id", None),
      ))
    else:
      trimmed.append(m)
  return trimmed

def _ensure_summary_message(messages):
  """Return (summary_msg_or_None, non_summary_messages)."""
  if messages and _is_summary_message(messages[0]):
    return messages[0], messages[1:]
  return None, messages

def _sanitize_for_summary(messages):
  """Strip tool-only structures so the summarizer never sees invalid schemas.
  - Drop ToolMessage entirely.
  - For AIMessage with tool_calls, keep only its textual content (no tool_calls).
  - Keep Human/System messages as-is.
  """
  sanitized = []
  dropped_tools = 0
  downgraded_ai = 0
  for m in messages:
    if isinstance(m, ToolMessage):
      dropped_tools += 1
      continue
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
      # keep just the text content to avoid OpenAI schema complaints during summarize
      sanitized.append(AIMessage(content=str(getattr(m, "content", ""))))
      downgraded_ai += 1
    else:
      sanitized.append(m)
  if dropped_tools or downgraded_ai:
    logger.info("Sanitize for summary: dropped_tool_msgs=%d downgraded_ai_with_tools=%d", dropped_tools, downgraded_ai)
  return sanitized

def _summarize_chunk_if_needed(messages):
  """If history is long, summarize earliest chunk into a running SystemMessage."""
  summary_msg, rest = _ensure_summary_message(messages)
  non_summary = list(rest)

  if len(non_summary) <= MAX_STORED_MESSAGES:
    return messages  # nothing to do

  # Identify the slice to summarize from the *start* of non_summary
  chunk = non_summary[:SUMMARY_CHUNK_SIZE]
  remainder = non_summary[SUMMARY_CHUNK_SIZE:]

  before = _messages_stats(messages)
  logger.info("Summarizing history: before=%s; chunk=%d; remainder=%d", before, len(chunk), len(remainder))

  # Build a concise additive summary with the base LLM (no tools)
  sanitized_chunk = _sanitize_for_summary(chunk)
  summary_prompt = [
    SystemMessage(content=(
      "Condense the following conversation chunk into 8-10 bullet points focusing on facts, decisions, "+
      "open questions, and user preferences. Be specific and avoid fluff."
    ))
  ] + sanitized_chunk

  try:
    new_summary_text = llm.invoke(summary_prompt).content
  except Exception as e:
    new_summary_text = f"(failed to summarize chunk due to {e})"

  preview = (new_summary_text or "").replace("\n", " ")[:200]
  logger.info("Summary chunk created: %d chars; preview=\"%s\"", len(new_summary_text or ""), preview)

  combined_text = new_summary_text if summary_msg is None else (
    f"{summary_msg.content}\n\n{new_summary_text}"
  )

  new_summary_msg = SystemMessage(content=f"{SUMMARY_PREFIX}\n{combined_text}")

  after = _messages_stats([new_summary_msg] + remainder)
  logger.info("Summarization applied: after=%s", after)

  # Return updated history: [SUMMARY] + remainder
  return [new_summary_msg] + remainder

def _trim_for_model(messages):
  """Create a model-facing view within token budget without mutating stored history.
  Atomic keep: ensure the most recent assistant-with-tools and its matching tool
  results form a contiguous block at the END of the view before trimming. This
  lets strategy='last' keep the whole pair and drop earlier history, avoiding
  new orphans at the cut line.
  """
  before = _messages_stats(messages)

  # --- Build an order where the protected tool block is last ---
  msgs = list(messages)
  # Find last assistant-with-tools
  ai_idx = None
  ids = []
  for i in range(len(msgs) - 1, -1, -1):
    m = msgs[i]
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
      # collect ids
      tmp = []
      try:
        for tc in m.tool_calls:
          tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
          if tc_id:
            tmp.append(tc_id)
      except Exception:
        pass
      ai_idx = i
      ids = tmp
      break

  reordered = msgs
  if ai_idx is not None and ids:
    need = set(ids)
    protected = [msgs[ai_idx]]
    tools = []
    tail_other = []
    for j in range(ai_idx + 1, len(msgs)):
      mj = msgs[j]
      if isinstance(mj, ToolMessage) and getattr(mj, "tool_call_id", None) in need:
        tools.append(mj)
      else:
        tail_other.append(mj)
    protected.extend(tools)
    # Put tail_other before protected so protected is at the very end
    reordered = msgs[:ai_idx] + tail_other + protected
    if len(reordered) != len(msgs):
      logger.info("Atomic keep reorder applied: before=%s -> after=%s", before, _messages_stats(reordered))

  try:
    trimmed = trim_messages(
      messages=reordered,
      max_tokens=MAX_MODEL_TOKENS,
      strategy="last",
      allow_partial=False,
      token_counter=llm,
    )
    after = _messages_stats(trimmed)
    logger.info("Trim for model: before=%s -> after=%s (token_budget=%d)", before, after, MAX_MODEL_TOKENS)
    return trimmed
  except Exception as e:
    # If token counting fails, fall back to a simple tail cut
    fallback = reordered[-MAX_STORED_MESSAGES:]
    after = _messages_stats(fallback)
    logger.warning("Trim fallback due to %s: before=%s -> after=%s (last %d messages)", e, before, after, MAX_STORED_MESSAGES)
    return fallback

def _repair_for_openai(messages):
  """
  OpenAI requires that any ToolMessage must be preceded by an assistant
  message that contains a matching tool_call id. If trimming cut that
  assistant message out, we drop the orphan ToolMessage to avoid 400 errors.
  Additionally, drop assistant messages with tool_calls whose ids are not satisfied
  by following ToolMessages in the same view.
  """
  # Robust normalizer for assistant/tool pairing across the entire view.
  # Strategy:
  # 1) Walk the list and, for each assistant-with-tools group, collect the contiguous tail until
  #    the next assistant. Within that tail, gather ToolMessages that match the group's ids.
  # 2) If ALL ids are satisfied, emit: [assistant_with_tools, matching_tool_messages...] in order,
  #    and keep any tail messages that are not those tools.
  #    If NOT satisfied, drop the assistant and any tool messages for its ids.
  # 3) Outside of any group, drop orphan ToolMessages (no prior assistant ids).
  cleaned = []
  i = 0
  n = len(messages)
  while i < n:
    m = messages[i]
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
      # Start a group for this assistant-with-tools
      group_ids = []
      try:
        for tc in m.tool_calls:
          tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
          if tc_id:
            group_ids.append(tc_id)
      except Exception:
        pass
      need = set(group_ids)
      found = set()
      tools_for_group = []
      tail_other = []

      j = i + 1
      while j < n and not (isinstance(messages[j], AIMessage)):
        tm = messages[j]
        if isinstance(tm, ToolMessage) and getattr(tm, "tool_call_id", None) in need:
          tools_for_group.append(tm)
          found.add(getattr(tm, "tool_call_id", None))
        else:
          tail_other.append(tm)
        j += 1

      if need and need.issubset(found):
        # Valid group: emit assistant, then its tools, then the rest of the tail (excluding those tools)
        cleaned.append(m)
        cleaned.extend(tools_for_group)
        # Keep non-matching tail content (exclude any duplicates of the tools we just appended)
        for x in tail_other:
          if not (isinstance(x, ToolMessage) and getattr(x, "tool_call_id", None) in found):
            cleaned.append(x)
      else:
        # Invalid group: drop assistant and any dangling tools for its ids
        logger.info("OpenAI repair: dropping assistant-with-tools; missing=%s", list(need - found))
        # Keep only tail content that is not one of the group's tool ids
        for x in tail_other:
          if not (isinstance(x, ToolMessage) and getattr(x, "tool_call_id", None) in need):
            cleaned.append(x)
      i = j
    else:
      # Outside a group: drop orphan ToolMessages with no prior assistant ids in 'cleaned'
      if isinstance(m, ToolMessage):
        # Check if we have an assistant-with-tools for this id earlier in cleaned
        tid = getattr(m, "tool_call_id", None)
        seen_valid_id = False
        for prev in reversed(cleaned):
          if isinstance(prev, AIMessage) and getattr(prev, "tool_calls", None):
            ids_prev = []
            try:
              for tc in prev.tool_calls:
                tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                if tc_id:
                  ids_prev.append(tc_id)
            except Exception:
              pass
            if tid in ids_prev:
              seen_valid_id = True
              break
          # Stop scanning if we hit a Human or other break in the chain
          if isinstance(prev, HumanMessage):
            break
        if seen_valid_id:
          cleaned.append(m)
        else:
          # Orphan tool message without a visible assistant-with-tools context; drop it
          logger.info("OpenAI repair: dropping orphan ToolMessage id=%s", tid)
      else:
        cleaned.append(m)
      i += 1

  return cleaned

# --- Graph nodes ---
def call_model_node(state: State):
    # Build prompt input with accumulated messages

    # Compute the current-turn slice (from the last HumanMessage to the end)
    def _slice_current_turn(msgs):
        last_human_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                last_human_idx = i
                break
        return msgs[last_human_idx:] if last_human_idx is not None else msgs

    prior_history = state.get("history", []) or []
    full_msgs = state.get("messages", []) or []
    current_turn = _slice_current_turn(full_msgs)

    # Working set for this turn = compacted history + current turn
    working_messages = prior_history + current_turn
    # Collapse multiple tool-call cycles within the current turn
    working_messages = _collapse_tool_cycles(working_messages)

    # Clean and compact stored history before using it
    logger.info("Phase: start; state=%s", _messages_stats(working_messages))
    cleaned_messages = _truncate_tool_outputs(working_messages)  # clamp big tool outputs
    logger.info("Phase: after clamp; state=%s", _messages_stats(cleaned_messages))
    compact_messages = _summarize_chunk_if_needed(cleaned_messages)  # roll-up old context
    logger.info("Phase: after summarize; state=%s", _messages_stats(compact_messages))

    # Use a token-trimmed view for the actual model call
    model_view = _trim_for_model(compact_messages)
    model_view = _repair_for_openai(model_view)
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