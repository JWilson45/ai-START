import subprocess
import json
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

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
llm = ChatOllama(model="llama3.2:3b", temperature=0)
# llm = ChatOllama(model="gpt-oss:20b", temperature=0)

tools = [run_bash]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You talk like a pirate. Keep replies concise.

TOOL RULES:
- Default to chat answers. Only call `run_bash` if live machine output is required (list files, top process, disk, hostname/IP, env, version check) or if user explicitly asks.
- Pass exactly ONE safe command; no chaining/pipes/redirects/subshells.
- At most one tool call per turn. If more is needed, ask first.
- If ambiguous or risky, ask for clarification.

EXAMPLES (no tool):
Q: "What does `ps` do?" → Explain in text.
Q: "How to check disk space?" → Suggest `df -h` in text.

EXAMPLES (tool):
Q: "List files in this folder" → `ls -la`
Q: "Show top process" → `ps aux`
Q: "$ whoami" → `whoami`
"""),
    MessagesPlaceholder("messages"),
])

# --- Graph nodes ---
def call_model_node(state: MessagesState):
    # Build prompt input with accumulated messages
    prompt_input = {"messages": state["messages"]}

    # Enforce a ONE-tool-call budget per user turn by switching to a no-tools model
    # if a ToolMessage already exists after the latest HumanMessage.
    # Count ToolMessages since the last HumanMessage.
    tool_calls_after_last_human = 0
    seen_human = False
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            seen_human = True
            break
        if isinstance(m, ToolMessage):
            tool_calls_after_last_human += 1
    tools_budget_exhausted = tool_calls_after_last_human >= 1

    model_to_use = llm if tools_budget_exhausted else llm_with_tools

    # Let the model respond with or without tools; preserve tool_calls for routing
    response = model_to_use.invoke(prompt.invoke(prompt_input))

    # Normalize to AIMessage
    if isinstance(response, AIMessage):
        msg = response
    else:
        msg = AIMessage(content=getattr(response, "content", str(response)))

    # Ensure we never return an empty reply unless a tool call is present
    if (not getattr(msg, "content", "").strip()) and not getattr(msg, "tool_calls", None):
        msg = AIMessage(content="Aye! How be ye?")

    return {"messages": state["messages"] + [msg]}

def build_app():
    builder = StateGraph(MessagesState)
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
    return builder.compile(checkpointer=InMemorySaver())

# --- CLI ---
def main():
    print("Welcome — pirate CLI. Type 'exit' to quit.")
    app = build_app()
    config = {"configurable": {"thread_id": "default"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        out = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        printed_ai = False

        # Handle interrupts (tool approvals) until resolved
        while "__interrupt__" in out:
            payload = out["__interrupt__"][0].value
            if isinstance(payload, dict) and payload.get("tool") == "run_bash":
                cmd = payload.get("command", "")
                print(f"Tool request: run_bash\n$ {cmd}")
                yn = input("Approve this single command? (y/n): ").strip().lower()
                if yn == "y":
                    out = app.invoke(Command(resume={"approve": True}), config)
                    # Print tool output (latest ToolMessage)
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                            print("\n―――― TOOL OUTPUT ――――")
                            print(m.content)
                            print("―――― END OUTPUT ――――\n")
                            break
                    # Immediately print assistant follow-up
                    last_ai = None
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, AIMessage):
                            last_ai = m
                            break
                    if last_ai and str(getattr(last_ai, "content", "")).strip():
                        print("AI:", last_ai.content, "\n")
                        printed_ai = True
                else:
                    reason = input("Why deny? ").strip()
                    out = app.invoke(Command(resume={"approve": False, "reason": reason}), config)
                    # Print tool response (denied message from ToolMessage)
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                            print("\n―――― TOOL RESPONSE ――――")
                            print(m.content)
                            print("―――― END RESPONSE ――――\n")
                            break
                    # Immediately print assistant follow-up
                    last_ai = None
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, AIMessage):
                            last_ai = m
                            break
                    if last_ai and str(getattr(last_ai, "content", "")).strip():
                        print("AI:", last_ai.content, "\n")
                        printed_ai = True
            else:
                # Non-run_bash interrupts are resumed once as "not approved" and then we break to print the model message
                out = app.invoke(Command(resume={"approve": False, "reason": "Non-run_bash interrupt"}), config)
                break

        if not printed_ai:
            last_ai = None
            for m in reversed(out.get("messages", [])):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            if last_ai and str(getattr(last_ai, "content", "")).strip():
                print("AI:", last_ai.content, "\n")
            else:
                print("AI:", "Ahoy! How can I assist ye today?", "\n")

if __name__ == "__main__":
    main()