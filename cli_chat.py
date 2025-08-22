import subprocess
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

# Helper: determine if the latest human explicitly requested a command
def user_explicitly_requests_command(state: MessagesState) -> bool:
    last_human: Optional[HumanMessage] = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    text = (last_human.content or "").lower() if last_human else ""
    allow_markers = (
        "run ", "execute", "bash", "shell", "command", "$ ",
        "approve", "tool:", "!", "sudo", "ls", "echo ", "cat ", "python ", "pip "
    )
    starts_markers = ("run", "bash", "sh", "$", "!")
    return text.strip().startswith(starts_markers) or any(k in text for k in allow_markers)

# Decide whether to route to ToolNode. Only allow if the latest HumanMessage
# clearly requests running a command or mentions shell-related intent.
def guarded_tools_condition(state: MessagesState):
    decision = tools_condition(state)
    if decision in ("tools", "action"):
        # Find the latest HumanMessage content
        last_human: Optional[HumanMessage] = None
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                last_human = m
                break
        text = (last_human.content or "").lower() if last_human else ""
        # Heuristic: only allow tool if user explicitly requests/permits it
        allow_markers = (
            "run ", "execute", "bash", "shell", "command", "$ ",
            "approve", "tool:", "!", "sudo", "ls", "echo ", "cat ", "python ", "pip "
        )
        starts_markers = ("run", "bash", "sh", "$", "!")
        allow = text.strip().startswith(starts_markers) or any(k in text for k in allow_markers)
        return "tools" if allow else "__end__"
    return decision

# --- Tool: run_bash (human approval required) ---
@tool
def run_bash(command: str) -> str:
    """
    Prepare ONE shell command to execute **only if** running it is necessary or explicitly requested. This tool will pause for human approval via interrupt before execution. If approval is denied, the tool should return the denial details and you should continue the conversation without re-calling the tool unless a clearly safer single-command alternative exists.
    """
    decision = interrupt({"tool": "run_bash", "command": command})
    approved = bool(decision.get("approve")) if isinstance(decision, dict) else False

    if approved:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout or result.stderr or "(no output)"
            return f"[EXECUTED]\n$ {command}\n{output}"
        except Exception as e:
            return f"[ERROR] $ {command}\n{e}"

    reason = (decision.get("reason") or "").strip() if isinstance(decision, dict) else ""
    return (
        f"[DENIED]\n$ {command}\n"
        f"Reason: {reason or 'no reason provided'}\n"
        "Please propose a safer/corrected command."
    )

# --- Model + tools (native tool calling) ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)
tools = [run_bash]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You talk like a pirate. Keep replies concise.
IMPORTANT TOOL POLICY:
- Only invoke the `run_bash` tool when executing a shell command is REQUIRED to complete the user's request or the user explicitly asks you to run a command.
- If a normal conversational answer suffices, DO NOT call any tool—just reply.
- When you do call `run_bash`, provide a single best-guess command. Do not chain multiple commands in one call.
- After a denied tool run, propose a safer/corrected single command only if you can justify it; otherwise return to normal conversation.
- If the user explicitly requests to run/execute a command (e.g., message starts with `$`, `!`, or the words `run`, `bash`, `execute`), you MUST call the `run_bash` tool with that single command.
"""),
    MessagesPlaceholder("messages"),
])

# --- Graph nodes ---
def call_model_node(state: MessagesState):
    # Build prompt input with accumulated messages
    prompt_input = {"messages": state["messages"]}

    # Let the model respond with or without tools; preserve tool_calls for routing
    response = llm_with_tools.invoke(prompt.invoke(prompt_input))

    # If the model attempted a tool call but the user did NOT request execution, re-ask WITHOUT tools
    if isinstance(response, AIMessage) and getattr(response, "tool_calls", None) and not user_explicitly_requests_command(state):
        response = llm.invoke(prompt.invoke(prompt_input))

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
        guarded_tools_condition,
        {
            "tools": "tools",     # common key => run ToolNode
            "action": "tools",    # some builds return "action"
            "__end__": END,       # no tool call => finish
            "continue": END,      # legacy key for no tool
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

        # Handle interrupts (tool approvals) until resolved
        while "__interrupt__" in out:
            payload = out["__interrupt__"][0].value
            if isinstance(payload, dict) and payload.get("tool") == "run_bash":
                cmd = payload.get("command", "")
                print(f"Tool request: run_bash\n$ {cmd}")
                yn = input("Approve? (y/n): ").strip().lower()
                if yn == "y":
                    out = app.invoke(Command(resume={"approve": True}), config)
                    # If the tool ran, print its output directly (latest ToolMessage)
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                            print(m.content)
                            break
                else:
                    reason = input("Why deny? ").strip()
                    out = app.invoke(Command(resume={"approve": False, "reason": reason}), config)
                    # If the tool ran, print its output directly (latest ToolMessage)
                    for m in reversed(out.get("messages", [])):
                        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                            print(m.content)
                            break
            else:
                # Non-run_bash interrupts are resumed once as "not approved" and then we break to print the model message
                out = app.invoke(Command(resume={"approve": False, "reason": "Non-run_bash interrupt"}), config)
                break

        # Print the last AI response (not the tool message)
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