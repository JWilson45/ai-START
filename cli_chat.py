import subprocess
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

# --- Tool: run_bash (human approval required) ---
@tool
def run_bash(command: str) -> str:
    """Generate a bash command. The human must approve or deny via interrupt."""
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
    ("system", "You talk like a pirate. Keep replies concise."),
    MessagesPlaceholder("messages"),
])

# --- Graph nodes ---
def call_model_node(state: MessagesState):
    response = llm_with_tools.invoke(prompt.invoke(state))
    msg = response if isinstance(response, AIMessage) else AIMessage(
        content=getattr(response, "content", str(response))
    )
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
                else:
                    reason = input("Why deny? ").strip()
                    out = app.invoke(Command(resume={"approve": False, "reason": reason}), config)
            else:
                # Fallback: unknown interrupt type
                out = app.invoke(Command(resume={"approve": False, "reason": "Unhandled interrupt"}), config)

        last = out["messages"][-1]
        print("AI:", getattr(last, "content", ""), "\n")

if __name__ == "__main__":
    main()