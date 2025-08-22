import asyncio

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import Command, interrupt
from IPython.display import Image, display

# initialize LLM
llm = OllamaLLM(model="llama3.2:3b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Build a very small graph that accepts a `messages` list and calls the LLM.
# We use the built-in MessagesState schema which is a standard chat message list.
builder = StateGraph(MessagesState)

# Node that calls the model. The model is expected to accept the messages state
# and return a list or single message. We normalize and return the updated messages.
async def call_model(state: MessagesState):
    # state["messages"] is a list of message dicts: {"role": "user"/"assistant", "content": "..."}
    # Call the LLM with the messages; many chat-model adapters accept a list of messages.
    prompt = prompt_template.invoke(state)
    print(prompt)
    response = llm.invoke(prompt)  # could be a message or a list of messages

    # Normalize response into a list of messages
    if isinstance(response, list):
        new_messages = state["messages"] + response
    else:
        # if response is a string-like object, wrap it in an assistant message dict
        try:
            text = response.content
        except Exception:
            text = str(response)
        new_messages = state["messages"] + [{"role": "assistant", "content": text}]

    return {"messages": new_messages}

# add the node and wire it up
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Use MemorySaver for short-term thread-persisted memory. This keeps message history
# per `thread_id` and lets you resume the conversation later.
checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

async def main():
    print("Welcome — LangGraph-powered CLI. Type 'exit' to quit.")
    session_id = "default"

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        # pass the user message as the initial `messages` state; thread persistence will append and persist
        user_msg = [HumanMessage(content=user_input)]

        # invoke the compiled graph; provide a thread_id so LangGraph persists state for that thread
        config = {"configurable": {"thread_id": session_id}}
        out = await app.ainvoke({"messages": user_msg}, config)

        # The compiled graph returns the full messages list in out['messages']
        msgs = out.get("messages", [])
        if not msgs:
            print("AI: (no response returned)\n")
            continue

        last = msgs[-1]
        text = last.get("content") if isinstance(last, dict) else str(last)

        print("AI:", text, "\n")

        # Debugging: show how many messages are stored in-thread
        try:
            state_snapshot = app.get_state(config)
            # state_snapshot.values is the latest state; messages length is a useful quick check
            messages_len = len(state_snapshot.values.get("messages", []))
            print(f"🔹 Stored messages for thread '{session_id}': {messages_len}\n")
        except Exception:
            # get_state might not be available for some checkpointer backends in older versions
            pass

if __name__ == "__main__":
    asyncio.run(main())