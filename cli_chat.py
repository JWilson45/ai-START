import asyncio
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize LLM
llm = OllamaLLM(model="llama3.2:3b")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You talk like a pirate. Answer all questions like one."),
    MessagesPlaceholder("messages"),
])

# Graph
builder = StateGraph(MessagesState)

async def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    # Normalize
    text = getattr(response, "content", str(response))
    return {"messages": state["messages"] + [{"role": "assistant", "content": text}]}

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

app = builder.compile(checkpointer=InMemorySaver())

async def main():
    print("Welcome — pirate CLI. Type 'exit' to quit.")
    session_id = "default"

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        config = {"configurable": {"thread_id": session_id}}
        out = await app.ainvoke({"messages": [HumanMessage(content=user_input)]}, config)

        print(out)

        last = out["messages"][-1]
        print("AI:", last["content"], "\n")

if __name__ == "__main__":
    asyncio.run(main())