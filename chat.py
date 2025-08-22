from langchain_ollama import OllamaLLM

# Point to your pulled model
llm = OllamaLLM(model="llama3.2:3b")

response = llm.invoke("Hello! Summarize AI in one sentence.")

print(response)