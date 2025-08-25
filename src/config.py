"""Configuration for the AI chat application."""

# Vector store (Chroma) + RAG helpers
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_COLLECTION = "kb_main"

# Model + tools (native tool calling)
MAX_MODEL_TOKENS = 3000        # ~60–70% of context window
MAX_STORED_MESSAGES = 40       # keep recent conversational detail compact
SUMMARY_CHUNK_SIZE = 20        # how many early messages to compress at a time
TOOL_OUTPUT_CHAR_LIMIT = 4000  # clamp tool outputs stored in history
