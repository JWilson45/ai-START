# server.py
import json
import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Command

# Your graph
from src.graph import build_app

app = FastAPI(title="OpenAI-Compatible LangGraph API")
# Enable CORS so Open WebUI (running in browser) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to specific origins if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = build_app()

# --- Simple per-thread state for approvals ---
PENDING_APPROVAL: Dict[str, Dict[str, Any]] = {}  # thread_id -> payload awaiting approval

# --- Helpers ---

def to_lc(msg: Dict[str, Any]) -> BaseMessage:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "tool":
        # For completeness; most UIs won't send explicit tool msgs
        name = (msg.get("name") or msg.get("tool_name") or "tool")
        return ToolMessage(content=content, name=name, tool_call_id=msg.get("tool_call_id"))
    else:
        # fallback to user
        return HumanMessage(content=content)

def run_graph(messages: List[Dict[str, Any]], thread_id: str) -> Dict[str, Any]:
    """
    Invoke the LangGraph app with the **full chat history** so memory is owned by the agent.
    The agent (via checkpointers / MemorySaver / custom hooks) can decide how to trim/summarize.
    """
    msgs_lc: List[BaseMessage] = [to_lc(m) for m in messages]

    cfg = {"configurable": {"thread_id": thread_id}}
    out = graph.invoke({"messages": msgs_lc}, cfg)

    # Interrupt handling: if a run_bash approval is needed, store it and instruct the user
    if "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        if isinstance(payload, dict) and payload.get("tool") == "run_bash":
            PENDING_APPROVAL[thread_id] = payload
            # Produce an assistant message that tells the UI/user what to do
            notice = (
                "⚠️ **Approval required** for tool `run_bash`.\n\n"
                f"Command:\n\n```\n{payload.get('command','')}\n```\n\n"
                "Reply with `/approve` to run it, or `/deny <reason>` to cancel."
            )
            msgs = out.get("messages", [])[:]
            msgs.append(AIMessage(content=notice))
            return {"messages": msgs}

        # Non-run_bash interrupts: resume automatically as 'denied'
        out = graph.invoke(Command(resume={"approve": False, "reason": "UI auto-deny (no handler)"}), cfg)

    return {"messages": out.get("messages", [])}

def extract_text_from_messages(msgs: List[BaseMessage]) -> str:
    """
    Return the last AI message's human-readable text, and (if present) append
    a Markdown appendix that surfaces metadata like title/tags/follow-ups.
    This keeps Open WebUI compatible (shows plain text), while still using
    all the extra info your agent produces.
    """
    last_ai = None
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    if last_ai is None:
        return " "

    # Gather primary text
    text_parts: List[str] = []
    meta = {"title": None, "tags": [], "follow_ups": []}

    if isinstance(last_ai.content, str):
        text_parts.append(last_ai.content)
    elif isinstance(last_ai.content, list):
        for chunk in last_ai.content:
            if isinstance(chunk, dict):
                ctype = chunk.get("type")
                if ctype == "text":
                    t = chunk.get("text")
                    if t:
                        text_parts.append(str(t))
                elif ctype == "json":
                    data = chunk.get("data") or {}
                    # Accept both snake_case and camelCase just in case
                    title = data.get("title") or data.get("Title")
                    tags = data.get("tags") or data.get("Tags")
                    follow_ups = data.get("follow_ups") or data.get("followUps") or data.get("FollowUps")
                    if title and not meta["title"]:
                        meta["title"] = str(title)
                    if isinstance(tags, list):
                        meta["tags"].extend([str(x) for x in tags if x is not None])
                    if isinstance(follow_ups, list):
                        meta["follow_ups"].extend([str(x) for x in follow_ups if x is not None])
    else:
        # Fallback: unknown content type
        text_parts.append(str(getattr(last_ai, "content", "")))

    primary = "\n".join([p for p in (tp.strip() for tp in text_parts) if p]) or " "

    # If no metadata, just return the text
    if not (meta["title"] or meta["tags"] or meta["follow_ups"]):
        return primary

    # Build a friendly appendix for Open WebUI to display as plain text
    lines = [primary, "\n---"]
    if meta["title"]:
        lines.append(f"**Title suggestion:** {meta['title']}")
    if meta["tags"]:
        lines.append("**Tags:** " + ", ".join(sorted(set(meta["tags"]))))
    if meta["follow_ups"]:
        lines.append("**Suggested follow-ups:**")
        for fu in meta["follow_ups"]:
            lines.append(f"- {fu}")
    return "\n".join(lines).strip() or " "

# --- OpenAI-compatible schemas (minimal) ---

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    # Optional: temperature, top_p, etc. – ignored but accepted for compat
    metadata: Optional[Dict[str, Any]] = None

# --- Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "pirate-graph", "object": "model", "owned_by": "local"}]}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest):
    # Thread selection
    thread_id = (req.metadata or {}).get("thread_id") or "default"

    # Intercept simple approval commands from the user:
    last = req.messages[-1].content.strip() if req.messages else ""
    if last.startswith("/approve") and thread_id in PENDING_APPROVAL:
        payload = PENDING_APPROVAL.pop(thread_id)
        cfg = {"configurable": {"thread_id": thread_id}}
        out = graph.invoke(Command(resume={"approve": True}), cfg)
        result_text = extract_text_from_messages(out.get("messages", []))
        return _as_nonstream_response(req.model, result_text)

    if last.startswith("/deny") and thread_id in PENDING_APPROVAL:
        reason = last[len("/deny"):].strip() or "No reason provided"
        payload = PENDING_APPROVAL.pop(thread_id)
        cfg = {"configurable": {"thread_id": thread_id}}
        out = graph.invoke(Command(resume={"approve": False, "reason": reason}), cfg)
        result_text = extract_text_from_messages(out.get("messages", []))
        return _as_nonstream_response(req.model, result_text)

    # Normal invoke
    res = run_graph([m.model_dump() for m in req.messages], thread_id)
    text = extract_text_from_messages(res.get("messages", [])) or " "

    if req.stream:
        return _as_streaming_response(req.model, text)
    else:
        return _as_nonstream_response(req.model, text)

def _as_nonstream_response(model: str, text: str):
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    now = int(time.time())
    return JSONResponse({
        "id": rid,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })

def _as_streaming_response(model: str, text: str):
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    now = int(time.time())

    async def gen():
        # initial role delta
        first = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        }
        yield f"data: {json.dumps(first)}\n\n"
        # chunk the content
        for i in range(0, len(text), 400):
            chunk = text[i:i+400]
            yield "data: " + json.dumps({
                "id": rid,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}}]
            }) + "\n\n"
            await asyncio.sleep(0)  # give the loop a tick
        # done
        yield "data: " + json.dumps({
            "id": rid,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")