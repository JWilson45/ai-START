"""Tool definitions for the AI chat application."""
import subprocess
import json
import os
import logging
from langchain_core.tools import tool
from langgraph.types import interrupt
from .vector_store import _vectorstore, _expand_paths, _load_documents, _chunk_documents, _ids_for_docs, _ensure_persist_dir
from .config import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

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

@tool
def ingest_documents(path: str, collection: str = DEFAULT_COLLECTION) -> str:
    """
    Ingest local files into the persistent Chroma vector store.

    Args:
    - path: A file path, directory, or glob (supports .pdf, .txt, .md)
    - collection: Optional Chroma collection name (default: "kb")

    Behavior:
    - Loads PDFs via PyPDFLoader and text/Markdown via TextLoader
    - Splits into ~1000-char chunks with 150 overlap
    - Embeds with OpenAI `text-embedding-3-small` and upserts into Chroma
    - Deduplicates by content hash-based IDs
    """
    logger.info("Ingest starting: path=%s collection=%s", path, collection)
    files = _expand_paths(path)
    logger.info("Expand paths -> %d files: %s", len(files), files)
    if not files:
        return json.dumps({
            "status": "no_files_found",
            "path": path,
        })

    raw_docs = _load_documents(files)
    logger.info("Loaded %d docs", len(raw_docs))
    if not raw_docs:
        return json.dumps({
            "status": "load_failed",
            "path": path,
            "files": files,
        })

    chunks = _chunk_documents(raw_docs)
    logger.info("Chunked into %d chunks", len(chunks))
    if not chunks:
        return json.dumps({
            "status": "no_chunks",
            "path": path,
        })

    try:
        vs = _vectorstore(collection)
        ids = _ids_for_docs(chunks)
        logger.info("Generated %d ids", len(ids))
        vs.add_documents(documents=chunks, ids=ids)
        logger.info("Upserted %d chunks to Chroma collection=%s", len(chunks), collection)
    except Exception as e:
        # Common causes: missing OPENAI_API_KEY, network error, incompatible chroma/langchain versions
        api_key = os.getenv("OPENAI_API_KEY", "")
        masked = (api_key[:6] + "…") if api_key else "(missing)"
        logger.exception("Ingest failed during vectorstore/add_documents: %s", e)
        return json.dumps({
            "status": "error",
            "stage": "vectorstore_or_upsert",
            "reason": str(e),
            "openai_api_key": masked,
            "collection": collection,
        })
    # Chroma persists automatically when persist_directory is set

    return json.dumps({
        "status": "ingested",
        "path": path,
        "files_count": len(files),
        "chunks_upserted": len(chunks),
        "collection": collection,
        "persist_dir": _ensure_persist_dir("chroma"),
    })


@tool
def search_corpus(query: str, k: int = 5, collection: str = DEFAULT_COLLECTION) -> str:
    """
    Retrieve top-k chunks from the Chroma vector store and return structured results.

    Args:
    - query: The search query
    - k: number of results (default 5)
    - collection: Chroma collection name (default "kb")

    Returns JSON with fields: query, k, results=[{score, snippet, source, page}]
    """
    if not query or not query.strip():
        return json.dumps({"status": "error", "reason": "empty_query"})

    vs = _vectorstore(collection)
    try:
        docs_scores = vs.similarity_search_with_score(query, k=k)
        logger.info("Search: query='%s' k=%d -> %d hits", query, k, len(docs_scores))
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return json.dumps({"status": "error", "reason": str(e)})

    results = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        results.append({
            "score": float(score),
            "snippet": doc.page_content[:500],
            "source": meta.get("source"),
            "page": meta.get("page"),
        })

    return json.dumps({
        "status": "ok",
        "query": query,
        "k": k,
        "collection": collection,
        "results": results,
    })
