"""Vector store (Chroma) + RAG helpers for the AI chat application."""
import os
import glob
import re
import logging
import hashlib
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

COLL_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]{1,510})[a-zA-Z0-9]$")

def _sanitize_collection_name(name: str) -> str:
    """Coerce a user-provided collection name to Chroma's constraints.
    Rules: 3-512 chars, [a-zA-Z0-9._-], start/end alnum.
    """
    if not isinstance(name, str):
        name = str(name or "kb_main")
    # Replace disallowed chars with '-'
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    # Ensure starts/ends with alnum
    if not name or not name[0].isalnum():
        name = f"k{name}"
    if not name[-1].isalnum():
        name = f"{name}0"
    # Enforce min length 3
    while len(name) < 3:
        name += "0"
    # Enforce max length 512
    name = name[:512]
    # Final check; if still invalid, fall back
    if not COLL_RE.match(name):
        return "kb_main"
    return name


def _ensure_persist_dir(subdir: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "..", ".state", subdir)
    os.makedirs(base, exist_ok=True)
    return base


def _vectorstore(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """
    Return a persistent Chroma vector store located under ./.state/chroma
    configured with OpenAI embeddings (text-embedding-3-small).
    """
    original = collection_name
    collection_name = _sanitize_collection_name(collection_name)
    if original != collection_name:
        logger.info("Sanitized collection name: '%s' -> '%s'", original, collection_name)
    persist_directory = _ensure_persist_dir("chroma")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        logger.info("Chroma vectorstore ready: collection=%s dir=%s", collection_name, persist_directory)
        return vs
    except Exception as e:
        logger.exception("Vectorstore init failed: %s", e)
        raise


def _expand_paths(path_or_glob: str) -> list[str]:
    # Accept file, directory, or glob. Return concrete file list.
    p = os.path.expanduser(path_or_glob)
    if os.path.isdir(p):
        # Recurse common doc types inside a directory
        files = []
        for ext in ("**/*.pdf", "**/*.txt", "**/*.md"):
            files.extend(glob.glob(os.path.join(p, ext), recursive=True))
        return sorted(files)
    # Glob or single file
    matches = glob.glob(p, recursive=True)
    return sorted([m for m in matches if os.path.isfile(m)])


def _load_documents(paths: list[str]):
    docs = []
    for fp in paths:
        try:
            if fp.lower().endswith(".pdf"):
                loader = PyPDFLoader(fp)
                docs.extend(loader.load())
            elif fp.lower().endswith((".txt", ".md")):
                loader = TextLoader(fp, encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            logger.warning("Skipping %s (%s)", fp, e)
    return docs


def _chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _ids_for_docs(docs) -> list[str]:
    ids = []
    for d in docs:
        src = str(d.metadata.get("source", ""))
        page = str(d.metadata.get("page", ""))
        h = hashlib.sha1((d.page_content + "|" + src + "|" + page).encode("utf-8", errors="ignore")).hexdigest()
        ids.append(h)
    return ids
