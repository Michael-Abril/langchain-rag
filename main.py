"""
LangChain RAG API — FastAPI app that indexes documents and answers questions
using vector similarity retrieval + LLM generation.

Endpoints:
  GET  /health          — liveness check
  POST /upload          — add a PDF or text file to the knowledge base
  POST /query           — ask a question, get a RAG-powered answer
"""

import os
import re
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_RAW_DB_URL = os.getenv("DATABASE_URL", "postgresql://varity:varity@localhost:5432/app")

def _sqlalchemy_url(url: str) -> str:
    """Ensure the URL uses the postgresql+psycopg2:// scheme for SQLAlchemy."""
    return re.sub(r"^postgresql://", "postgresql+psycopg2://", url)

DB_URL = _sqlalchemy_url(_RAW_DB_URL)
COLLECTION_NAME = "rag_documents"

# LLM config — uses Anthropic Claude (configurable via env)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# Embedding model (fastembed — no torch required, ~60 MB download on first use)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_embeddings = None
_vector_store = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_vector_store(retries: int = 5, delay: float = 3.0):
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    from langchain_community.vectorstores import PGVector

    last_err = None
    for attempt in range(retries):
        try:
            vs = PGVector(
                connection_string=DB_URL,
                embedding_function=_get_embeddings(),
                collection_name=COLLECTION_NAME,
                pre_delete_collection=False,
            )
            _vector_store = vs
            return vs
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(delay)

    raise RuntimeError(f"Could not connect to database after {retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _get_vector_store()
        print("Vector store initialised")
    except Exception as exc:
        print(f"Warning: vector store not ready at startup — {exc}")
    yield


app = FastAPI(title="LangChain RAG API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "langchain-rag"}


class QueryRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a PDF or plain-text file into the vector knowledge base."""
    content = await file.read()
    filename = file.filename or "document.txt"
    suffix = os.path.splitext(filename)[1].lower() or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vs = _get_vector_store()
        vs.add_documents(chunks)

        return {"status": "ok", "chunks_stored": len(chunks), "filename": filename}
    finally:
        os.unlink(tmp_path)


@app.post("/query")
def query_documents(req: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    vs = _get_vector_store()
    docs = vs.similarity_search(req.question, k=req.k)

    if not docs:
        return {"answer": "No relevant documents found in the knowledge base.", "sources": []}

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=1024,
        )
        prompt = (
            "Use only the context below to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer:"
        )
        response = llm.invoke(prompt)
        answer = response.content
    else:
        # Extractive fallback when no LLM is configured
        answer = docs[0].page_content

    return {
        "answer": answer,
        "sources": [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in docs
        ],
    }
