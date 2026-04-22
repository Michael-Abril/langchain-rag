"""
LangChain RAG API — FastAPI app that indexes documents and answers questions
using vector similarity retrieval + LLM generation.

Endpoints:
  GET  /health          — liveness check
  POST /upload          — add a PDF or text file to the knowledge base
  POST /query           — ask a question, get a RAG-powered answer
"""

import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


def _vec_str(v: List[float]) -> str:
    """Format a float list as a PostgreSQL vector literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://varity:varity@localhost:5432/app")
COLLECTION_NAME = "rag_documents"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _init_db(retries: int = 8, delay: float = 4.0):
    """Create the embeddings table (with retry for container startup lag)."""
    last_err = None
    for attempt in range(retries):
        try:
            conn = _get_conn()
            with conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {COLLECTION_NAME} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding vector(384)
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {COLLECTION_NAME}_embedding_idx
                    ON {COLLECTION_NAME} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            conn.close()
            return
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"Warning: DB init failed after {retries} attempts: {last_err}")


def _insert_chunks(chunks: List[dict]):
    """Insert text chunks with embeddings into the table."""
    conn = _get_conn()
    with conn, conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                f"INSERT INTO {COLLECTION_NAME} (content, metadata, embedding) VALUES (%s, %s, %s::vector)",
                (chunk["content"], psycopg2.extras.Json(chunk["metadata"]), _vec_str(chunk["embedding"])),
            )
    conn.close()


def _search(query_embedding: List[float], k: int = 4) -> List[dict]:
    """Return the k most similar chunks by cosine distance."""
    vec = _vec_str(query_embedding)
    conn = _get_conn()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            f"""
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {COLLECTION_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, k),
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Embeddings (lazy singleton)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)
    return _embed_model


def _embed(texts: List[str]) -> List[List[float]]:
    model = _get_embed_model()
    return [emb.tolist() for emb in model.embed(texts)]


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple sliding-window chunker — no langchain dependency needed."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
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
        if suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(tmp_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = content.decode("utf-8", errors="replace")

        raw_chunks = _chunk_text(text)
        embeddings = _embed(raw_chunks)

        chunks = [
            {
                "content": raw_chunks[i],
                "metadata": {"source": filename, "chunk": i},
                "embedding": embeddings[i],
            }
            for i in range(len(raw_chunks))
        ]
        _insert_chunks(chunks)

        return {"status": "ok", "chunks_stored": len(chunks), "filename": filename}
    finally:
        os.unlink(tmp_path)


@app.post("/query")
def query_documents(req: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    [query_embedding] = _embed([req.question])
    results = _search(query_embedding, k=req.k)

    if not results:
        return {"answer": "No relevant documents found in the knowledge base.", "sources": []}

    context = "\n\n---\n\n".join(r["content"] for r in results)

    if ANTHROPIC_API_KEY:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    "Use only the context below to answer the question. "
                    "If the answer is not in the context, say so.\n\n"
                    f"Context:\n{context}\n\nQuestion: {req.question}"
                ),
            }],
        )
        answer = message.content[0].text
    else:
        answer = results[0]["content"]

    return {
        "answer": answer,
        "sources": [
            {"content": r["content"][:200], "metadata": r["metadata"], "similarity": r["similarity"]}
            for r in results
        ],
    }
