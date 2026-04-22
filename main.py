"""
LangChain RAG API — FastAPI app for document ingestion and Q&A.

Uses feature-hashing for embeddings (pure numpy, no ML model download),
pgvector for vector storage, and Anthropic Claude for answer generation.

Endpoints:
  GET  /health  — liveness check
  POST /upload  — ingest a PDF or text file into the knowledge base
  POST /query   — retrieve relevant chunks and return a grounded answer
"""

import asyncio
import hashlib
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://varity:varity@localhost:5432/app")
COLLECTION = "rag_documents"
EMBED_DIM = 384

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


# ---------------------------------------------------------------------------
# Embedding — feature hashing (no model download, O(words) per document)
# ---------------------------------------------------------------------------

def _embed(texts: List[str]) -> List[List[float]]:
    """Return L2-normalised 384-dim feature-hash embeddings for each text."""
    results = []
    for text in texts:
        vec = np.zeros(EMBED_DIM)
        for word in text.lower().split():
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % EMBED_DIM
            sign = 1 if (h >> 8) & 1 else -1
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        results.append((vec / norm if norm > 0 else vec).tolist())
    return results


def _vec_str(v: List[float]) -> str:
    """Format a float list as a PostgreSQL vector literal '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _init_db(retries: int = 8, delay: float = 4.0):
    """Create the pgvector table, retrying while postgres starts up."""
    last_err = None
    for attempt in range(retries):
        try:
            conn = _get_conn()
            with conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {COLLECTION} (
                        id      SERIAL PRIMARY KEY,
                        content TEXT    NOT NULL,
                        source  TEXT,
                        chunk   INT,
                        embedding vector({EMBED_DIM})
                    )
                """)
            conn.close()
            return
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"Warning: DB init failed: {last_err}")


def _insert_chunks(filename: str, chunks: List[str], embeddings: List[List[float]]):
    conn = _get_conn()
    with conn, conn.cursor() as cur:
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                f"""INSERT INTO {COLLECTION} (content, source, chunk, embedding)
                    VALUES (%s, %s, %s, %s::vector)""",
                (chunk, filename, i, _vec_str(emb)),
            )
    conn.close()


def _search(query_emb: List[float], k: int = 4) -> List[dict]:
    vec = _vec_str(query_emb)
    conn = _get_conn()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            f"""SELECT content, source, chunk,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {COLLECTION}
                ORDER BY embedding <=> %s::vector
                LIMIT %s""",
            (vec, vec, k),
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _init_db)
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
    raw = await file.read()
    filename = file.filename or "document.txt"
    suffix = os.path.splitext(filename)[1].lower() or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(tmp_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = raw.decode("utf-8", errors="replace")

        chunks = _chunk(text)
        if not chunks:
            return {"status": "ok", "chunks_stored": 0, "filename": filename}

        embeddings = _embed(chunks)
        _insert_chunks(filename, chunks, embeddings)

        return {"status": "ok", "chunks_stored": len(chunks), "filename": filename}
    finally:
        os.unlink(tmp_path)


@app.post("/query")
def query_documents(req: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    [query_emb] = _embed([req.question])
    results = _search(query_emb, k=req.k)

    if not results:
        return {"answer": "No relevant documents found in the knowledge base.", "sources": []}

    context = "\n\n---\n\n".join(r["content"] for r in results)

    if ANTHROPIC_API_KEY:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
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
        answer = msg.content[0].text
    else:
        answer = results[0]["content"]

    return {
        "answer": answer,
        "sources": [
            {
                "content": r["content"][:200],
                "source": r.get("source"),
                "similarity": float(r["similarity"]),
            }
            for r in results
        ],
    }
