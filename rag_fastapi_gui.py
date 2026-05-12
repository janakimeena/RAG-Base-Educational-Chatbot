#!/usr/bin/env python3
"""
FastAPI + simple browser GUI for the FAISS + Gemini RAG chatbot.

Run:
  export GEMINI_API_KEY="YOUR_KEY"
  uvicorn rag_fastapi_gui:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from chat_faiss import (
    build_context,
    build_prompt,
    generate_answer_with_gemini,
    load_chunks,
    load_json,
    retrieve,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve")
    chunk_char_limit: int = Field(
        default=1000,
        ge=200,
        le=4000,
        description="Max chars from each retrieved chunk for context",
    )
    gemini_model: str = Field(default="gemini-2.5-flash")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]


class AppState:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self.embedding_model: SentenceTransformer | None = None
        self.normalize_embeddings: bool = True
        self.api_key: str | None = None


state = AppState()
app = FastAPI(title="PyTorch Docs RAG Chatbot", version="1.0.0")

# Guardrail defaults (tunable via env vars).
MAX_QUERY_CHARS = int(os.getenv("RAG_MAX_QUERY_CHARS", "4000"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RAG_RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RAG_RATE_LIMIT_MAX_REQUESTS", "20"))
ALLOWED_SOURCE_PREFIXES = ("https://pytorch.org/docs/",)

_rate_limit_buckets: dict[str, deque[float]] = defaultdict(deque)

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"(reveal|show|print).*(system\s+prompt|hidden\s+prompt)", re.IGNORECASE),
    re.compile(r"(api[_\s-]?key|password|secret|token).*(reveal|show|print|dump)", re.IGNORECASE),
]

_UNSAFE_REQUEST_PATTERNS = [
    re.compile(r"\b(build|create|write)\b.*\b(malware|ransomware|keylogger|exploit)\b", re.IGNORECASE),
    re.compile(r"\b(ddos|phishing|credential\s+stuffing)\b", re.IGNORECASE),
]

_SENSITIVE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
]


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _get_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        # first IP in forwarded chain is the client
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    client_id = _get_client_id(request)
    now = time.time()
    bucket = _rate_limit_buckets[client_id]

    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SEC} seconds."
            ),
        )

    bucket.append(now)


def validate_user_query(query: str) -> None:
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Maximum allowed length is {MAX_QUERY_CHARS} characters.",
        )

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            raise HTTPException(
                status_code=400,
                detail="Query blocked due to prompt-injection-like content.",
            )

    for pattern in _UNSAFE_REQUEST_PATTERNS:
        if pattern.search(query):
            raise HTTPException(
                status_code=400,
                detail="Query blocked due to unsafe content policy.",
            )


def filter_trusted_sources(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trusted: list[dict[str, Any]] = []
    for item in retrieved:
        url = str(item.get("source_url", ""))
        if url.startswith(ALLOWED_SOURCE_PREFIXES):
            trusted.append(item)
    return trusted


def load_runtime_assets() -> None:
    index_path = Path(os.getenv("RAG_INDEX_PATH", "pytorch_docs.faiss"))
    chunks_path = Path(os.getenv("RAG_CHUNKS_PATH", "pytorch_docs_chunks.jsonl"))
    meta_path = Path(os.getenv("RAG_META_PATH", "pytorch_docs_index_meta.json"))
    embedding_model_override = os.getenv("RAG_EMBEDDING_MODEL")

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk metadata file not found: {chunks_path}")

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = load_json(meta_path)
    elif not embedding_model_override:
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}. "
            "Set RAG_EMBEDDING_MODEL env var to continue."
        )

    embedding_model_name = embedding_model_override or meta.get("embedding_model")
    if not embedding_model_name:
        raise ValueError("Could not resolve embedding model name from metadata or env var.")

    state.index = faiss.read_index(str(index_path))
    state.chunks = load_chunks(chunks_path)
    if not state.chunks:
        raise RuntimeError("Chunks file loaded but contains no valid chunks.")

    state.normalize_embeddings = bool(meta.get("normalize_embeddings", True))
    state.embedding_model = SentenceTransformer(embedding_model_name)
    state.api_key = os.getenv("GEMINI_API_KEY")
    if not state.api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please export it before running the server.")


@app.on_event("startup")
def startup_event() -> None:
    load_runtime_assets()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_vectors": state.index.ntotal if state.index else 0,
        "chunks_loaded": len(state.chunks),
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest, request: Request) -> QueryResponse:
    if not state.index or not state.embedding_model or not state.api_key:
        raise HTTPException(status_code=500, detail="Server is not initialized correctly.")

    try:
        enforce_rate_limit(request)
        validate_user_query(payload.query)

        retrieved = retrieve(
            payload.query,
            index=state.index,
            chunks=state.chunks,
            embedding_model=state.embedding_model,
            normalize_embeddings=state.normalize_embeddings,
            top_k=payload.top_k,
        )
        retrieved = filter_trusted_sources(retrieved)
        if not retrieved:
            raise HTTPException(status_code=404, detail="No documents retrieved for this query.")

        context = build_context(retrieved, per_chunk_char_limit=payload.chunk_char_limit)
        context = context[:MAX_CONTEXT_CHARS]
        prompt = build_prompt(payload.query, context)
        answer = generate_answer_with_gemini(
            api_key=state.api_key,
            model=payload.gemini_model,
            prompt=prompt,
            timeout=60,
        )
        answer = redact_sensitive_text(answer)
        sources = [
            {
                "score": item["score"],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "chunk_id": item["chunk_id"],
            }
            for item in retrieved
        ]
        return QueryResponse(query=payload.query, answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        safe_error = redact_sensitive_text(str(exc))
        raise HTTPException(status_code=500, detail=safe_error) from exc


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>PyTorch Docs RAG Chatbot</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        background: #f7f7f8;
      }
      .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 14px;
      }
      h1 {
        margin: 0 0 6px 0;
      }
      .muted {
        color: #666;
        margin-bottom: 8px;
      }
      .chat-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        min-height: 170px;
        max-height: 36vh;
        overflow-y: auto;
        padding: 10px;
      }
      .msg {
        margin-bottom: 6px;
        padding: 10px 12px;
        border-radius: 10px;
        line-height: 1.4;
        white-space: pre-wrap;
      }
      .user {
        background: #e8f0fe;
        margin-left: 10%;
      }
      .bot {
        background: #f1f3f4;
        margin-right: 10%;
      }
      .sources {
        margin-top: 8px;
        font-size: 13px;
      }
      .sources a {
        color: #1a73e8;
        text-decoration: none;
      }
      .composer-wrap {
        margin-top: 8px;
      }
      .input-label {
        display: block;
        margin-bottom: 6px;
        font-weight: 600;
      }
      .composer {
        display: flex;
        align-items: flex-end;
        gap: 10px;
      }
      .composer textarea {
        flex: 1;
        font-size: 15px;
        padding: 11px 12px;
        border: 1px solid #ccc;
        border-radius: 8px;
        min-height: 220px;
        max-height: 40vh;
        resize: vertical;
      }
      .composer button {
        font-size: 15px;
        padding: 10px 16px;
        border: 1px solid #0b57d0;
        border-radius: 8px;
        background: #1a73e8;
        color: white;
        cursor: pointer;
      }
      .composer button:disabled {
        opacity: 0.7;
        cursor: default;
      }
      .error {
        color: #b00020;
        white-space: pre-wrap;
        margin-top: 8px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>PyTorch Docs RAG Chatbot</h1>
      <p class="muted">Ask a question about PyTorch docs.</p>

      <div id="chatBox" class="chat-box">
        <div class="msg bot">Hello! Ask me anything about PyTorch documentation.</div>
      </div>

      <div class="composer-wrap">
        <label class="input-label" for="queryInput">Type your question</label>
        <div class="composer">
          <textarea id="queryInput" rows="12" placeholder="Type your question..."></textarea>
          <button id="askBtn">Ask</button>
        </div>
      </div>
      <div id="error" class="error"></div>
    </div>

    <script>
      const askBtn = document.getElementById("askBtn");
      const queryInputEl = document.getElementById("queryInput");
      const chatBoxEl = document.getElementById("chatBox");
      const errorEl = document.getElementById("error");

      function escapeHtml(text) {
        return text.replace(/[&<>"']/g, function(m) {
          return ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" })[m];
        });
      }

      function appendMessage(kind, html) {
        const el = document.createElement("div");
        el.className = "msg " + kind;
        el.innerHTML = html;
        chatBoxEl.appendChild(el);
        chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
      }

      async function submitQuestion() {
        const query = queryInputEl.value.trim();
        if (!query) {
          errorEl.textContent = "Please enter a question.";
          return;
        }

        errorEl.textContent = "";
        appendMessage("user", escapeHtml(query));
        queryInputEl.value = "";
        askBtn.disabled = true;
        askBtn.textContent = "Asking...";

        try {
          const resp = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query
            })
          });

          const data = await resp.json();
          if (!resp.ok) {
            throw new Error(data.detail || "Query failed");
          }

          const sourcesHtml = data.sources.map((s, i) => `
            <div>
              [${i + 1}] <a href="${s.source_url}" target="_blank">${escapeHtml(s.source_title || s.source_url)}</a>
            </div>
          `).join("");

          const answerHtml = `
            <div>${escapeHtml(data.answer)}</div>
            <div class="sources"><strong>Sources:</strong>${sourcesHtml}</div>
          `;
          appendMessage("bot", answerHtml);
        } catch (err) {
          errorEl.textContent = String(err);
        } finally {
          askBtn.disabled = false;
          askBtn.textContent = "Ask";
        }
      }

      askBtn.addEventListener("click", submitQuestion);
      queryInputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          submitQuestion();
        }
      });
    </script>
  </body>
</html>
"""
