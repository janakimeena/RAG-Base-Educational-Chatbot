#!/usr/bin/env python3
"""
Query a FAISS index built from PyTorch docs chunks.

Example:
  python query_faiss.py \
      --query "How does nn.Conv2d work?" \
      --index pytorch_docs.faiss \
      --chunks pytorch_docs_chunks.jsonl \
      --meta pytorch_docs_index_meta.json \
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                chunks.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping bad JSON line {line_number}: {exc}")
    return chunks


def embed_query(model: SentenceTransformer, query: str, normalize_embeddings: bool) -> np.ndarray:
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local FAISS index.")
    parser.add_argument("--query", required=True, help="User question/query text.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("pytorch_docs.faiss"),
        help="Path to FAISS index file.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("pytorch_docs_chunks.jsonl"),
        help="Path to chunks JSONL file.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("pytorch_docs_index_meta.json"),
        help="Path to metadata JSON from indexing step.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model; defaults to model in --meta.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return.",
    )
    parser.add_argument(
        "--show-text-chars",
        type=int,
        default=350,
        help="Max number of characters to print per chunk.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.index.exists():
        raise FileNotFoundError(f"FAISS index not found: {args.index}")
    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks}")

    meta: dict[str, Any] = {}
    if args.meta.exists():
        meta = load_json(args.meta)
    elif args.embedding_model is None:
        raise FileNotFoundError(
            f"Meta file not found: {args.meta}. "
            "Pass --embedding-model explicitly when no meta file is available."
        )

    embedding_model = args.embedding_model or meta.get("embedding_model")
    if not embedding_model:
        raise ValueError("Unable to determine embedding model. Pass --embedding-model.")

    normalize_embeddings = bool(meta.get("normalize_embeddings", True))
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    print(f"Loading index: {args.index}")
    index = faiss.read_index(str(args.index))
    print(f"Index vectors: {index.ntotal}")

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise RuntimeError("Chunks file is empty or invalid.")

    query_vec = embed_query(model, args.query, normalize_embeddings=normalize_embeddings)
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    top_k = min(args.top_k, index.ntotal)
    scores, ids = index.search(query_vec, top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", ""),
                "source_title": chunk.get("source_title", ""),
                "source_url": chunk.get("source_url", ""),
                "chunk_index": chunk.get("chunk_index", -1),
                "text": chunk.get("text", ""),
            }
        )

    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
        return

    print("\nQuery:", args.query)
    print("-" * 80)
    for rank, item in enumerate(results, start=1):
        snippet = item["text"][: args.show_text_chars].strip()
        print(f"[{rank}] score={item['score']:.4f} chunk_id={item['chunk_id']}")
        print(f"    title: {item['source_title']}")
        print(f"    url:   {item['source_url']}")
        print(f"    text:  {snippet}")
        print("-" * 80)


if __name__ == "__main__":
    main()
