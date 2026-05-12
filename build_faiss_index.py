#!/usr/bin/env python3
"""
Build a FAISS vector index from scraped PyTorch docs JSONL.

Input format (one JSON object per line):
{
  "url": "...",
  "title": "...",
  "text": "..."
}

Example:
  python build_faiss_index.py \
      --input pytorch_nn_docs.jsonl \
      --index-out pytorch_docs.faiss \
      --chunks-out pytorch_docs_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SourceDocument:
    url: str
    title: str
    text: str


@dataclass
class TextChunk:
    chunk_id: str
    source_url: str
    source_title: str
    chunk_index: int
    text: str


def load_jsonl_documents(path: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping invalid JSON on line {line_number}: {exc}")
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            docs.append(
                SourceDocument(
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    text=text,
                )
            )
    return docs


def clean_text(text: str) -> str:
    # Remove repeated boilerplate snippets common in docs pages.
    boilerplate_patterns = [
        r"\bRate this Page\b",
        r"\bEdit on GitHub\b",
        r"\bShow Source\b",
        r"\bOn this page\b",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_words(words: list[str], chunk_size: int, chunk_overlap: int) -> Iterable[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = words[start:end]
        if chunk:
            yield chunk


def make_chunks(
    docs: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
    min_words: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for doc_idx, doc in enumerate(docs):
        cleaned = clean_text(doc.text)
        words = cleaned.split()
        if not words:
            continue

        chunk_idx = 0
        for piece in chunk_words(words, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            if len(piece) < min_words:
                continue
            text = " ".join(piece).strip()
            if not text:
                continue
            chunk_id = f"doc{doc_idx:05d}_chunk{chunk_idx:05d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source_url=doc.url,
                    source_title=doc.title,
                    chunk_index=chunk_idx,
                    text=text,
                )
            )
            chunk_idx += 1
    return chunks


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")

    dim = vectors.shape[1]
    # Use inner product. With normalized embeddings this equals cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_chunks(chunks: list[TextChunk], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def save_meta(meta_path: Path, *, args: argparse.Namespace, num_docs: int, num_chunks: int) -> None:
    payload = {
        "input_file": str(args.input),
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "min_words": args.min_words,
        "normalize_embeddings": args.normalize_embeddings,
        "num_docs": num_docs,
        "num_chunks": num_chunks,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split docs JSONL into chunks, embed, and store in FAISS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pytorch_nn_docs.jsonl"),
        help="Path to docs JSONL file.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=180,
        help="Chunk size in words.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Word overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=30,
        help="Drop chunks shorter than this many words.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=Path("pytorch_docs.faiss"),
        help="FAISS index output path.",
    )
    parser.add_argument(
        "--chunks-out",
        type=Path,
        default=Path("pytorch_docs_chunks.jsonl"),
        help="Chunk metadata output JSONL.",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("pytorch_docs_index_meta.json"),
        help="Index metadata output JSON.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        default=True,
        help="L2-normalize embeddings before indexing (recommended for cosine search).",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable embedding normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[1/4] Loading documents from {args.input}")
    docs = load_jsonl_documents(args.input)
    if not docs:
        raise RuntimeError("No valid documents found in input JSONL.")
    print(f"Loaded {len(docs)} documents")

    print("[2/4] Chunking documents")
    chunks = make_chunks(
        docs=docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_words=args.min_words,
    )
    if not chunks:
        raise RuntimeError("No chunks produced. Adjust chunk_size/chunk_overlap/min_words.")
    print(f"Created {len(chunks)} chunks")

    print(f"[3/4] Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(
        model=model,
        texts=texts,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )

    print("[4/4] Building and saving FAISS index")
    index = build_faiss_index(vectors)
    faiss.write_index(index, str(args.index_out))
    save_chunks(chunks, args.chunks_out)
    save_meta(args.meta_out, args=args, num_docs=len(docs), num_chunks=len(chunks))

    print(f"\nSaved FAISS index: {args.index_out}")
    print(f"Saved chunk file:  {args.chunks_out}")
    print(f"Saved metadata:    {args.meta_out}")


if __name__ == "__main__":
    main()
