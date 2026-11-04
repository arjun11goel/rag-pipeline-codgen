"""Split the source text into overlapping, word-based chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SOURCE_DOCUMENT_PATH,
    CHUNKS_PATH,
)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    """Create chunks of at most ``chunk_size`` words with a shared overlap.

    Word-based windows make the size and overlap easy to inspect and reproduce.
    The source wording and punctuation are preserved within each window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[dict] = []

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        end = start + len(window)
        chunks.append(
            {
                "id": f"chunk_{len(chunks) + 1:04d}",
                "text": " ".join(window),
                "word_count": len(window),
                "start_word": start,
                "end_word": end,
            }
        )
        if end >= len(words):
            break

    return chunks


def write_chunks(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create overlapping document chunks.")
    parser.add_argument("--input", type=Path, default=SOURCE_DOCUMENT_PATH)
    parser.add_argument("--output", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Maximum words per chunk (default: 250).")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Words shared between consecutive chunks (default: 50).")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Source document not found: {args.input}")

    source_text = args.input.read_text(encoding="utf-8")
    word_count = len(source_text.split())
    chunks = chunk_text(source_text, args.chunk_size, args.overlap)
    if not chunks:
        raise ValueError("The source document is empty; nothing to chunk.")

    write_chunks(chunks, args.output)
    counts = [item["word_count"] for item in chunks]
    print("Document chunking complete")
    print(f"Source: {args.input}")
    print(f"Source words: {word_count:,}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size} words | Overlap: {args.overlap} words")
    print(f"Chunk word-count range: {min(counts)}–{max(counts)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
