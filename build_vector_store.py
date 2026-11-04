"""Embed document chunks and persist them in a local ChromaDB collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run `python chunk_document.py` first."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The chunk file must contain a non-empty JSON list.")

    seen_ids = set()
    for item in chunks:
        if not isinstance(item, dict) or not item.get("id") or not item.get("text"):
            raise ValueError("Every chunk must contain non-empty 'id' and 'text' fields.")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID found: {item['id']}")
        seen_ids.add(item["id"])
    return chunks


def build_collection(chunks: list[dict], persist_path: Path = CHROMA_DIR,
                     collection_name: str = COLLECTION_NAME) -> int:
    """Rebuild the collection from the supplied chunks and return its count."""
    print(f"Loading embedding model on CPU: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    texts = [item["text"] for item in chunks]
    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RuntimeError("The embedding model returned an unexpected vector matrix.")

    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))

    # Rebuild by default so deleted/changed chunks cannot remain as stale records.
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        # Chroma raises when the collection does not exist. Avoid hiding other errors.
        if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
            raise

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "description": "Northstar student digital-services handbook chunks",
        },
    )

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_vectors = vectors[start:start + batch_size]
        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=batch_vectors.tolist(),
            documents=[item["text"] for item in batch],
            metadatas=[
                {
                    "chunk_index": int(index + start),
                    "word_count": int(item["word_count"]),
                    "start_word": int(item["start_word"]),
                    "end_word": int(item["end_word"]),
                    "source": "source_document.txt",
                }
                for index, item in enumerate(batch)
            ],
        )

    return collection.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local ChromaDB vector store.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DIR)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    count = build_collection(chunks, args.persist_dir)

    # Read back one stored record to verify persistence and inspect returned text.
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    # Query the collection directly using a stored vector and inspect the hit.
    stored_vector_value = collection.get(
        ids=[chunks[0]["id"]], include=["embeddings"]
    )["embeddings"][0]
    stored_vector = (
        stored_vector_value.tolist()
        if hasattr(stored_vector_value, "tolist")
        else list(stored_vector_value)
    )
    sample = collection.query(
        query_embeddings=[stored_vector],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    print("\nChromaDB build complete")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored records: {count}")
    if sample.get("ids") and sample["ids"][0]:
        print(f"Verification query result: {sample['ids'][0][0]}")
        print(f"Metadata: {sample['metadatas'][0][0]}")
        print(f"Cosine distance: {sample['distances'][0][0]:.4f}")
        print(f"Text preview: {sample['documents'][0][0][:240]}...")
    print(f"Persistent directory: {args.persist_dir}")


if __name__ == "__main__":
    main()
