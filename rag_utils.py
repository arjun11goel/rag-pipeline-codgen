"""Small, dependency-light helpers used by the RAG pipeline and its tests."""

from __future__ import annotations

ABSTENTION_ANSWER = "I don't know based on the provided context."


def should_abstain(best_similarity: float | None,
                   minimum_similarity: float = 0.25) -> bool:
    """Return True when retrieval is empty or below the configured threshold."""
    if best_similarity is None:
        return True
    return best_similarity < minimum_similarity


def build_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved content with stable chunk identifiers for the prompt."""
    if not retrieved_chunks:
        return "[No relevant context was retrieved.]"

    sections = []
    for chunk in retrieved_chunks:
        chunk_id = chunk.get("id", "unknown_chunk")
        text = chunk.get("text", "").strip()
        sections.append(f"[{chunk_id}]\n{text}")
    return "\n\n---\n\n".join(sections)


def format_source_summary(retrieved_chunks: list[dict]) -> list[str]:
    """Return concise source labels for display below an answer."""
    return [
        f"{chunk['id']} (cosine similarity: {chunk['similarity']:.3f})"
        for chunk in retrieved_chunks
    ]
