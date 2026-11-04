"""Run a controlled before/after comparison of weak vs. explicit grounding prompts."""

from __future__ import annotations

import argparse

from rag_query import generate_answer, load_runtime, retrieve_chunks

DEFAULT_QUESTION = "What is the monthly tuition fee at Northstar Learning Institute?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a basic prompt with the explicit RAG grounding instruction."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    runtime = load_runtime()
    chunks = retrieve_chunks(args.question, runtime, top_k=args.top_k)
    print("\nQuestion:", args.question)
    print("\nRetrieved context IDs:", ", ".join(item["id"] for item in chunks) or "none")

    print("\n=== BEFORE: Basic prompt (no explicit refusal rule) ===")
    before = generate_answer(args.question, chunks, runtime, enforce_grounding=False)
    print(before)

    print("\n=== AFTER: Explicit grounding and abstention instructions ===")
    after = generate_answer(args.question, chunks, runtime, enforce_grounding=True)
    print(after)

    print(
        "\nRecord the actual outputs in reflections.md. One run is an observation, "
        "not proof that hallucinations are eliminated."
    )


if __name__ == "__main__":
    main()
