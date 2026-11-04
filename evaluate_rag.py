"""Run the bundled answerable/unanswerable question set through the RAG pipeline."""

from __future__ import annotations

import json

from rag_config import EVALUATION_QUESTIONS_PATH
from rag_query import load_runtime, rag_query


def main() -> None:
    questions = json.loads(EVALUATION_QUESTIONS_PATH.read_text(encoding="utf-8"))
    runtime = load_runtime()
    print("PromptForge Week 3 RAG evaluation")
    print("Review answer quality and citations manually; this script does not grade semantics.\n")

    for index, item in enumerate(questions, start=1):
        result = rag_query(item["question"], runtime=runtime, debug=False)
        expected = "answerable" if item["answerable"] else "unanswerable"
        actual = "abstained" if result["abstained"] else "generated an answer"
        print(f"[{index}] Expected: {expected} | System: {actual}")
        print("Q:", item["question"])
        print("A:", result["answer"])
        if result["retrieved_chunks"]:
            top = result["retrieved_chunks"][0]
            print(f"Top chunk: {top['id']} | similarity={top['similarity']:.3f}")
        print("-" * 72)


if __name__ == "__main__":
    main()
