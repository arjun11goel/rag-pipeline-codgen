import unittest
from types import SimpleNamespace

import numpy as np

from rag_query import RAGRuntime, generate_answer, rag_query, retrieve_chunks


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


class FakeCollection:
    def __init__(self, distance=0.2):
        self.distance = distance
        self.last_query = None

    def count(self):
        return 1

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "ids": [["chunk_0001"]],
            "documents": [["Northstar-Secure is the student wireless network."]],
            "metadatas": [[{"word_count": 9}]],
            "distances": [[self.distance]],
        }


class FakeCompletions:
    def __init__(self, content="Use Northstar-Secure. [chunk_0001]"):
        self.content = content
        self.last_request = None
        self.call_count = 0

    def create(self, **kwargs):
        self.last_request = kwargs
        self.call_count += 1
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeGroqClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def make_runtime(distance=0.2, answer="Use Northstar-Secure. [chunk_0001]", threshold=0.25):
    completions = FakeCompletions(answer)
    runtime = RAGRuntime(
        embedding_model=FakeEmbeddingModel(),
        collection=FakeCollection(distance=distance),
        groq_client=FakeGroqClient(completions),
        model_name="test-model",
        minimum_similarity=threshold,
    )
    return runtime, completions


class PipelineTests(unittest.TestCase):
    def test_retrieval_converts_cosine_distance_to_similarity(self):
        runtime, _ = make_runtime(distance=0.2)
        retrieved = retrieve_chunks("How do I connect?", runtime)
        self.assertEqual(retrieved[0]["id"], "chunk_0001")
        self.assertAlmostEqual(retrieved[0]["similarity"], 0.8)
        self.assertEqual(retrieved[0]["metadata"]["word_count"], 9)

    def test_low_similarity_abstains_without_calling_groq(self):
        runtime, completions = make_runtime(distance=0.9, threshold=0.25)
        result = rag_query("A question unrelated to the handbook", runtime=runtime)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], "I don't know based on the provided context.")
        self.assertEqual(completions.call_count, 0)

    def test_relevant_retrieval_calls_groq_with_grounding_instruction(self):
        runtime, completions = make_runtime(distance=0.2, threshold=0.25)
        result = rag_query("Which Wi-Fi should students use?", runtime=runtime)
        self.assertFalse(result["abstained"])
        self.assertIn("Northstar-Secure", result["answer"])
        request = completions.last_request
        self.assertEqual(request["model"], "test-model")
        self.assertIn("I don't know based on the provided context.", request["messages"][0]["content"])
        self.assertIn("chunk_0001", request["messages"][0]["content"])

    def test_baseline_comparison_prompt_is_separate_from_strict_prompt(self):
        runtime, completions = make_runtime(distance=0.2)
        chunks = retrieve_chunks("What is the Wi-Fi name?", runtime)
        generate_answer("What is the Wi-Fi name?", chunks, runtime, enforce_grounding=False)
        prompt = completions.last_request["messages"][0]["content"]
        self.assertIn("EXCERPTS:", prompt)
        self.assertNotIn("Answer the user's question using ONLY the supplied CONTEXT", prompt)


if __name__ == "__main__":
    unittest.main()
