import unittest

from rag_utils import ABSTENTION_ANSWER, build_context, format_source_summary, should_abstain


class GroundingHelperTests(unittest.TestCase):
    def test_empty_retrieval_abstains(self):
        self.assertTrue(should_abstain(None, 0.25))

    def test_below_threshold_abstains(self):
        self.assertTrue(should_abstain(0.19, 0.25))

    def test_at_threshold_does_not_abstain(self):
        self.assertFalse(should_abstain(0.25, 0.25))

    def test_context_contains_chunk_identifiers_and_text(self):
        context = build_context([
            {"id": "chunk_0001", "text": "Northstar-Secure is the student Wi-Fi network."}
        ])
        self.assertIn("[chunk_0001]", context)
        self.assertIn("Northstar-Secure", context)

    def test_empty_context_is_explicit(self):
        self.assertIn("No relevant context", build_context([]))

    def test_abstention_phrase_is_exact(self):
        self.assertEqual(ABSTENTION_ANSWER, "I don't know based on the provided context.")

    def test_source_summary_formats_score(self):
        summary = format_source_summary([
            {"id": "chunk_0002", "similarity": 0.73125}
        ])
        self.assertEqual(summary, ["chunk_0002 (cosine similarity: 0.731)"])


if __name__ == "__main__":
    unittest.main()
