import unittest

from chunk_document import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_chunks_respect_maximum_size(self):
        text = " ".join(f"word{i}" for i in range(610))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertTrue(chunks)
        self.assertTrue(all(item["word_count"] <= 100 for item in chunks))
        self.assertEqual(chunks[0]["word_count"], 100)

    def test_consecutive_chunks_share_requested_overlap(self):
        text = " ".join(f"word{i}" for i in range(230))
        chunks = chunk_text(text, chunk_size=100, overlap=25)
        first_words = chunks[0]["text"].split()
        second_words = chunks[1]["text"].split()
        self.assertEqual(first_words[-25:], second_words[:25])

    def test_chunk_metadata_and_ids_are_stable(self):
        chunks = chunk_text("one two three four five six", chunk_size=4, overlap=1)
        self.assertEqual(chunks[0]["id"], "chunk_0001")
        self.assertEqual(chunks[1]["id"], "chunk_0002")
        self.assertEqual(chunks[1]["start_word"], 3)
        self.assertEqual(chunks[1]["end_word"], 6)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("one two three", chunk_size=3, overlap=3)

    def test_negative_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("one two three", chunk_size=3, overlap=-1)


if __name__ == "__main__":
    unittest.main()
