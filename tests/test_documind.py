"""
DocuMind — Unit Tests
Tests for chunking logic, ingestion, search, and QA pipeline (using mocks).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.rag_engine import RAGEngine, Chunk


class TestChunking(unittest.TestCase):
    """Test the text chunking utility."""

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = RAGEngine._chunk_text(text, size=512, overlap=64)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_long_text_multiple_chunks(self):
        text = "word " * 300          # 1500 chars
        chunks = RAGEngine._chunk_text(text, size=200, overlap=40)
        self.assertGreater(len(chunks), 1)

    def test_overlap(self):
        text = "A" * 300
        chunks = RAGEngine._chunk_text(text, size=100, overlap=20)
        # Each subsequent chunk should share 20 chars with previous
        self.assertEqual(chunks[1][:20], chunks[0][-20:])

    def test_no_chunk_exceeds_size(self):
        text = "word " * 500
        chunks = RAGEngine._chunk_text(text, size=100, overlap=10)
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_whitespace_normalisation(self):
        text = "hello   \n\n  world"
        chunks = RAGEngine._chunk_text(text)
        self.assertNotIn("\n", chunks[0])
        self.assertNotIn("  ", chunks[0])


class TestRAGEngineIngest(unittest.TestCase):
    """Test ingestion with mocked Endee client."""

    def _make_engine(self):
        with patch("src.rag_engine.Endee") as MockEndee, \
             patch("src.rag_engine.SentenceTransformer") as MockST:

            mock_client   = MagicMock()
            mock_index    = MagicMock()
            mock_embedder = MagicMock()

            MockEndee.return_value   = mock_client
            MockST.return_value      = mock_embedder
            mock_client.list_indexes.return_value = []
            mock_client.get_index.return_value    = mock_index
            mock_embedder.encode.return_value     = [0.0] * 384

            engine = RAGEngine.__new__(RAGEngine)
            engine.client   = mock_client
            engine.index    = mock_index
            engine.embedder = mock_embedder
            return engine, mock_index

    def test_ingest_returns_correct_structure(self):
        engine, mock_index = self._make_engine()
        result = engine.ingest_text("Some test document content.", filename="test.txt")
        self.assertIn("doc_id",     result)
        self.assertIn("filename",   result)
        self.assertIn("num_chunks", result)
        self.assertEqual(result["filename"], "test.txt")

    def test_upsert_called_on_ingest(self):
        engine, mock_index = self._make_engine()
        engine.ingest_text("Hello world", filename="test.txt")
        mock_index.upsert.assert_called_once()

    def test_ingest_chunk_count(self):
        engine, _ = self._make_engine()
        text = "word " * 400   # ~2000 chars; with CHUNK_SIZE=512, overlap=64 → ~5 chunks
        result = engine.ingest_text(text, filename="big.txt")
        self.assertGreater(result["num_chunks"], 1)


class TestRAGEngineSearch(unittest.TestCase):
    """Test search / retrieval."""

    def _make_engine_with_results(self, raw_results):
        engine = RAGEngine.__new__(RAGEngine)

        mock_index    = MagicMock()
        mock_embedder = MagicMock()

        engine.index    = mock_index
        engine.embedder = mock_embedder
        mock_embedder.encode.return_value = [0.0] * 384
        mock_index.query.return_value     = raw_results
        return engine

    def test_search_returns_search_results(self):
        raw = [{"id": "doc1_c0001", "similarity": 0.95,
                "meta": {"doc_id": "doc1", "filename": "test.txt", "text": "hello"}}]
        engine = self._make_engine_with_results(raw)
        results = engine.search("hello")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].similarity, 0.95)
        self.assertEqual(results[0].filename,   "test.txt")

    def test_search_empty_results(self):
        engine = self._make_engine_with_results([])
        results = engine.search("random query")
        self.assertEqual(results, [])

    def test_build_context_formats_correctly(self):
        raw = [{"id": "a", "similarity": 0.9,
                "meta": {"doc_id": "d1", "filename": "foo.txt", "text": "relevant content"}}]
        engine = self._make_engine_with_results(raw)
        context, results = engine.build_context("query")
        self.assertIn("foo.txt",          context)
        self.assertIn("relevant content", context)
        self.assertEqual(len(results), 1)


class TestQAPipeline(unittest.TestCase):
    """Test the QA pipeline (generative + extractive modes)."""

    def _make_qa(self, search_results, use_llm=False):
        from src.qa_pipeline import QAPipeline, SearchResult

        mock_rag = MagicMock()
        mock_results = [
            SearchResult(
                chunk_id="c1", doc_id="d1", filename="test.txt",
                text="Paris is the capital of France.", similarity=0.98
            )
        ] if search_results else []

        mock_rag.build_context.return_value = (
            "Paris is the capital of France.",
            mock_results,
        )
        mock_rag.search.return_value = mock_results

        qa = QAPipeline.__new__(QAPipeline)
        qa.rag     = mock_rag
        qa.use_llm = use_llm
        return qa

    def test_extractive_answer_structure(self):
        qa = self._make_qa(search_results=True, use_llm=False)
        result = qa.ask("What is the capital of France?")
        self.assertIn("question", result)
        self.assertIn("answer",   result)
        self.assertIn("sources",  result)
        self.assertIn("mode",     result)

    def test_no_results_returns_graceful_message(self):
        qa = self._make_qa(search_results=False)
        result = qa.ask("Unknown topic")
        self.assertIn("couldn't find", result["answer"].lower())
        self.assertEqual(result["mode"], "no_results")

    def test_sources_contain_expected_fields(self):
        qa = self._make_qa(search_results=True, use_llm=False)
        result = qa.ask("What is the capital of France?")
        for s in result["sources"]:
            self.assertIn("filename",   s)
            self.assertIn("similarity", s)
            self.assertIn("excerpt",    s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
