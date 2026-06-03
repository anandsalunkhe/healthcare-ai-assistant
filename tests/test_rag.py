"""Unit tests for RAG pipeline components."""
from app.rag import _splitter


class TestTextSplitter:
    """Verify the text splitter configuration produces reasonable chunks."""

    def test_long_text_is_split(self):
        sample = "Healthcare is important. " * 200  # ~5000 chars
        chunks = _splitter.split_text(sample)
        assert len(chunks) > 1, "Long text should be split into multiple chunks."

    def test_chunk_size_respected(self):
        sample = "Patient discharge instructions. " * 200
        chunks = _splitter.split_text(sample)
        # Allow some tolerance for overlap
        for chunk in chunks:
            assert len(chunk) <= 700, (
                f"Chunk too long ({len(chunk)} chars); "
                f"expected ≤ CHUNK_SIZE + tolerance."
            )

    def test_short_text_single_chunk(self):
        sample = "Short text."
        chunks = _splitter.split_text(sample)
        assert len(chunks) == 1, "Short text should remain a single chunk."

    def test_empty_text(self):
        chunks = _splitter.split_text("")
        assert chunks == [], "Empty text should produce no chunks."
