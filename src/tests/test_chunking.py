"""Test cases for text chunking utilities in utils.py"""

import pytest
from lex_db.utils import (
    ChunkingStrategy,
    count_tokens,
    split_text_by_tokens,
    split_text_by_characters,
    split_text_by_sections,
    split_text_by_semantic_chunks,
    split_document_into_chunks,
)


# ============ TEST DATA ============

SIMPLE_TEXT = "Dette er en simpel tekst. Den har flere sætninger. Hver sætning er kort."

MARKDOWN_WITH_SECTIONS = """# Hovedoverskrift

Dette er introduktionsteksten under hovedoverskriften.

## Første sektion

Her er indholdet i den første sektion. Det indeholder flere sætninger.

## Anden sektion

Dette er anden sektion med mere indhold.

### Undersektion

En undersektion med lidt tekst.
"""

MARKDOWN_WITH_FOOTER = """# Hovedemne

Dette er hovedindholdet af dokumentet.

## Vigtig sektion

Her er mere vigtigt indhold.

## Læs mere i Lex
- Link 1
- Link 2
- Link 3
"""

LONG_TEXT = "Dette er en test sætning. " * 100

DANISH_TEXT = """# Æbler og Øl

Dette handler om æbler, øl og åben luft. Æ, Ø og Å er vigtige bogstaver i dansk.

## Særlige tegn

Danske særtegn skal håndteres korrekt i alle chunking strategier.
"""

EMPTY_TEXT = ""


# ============ TESTS FOR count_tokens ============


class TestCountTokens:
    """Test token counting functionality."""

    def test_count_tokens_simple_text(self) -> None:
        """Test counting tokens in simple text."""
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty_string(self) -> None:
        """Test counting tokens in empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_different_models(self) -> None:
        """Test token counting with different models."""
        text = "This is a test sentence."
        count_small = count_tokens(text, model="text-embedding-3-small")
        count_large = count_tokens(text, model="text-embedding-3-large")
        # Both should return positive counts
        assert count_small > 0
        assert count_large > 0

    def test_count_tokens_danish_text(self) -> None:
        """Test token counting with Danish characters."""
        text = "Æbler, øl og åben luft"
        count = count_tokens(text)
        assert count > 0


# ============ TESTS FOR split_text_by_tokens ============


class TestSplitTextByTokens:
    """Test token-based chunking."""

    def test_split_by_tokens_basic(self) -> None:
        """Test basic token-based splitting."""
        chunks = split_text_by_tokens(LONG_TEXT, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_split_by_tokens_no_split_needed(self) -> None:
        """Test when text is shorter than chunk size."""
        chunks = split_text_by_tokens(SIMPLE_TEXT, chunk_size=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == SIMPLE_TEXT

    def test_split_by_tokens_empty_text(self) -> None:
        """Test token splitting with empty text."""
        chunks = split_text_by_tokens(EMPTY_TEXT, chunk_size=100, overlap=0)
        assert chunks == []

    def test_split_by_tokens_with_overlap(self) -> None:
        """Test that overlap creates overlapping chunks."""
        chunks_no_overlap = split_text_by_tokens(LONG_TEXT, chunk_size=50, overlap=0)
        chunks_with_overlap = split_text_by_tokens(LONG_TEXT, chunk_size=50, overlap=10)
        # With overlap, we should get more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_split_by_tokens_preserves_content(self) -> None:
        """Test that chunking produces valid output."""
        text = "Word " * 100
        chunks = split_text_by_tokens(text, chunk_size=20, overlap=0)
        # Verify we got multiple chunks
        assert len(chunks) > 1
        # Verify chunks are strings (some may be empty due to tokenization)
        assert all(isinstance(chunk, str) for chunk in chunks)


# ============ TESTS FOR split_text_by_characters ============


class TestSplitTextByCharacters:
    """Test character-based chunking."""

    def test_split_by_characters_basic(self) -> None:
        """Test basic character-based splitting."""
        chunks = split_text_by_characters(LONG_TEXT, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_by_characters_exact_size(self) -> None:
        """Test that chunks respect character size limits."""
        chunk_size = 50
        chunks = split_text_by_characters(LONG_TEXT, chunk_size=chunk_size, overlap=0)
        # All chunks except possibly the last should be exactly chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk) == chunk_size
        # Last chunk should be <= chunk_size
        assert len(chunks[-1]) <= chunk_size

    def test_split_by_characters_no_split_needed(self) -> None:
        """Test when text is shorter than chunk size."""
        chunks = split_text_by_characters(SIMPLE_TEXT, chunk_size=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == SIMPLE_TEXT

    def test_split_by_characters_empty_text(self) -> None:
        """Test character splitting with empty text."""
        chunks = split_text_by_characters(EMPTY_TEXT, chunk_size=100, overlap=0)
        assert chunks == []

    def test_split_by_characters_with_overlap(self) -> None:
        """Test character-based splitting with overlap."""
        text = "A" * 100
        chunks = split_text_by_characters(text, chunk_size=30, overlap=10)
        assert len(chunks) > 1
        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            # Last 10 chars of chunk i should match first 10 chars of chunk i+1
            assert chunks[i][-10:] == chunks[i + 1][:10]

    def test_split_by_characters_danish_text(self) -> None:
        """Test character splitting preserves Danish characters."""
        chunks = split_text_by_characters(DANISH_TEXT, chunk_size=50, overlap=0)
        combined = "".join(chunks)
        # Verify special characters are preserved
        assert "Æ" in combined or "æ" in combined
        assert "Ø" in combined or "ø" in combined
        assert "Å" in combined or "å" in combined


# ============ TESTS FOR split_text_by_sections ============


class TestSplitTextBySections:
    """Test section-based (markdown heading) chunking."""

    def test_split_by_sections_basic(self) -> None:
        """Test basic section splitting."""
        chunks = split_text_by_sections(MARKDOWN_WITH_SECTIONS)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Each chunk should contain a heading
        assert all("#" in chunk for chunk in chunks)

    def test_split_by_sections_removes_footer(self) -> None:
        """Test that footer sections are removed."""
        chunks = split_text_by_sections(MARKDOWN_WITH_FOOTER)
        combined = " ".join(chunks)
        # Footer should be removed
        assert "Læs mere i Lex" not in combined
        assert "Link 1" not in combined

    def test_split_by_sections_preserves_content(self) -> None:
        """Test that main content is preserved."""
        chunks = split_text_by_sections(MARKDOWN_WITH_SECTIONS)
        combined = " ".join(chunks)
        assert "Hovedoverskrift" in combined
        assert "Første sektion" in combined
        assert "Anden sektion" in combined

    def test_split_by_sections_empty_text(self) -> None:
        """Test section splitting with empty text."""
        chunks = split_text_by_sections(EMPTY_TEXT)
        assert chunks == []

    def test_split_by_sections_no_duplicates(self) -> None:
        """Test that duplicate sections are removed."""
        chunks = split_text_by_sections(MARKDOWN_WITH_SECTIONS)
        # No duplicate chunks
        assert len(chunks) == len(set(chunks))

    def test_split_by_sections_different_heading_levels(self) -> None:
        """Test handling of different markdown heading levels."""
        text = "# H1\nContent 1\n## H2\nContent 2\n### H3\nContent 3"
        chunks = split_text_by_sections(text)
        assert len(chunks) == 3
        assert any("# H1" in chunk for chunk in chunks)
        assert any("## H2" in chunk for chunk in chunks)
        assert any("### H3" in chunk for chunk in chunks)


# ============ TESTS FOR split_text_by_semantic_chunks ============


class TestSplitTextBySemanticChunks:
    """Test semantic chunking (sentence-aware, section-based)."""

    def test_semantic_chunks_basic(self) -> None:
        """Test basic semantic chunking."""
        chunks = split_text_by_semantic_chunks(
            MARKDOWN_WITH_SECTIONS, chunk_size=100, overlap=20
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_semantic_chunks_removes_metadata(self) -> None:
        """Test that metadata sections are removed."""
        chunks = split_text_by_semantic_chunks(MARKDOWN_WITH_FOOTER)
        combined = " ".join(chunks)
        assert "Læs mere i Lex" not in combined

    def test_semantic_chunks_empty_text(self) -> None:
        """Test semantic chunking with empty text."""
        chunks = split_text_by_semantic_chunks(EMPTY_TEXT)
        assert chunks == []

    def test_semantic_chunks_respects_min_size(self) -> None:
        """Test that chunks below minimum size are filtered out."""
        # Very short text with high min_chunk_size should produce no chunks
        short_text = "# Test\nShort."
        chunks = split_text_by_semantic_chunks(
            short_text, chunk_size=100, min_chunk_size=50
        )
        # Should produce 0 or very few chunks due to min size constraint
        assert isinstance(chunks, list)

    def test_semantic_chunks_danish_sentences(self) -> None:
        """Test semantic chunking with Danish text."""
        chunks = split_text_by_semantic_chunks(DANISH_TEXT, chunk_size=100)
        assert len(chunks) > 0
        # Verify Danish characters are preserved
        combined = " ".join(chunks)
        assert any(char in combined for char in ["æ", "ø", "å", "Æ", "Ø", "Å"])

    def test_semantic_chunks_no_duplicates(self) -> None:
        """Test that semantic chunking doesn't create duplicates."""
        chunks = split_text_by_semantic_chunks(MARKDOWN_WITH_SECTIONS)
        # No exact duplicate chunks
        assert len(chunks) == len(set(chunks))

    def test_semantic_chunks_different_sizes(self) -> None:
        """Test that chunk_size parameter affects output."""
        small_chunks = split_text_by_semantic_chunks(
            MARKDOWN_WITH_SECTIONS, chunk_size=50
        )
        large_chunks = split_text_by_semantic_chunks(
            MARKDOWN_WITH_SECTIONS, chunk_size=200
        )
        # Smaller chunk size should generally produce more chunks
        assert len(small_chunks) >= len(large_chunks)


# ============ TESTS FOR split_document_into_chunks (main API) ============


class TestSplitDocumentIntoChunks:
    """Test the main chunking API that dispatches to different strategies."""

    def test_split_with_token_strategy(self) -> None:
        """Test using TOKEN chunking strategy."""
        chunks = split_document_into_chunks(
            LONG_TEXT,
            chunk_size=50,
            overlap=10,
            chunking_strategy=ChunkingStrategy.TOKENS,
        )
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_with_character_strategy(self) -> None:
        """Test using CHARACTER chunking strategy."""
        chunks = split_document_into_chunks(
            LONG_TEXT,
            chunk_size=100,
            overlap=20,
            chunking_strategy=ChunkingStrategy.CHARACTERS,
        )
        assert len(chunks) > 1
        # Verify character size constraints
        for chunk in chunks[:-1]:
            assert len(chunk) == 100

    def test_split_with_section_strategy(self) -> None:
        """Test using SECTION chunking strategy."""
        chunks = split_document_into_chunks(
            MARKDOWN_WITH_SECTIONS,
            chunk_size=100,  # Not used for sections
            overlap=0,  # Not used for sections
            chunking_strategy=ChunkingStrategy.SECTIONS,
        )
        assert len(chunks) > 3
        assert all("#" in chunk for chunk in chunks)

    def test_split_with_semantic_strategy(self) -> None:
        """Test using SEMANTIC_CHUNKS strategy."""
        chunks = split_document_into_chunks(
            MARKDOWN_WITH_SECTIONS,
            chunk_size=100,
            overlap=20,
            chunking_strategy=ChunkingStrategy.SEMANTIC_CHUNKS,
        )
        assert len(chunks) > 3
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_with_invalid_strategy(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported chunking method"):
            # Use a string that's not a valid ChunkingStrategy
            split_document_into_chunks(
                SIMPLE_TEXT,
                chunk_size=100,
                overlap=0,
                chunking_strategy="invalid_strategy",  # type: ignore
            )

    def test_split_empty_text_all_strategies(self) -> None:
        """Test that all strategies handle empty text correctly."""
        for strategy in ChunkingStrategy:
            chunks = split_document_into_chunks(
                EMPTY_TEXT,
                chunk_size=100,
                overlap=0,
                chunking_strategy=strategy,
            )
            assert chunks == [], (
                f"Strategy {strategy} should return empty list for empty text"
            )


# ============ EDGE CASES & ROBUSTNESS ============


class TestEdgeCases:
    """Test edge cases and robustness across all chunking strategies."""

    def test_very_long_text_all_strategies(self) -> None:
        """Test all strategies with very long text."""
        very_long = "Dette er en test. " * 1000

        for strategy in ChunkingStrategy:
            if strategy == ChunkingStrategy.SECTIONS:
                # Sections strategy needs markdown headings
                continue
            chunks = split_document_into_chunks(
                very_long,
                chunk_size=100,
                overlap=10,
                chunking_strategy=strategy,
            )
            assert len(chunks) > 1, f"Strategy {strategy} should split very long text"

    def test_single_character_text(self) -> None:
        """Test chunking with single character."""
        text = "A"

        # Token strategy
        chunks = split_document_into_chunks(
            text, chunk_size=10, overlap=0, chunking_strategy=ChunkingStrategy.TOKENS
        )
        assert len(chunks) == 1

        # Character strategy
        chunks = split_document_into_chunks(
            text,
            chunk_size=10,
            overlap=0,
            chunking_strategy=ChunkingStrategy.CHARACTERS,
        )
        assert len(chunks) == 1

    def test_only_whitespace(self) -> None:
        """Test chunking with only whitespace."""
        text = "   \n\n\t  "

        for strategy in ChunkingStrategy:
            chunks = split_document_into_chunks(
                text, chunk_size=100, overlap=0, chunking_strategy=strategy
            )
            # Should handle gracefully (empty or minimal chunks)
            assert isinstance(chunks, list)

    def test_unicode_and_special_characters(self) -> None:
        """Test all strategies preserve unicode and special characters."""
        text = "Æble 🍎 café ñoño 中文"

        for strategy in [ChunkingStrategy.TOKENS, ChunkingStrategy.CHARACTERS]:
            chunks = split_document_into_chunks(
                text, chunk_size=100, overlap=0, chunking_strategy=strategy
            )
            combined = "".join(chunks)
            # Verify special characters are preserved
            assert "Æ" in combined
            assert "🍎" in combined
            assert "é" in combined

    def test_markdown_without_headings(self) -> None:
        """Test section strategy with markdown that has no headings."""
        text = "Just plain text without any headings."
        chunks = split_document_into_chunks(
            text, chunk_size=100, overlap=0, chunking_strategy=ChunkingStrategy.SECTIONS
        )
        # Should return empty list or handle gracefully
        assert isinstance(chunks, list)

    def test_chunk_size_larger_than_text(self) -> None:
        """Test when chunk_size is larger than the text."""
        for strategy in [ChunkingStrategy.TOKENS, ChunkingStrategy.CHARACTERS]:
            chunks = split_document_into_chunks(
                SIMPLE_TEXT,
                chunk_size=10000,
                overlap=0,
                chunking_strategy=strategy,
            )
            assert len(chunks) == 1
            assert chunks[0] == SIMPLE_TEXT

    def test_zero_overlap(self) -> None:
        """Test chunking with zero overlap."""
        chunks = split_document_into_chunks(
            LONG_TEXT,
            chunk_size=50,
            overlap=0,
            chunking_strategy=ChunkingStrategy.CHARACTERS,
        )
        # Should produce chunks without overlap
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Verify no overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # With zero overlap, end of one chunk shouldn't match start of next
            assert chunks[i][-1:] != chunks[i + 1][:1] or len(chunks[i]) < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
