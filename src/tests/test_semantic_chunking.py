"""Test cases for semantic chunking functions in utils.py"""

import pytest
from lex_db.utils import (
    clean_markdown,
    split_text_by_sections_with_headings,
    tokenize,
    reconstruct_text,
    chunk_section,
    split_text_by_semantic_chunks,
)


# ============ DUMMY TEST DATA ============

SAMPLE_MARKDOWN_SIMPLE: str = """# Introduktion

Dette er en simpel introduktion til emnet. Den indeholder bare nogle få sætninger som test.

## Undersektion

Her er lidt mere tekst under undersektionen. Den skal også opdeles korrekt i chunks.
"""

SAMPLE_MARKDOWN_WITH_FOOTER: str = """# Hovedemne

Dette er hovedindholdet af dokumentet som skal splittes op i semantiske chunks.

## Sektion 2

Her er mere indhold som skal processeres korrekt.

## Læs mere i Lex
- Link 1
- Link 2
"""

SAMPLE_MARKDOWN_COMPLEX: str = """# Dansk Grammatik

Dansk er et nordgermansk sprog som tales af omkring 6 millioner mennesker. 
Det er det officielle sprog i Danmark.

## Ordklasser

Der er flere vigtige ordklasser i dansk. Navneord er én af de vigtigste ordklasser.
Verber er en anden vigtig ordklasse som beskriver handlinger.

### Navneord

Navneord henviser til personer, steder, ting eller idéer. De kan være enten hankøn, hunkøn eller intetkøn.

### Verber

Verber udtrykker handlinger eller tilstande. De kan konjugeres efter person og tid.

## Syntaks

Sætninger sammensættes af ord i en bestemt rækkefølge. Dansk har som hovedregel ordstillingen SVO (Subjekt-Verbum-Objekt).

## Se også
- Dansk sprog
- Grammatik
- Nordiske sprog
"""

SAMPLE_TEXT_SHORT: str = "Dette er en kort tekst."

SAMPLE_TEXT_EMPTY: str = ""

SAMPLE_TEXT_WITH_MARKDOWN_METADATA: str = """# Emne

Her er noget indhold.

## Læs mere i Lex
- Link til Lex
"""


# ============ TESTS FOR TOKENIZE ============


class TestTokenize:
    """Test the tokenize function."""

    def test_tokenize_simple_text(self) -> None:
        text: str = "Dette er en test"
        result: list[str] = tokenize(text)
        assert result == ["Dette", "er", "en", "test"]

    def test_tokenize_empty_string(self) -> None:
        text: str = ""
        result: list[str] = tokenize(text)
        assert result == []

    def test_tokenize_with_punctuation(self) -> None:
        text: str = "Hej, verden!"
        result: list[str] = tokenize(text)
        assert result == ["Hej,", "verden!"]


# ============ TESTS FOR RECONSTRUCT_TEXT ============


class TestReconstructText:
    """Test the reconstruct_text function."""

    def test_reconstruct_simple(self) -> None:
        tokens: list[str] = ["Dette", "er", "en", "test"]
        result: str = reconstruct_text(tokens)
        assert result == "Dette er en test"

    def test_reconstruct_with_punctuation(self) -> None:
        tokens: list[str] = ["Hej", ",", "verden", "!"]
        result: str = reconstruct_text(tokens)
        assert result == "Hej, verden!"

    def test_reconstruct_empty(self) -> None:
        tokens: list[str] = []
        result: str = reconstruct_text(tokens)
        assert result == ""

    def test_reconstruct_with_parentheses(self) -> None:
        tokens: list[str] = ["Dette", "(", "vigtig", "info", ")", "er", "her"]
        result: str = reconstruct_text(tokens)
        assert result == "Dette (vigtig info) er her"


# ============ TESTS FOR CLEAN_MARKDOWN ============


class TestCleanMarkdown:
    """Test the clean_markdown function."""

    def test_clean_markdown_removes_footer(self) -> None:
        result: str = clean_markdown(SAMPLE_MARKDOWN_WITH_FOOTER)
        assert "Læs mere i Lex" not in result
        assert "Link 1" not in result

    def test_clean_markdown_empty_string(self) -> None:
        result: str = clean_markdown(SAMPLE_TEXT_EMPTY)
        assert result == ""

    def test_clean_markdown_preserves_content(self) -> None:
        result: str = clean_markdown(SAMPLE_MARKDOWN_SIMPLE)
        assert "Introduktion" in result
        assert "Undersektion" in result

    def test_clean_markdown_normalizes_whitespace(self) -> None:
        text_with_extra_newlines: str = "Linje 1\n\n\n\nLinje 2"
        result: str = clean_markdown(text_with_extra_newlines)
        assert "\n\n\n\n" not in result


# ============ TESTS FOR SPLIT_TEXT_BY_SECTIONS_WITH_HEADINGS ============


class TestSplitTextBySectionsWithHeadings:
    """Test the split_text_by_sections_with_headings function."""

    def test_split_simple_markdown(self) -> None:
        result: list[tuple[str, str]] = split_text_by_sections_with_headings(
            SAMPLE_MARKDOWN_SIMPLE
        )
        assert len(result) > 0
        # Should have tuples of (heading, content)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_split_empty_markdown(self) -> None:
        result: list[tuple[str, str]] = split_text_by_sections_with_headings(
            SAMPLE_TEXT_EMPTY
        )
        assert result == []

    def test_split_complex_markdown(self) -> None:
        result: list[tuple[str, str]] = split_text_by_sections_with_headings(
            SAMPLE_MARKDOWN_COMPLEX
        )
        assert len(result) > 0
        headings: list[str] = [heading for heading, content in result]
        assert any("Grammatik" in h for h in headings)


# ============ TESTS FOR CHUNK_SECTION ============


class TestChunkSection:
    """Test the chunk_section function."""

    def test_chunk_section_simple(self) -> None:
        heading: str = "Introduktion"
        text: str = "Dette er en lang tekst som skal opdeles i mindre chunks. " * 20
        result: list[str] = chunk_section(
            heading, text, min_chunk_size=5, chunk_size=100
        )
        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_chunk_section_empty_text(self) -> None:
        result: list[str] = chunk_section("Heading", "", min_chunk_size=5, chunk_size=100)
        assert result == []

    def test_chunk_section_too_short(self) -> None:
        heading: str = "Short"
        text: str = "En"
        result: list[str] = chunk_section(
            heading, text, min_chunk_size=100, chunk_size=100
        )
        assert result == []

    def test_chunk_section_respects_chunk_size(self) -> None:
        heading: str = "Test"
        text: str = "Dette er en test. " * 50
        result: list[str] = chunk_section(
            heading, text, min_chunk_size=5, chunk_size=50
        )
        # Each chunk should be reasonably sized (this is approximate due to sentence boundaries)
        assert all(len(chunk.split()) > 0 for chunk in result)


# ============ TESTS FOR SPLIT_TEXT_BY_SEMANTIC_CHUNKS ============


class TestSplitTextBySemanticChunks:
    """Test the split_text_by_semantic_chunks function."""

    def test_semantic_chunks_simple(self) -> None:
        result: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_SIMPLE, chunk_size=100
        )
        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_semantic_chunks_empty(self) -> None:
        result: list[str] = split_text_by_semantic_chunks(SAMPLE_TEXT_EMPTY)
        assert result == []

    def test_semantic_chunks_complex(self) -> None:
        result: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_COMPLEX, chunk_size=150, overlap=30
        )
        assert len(result) > 0
        # All chunks should be non-empty strings
        assert all(isinstance(chunk, str) and len(chunk.strip()) > 0 for chunk in result)

    def test_semantic_chunks_removes_metadata(self) -> None:
        result: list[str] = split_text_by_semantic_chunks(SAMPLE_MARKDOWN_WITH_FOOTER)
        # Result should not contain footer markers
        result_text: str = " ".join(result)
        assert "Læs mere i Lex" not in result_text

    def test_semantic_chunks_short_text(self) -> None:
        result: list[str] = split_text_by_semantic_chunks(
            SAMPLE_TEXT_SHORT, chunk_size=100, min_chunk_size=1
        )
        # Short text might result in 0 or 1 chunks depending on min_chunk_size
        assert isinstance(result, list)

    def test_semantic_chunks_with_overlap(self) -> None:
        """Test that overlap parameter doesn't cause errors."""
        result: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_COMPLEX,
            chunk_size=150,
            overlap=50,
            min_chunk_size=5,
        )
        assert len(result) > 0


# ============ INTEGRATION TESTS ============


class TestIntegration:
    """Integration tests for the full semantic chunking pipeline."""

    def test_full_pipeline_simple(self) -> None:
        """Test complete pipeline with simple markdown."""
        result: list[str] = split_text_by_semantic_chunks(SAMPLE_MARKDOWN_SIMPLE)
        assert len(result) > 0
        assert all(chunk for chunk in result)  # No empty chunks

    def test_full_pipeline_complex(self) -> None:
        """Test complete pipeline with complex markdown."""
        result: list[str] = split_text_by_semantic_chunks(SAMPLE_MARKDOWN_COMPLEX)
        assert len(result) > 0
        # Verify no duplicate chunks
        assert len(result) == len(set(result))

    def test_full_pipeline_different_chunk_sizes(self) -> None:
        """Test pipeline with different chunk size parameters."""
        small_chunks: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_COMPLEX, chunk_size=50
        )
        large_chunks: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_COMPLEX, chunk_size=200
        )

        # Smaller chunk_size should generally result in more chunks
        assert len(small_chunks) >= len(large_chunks)

    def test_full_pipeline_with_models(self) -> None:
        """Test pipeline works with different model names."""
        result1: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_SIMPLE,
            model="text-embedding-3-small",
        )
        result2: list[str] = split_text_by_semantic_chunks(
            SAMPLE_MARKDOWN_SIMPLE,
            model="text-embedding-3-large",
        )
        assert len(result1) > 0
        assert len(result2) > 0


# ============ EDGE CASES & ROBUSTNESS ============


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_only_headings_no_content(self) -> None:
        """Test markdown with only headings."""
        text: str = "# Heading 1\n## Heading 2\n### Heading 3"
        result: list[str] = split_text_by_semantic_chunks(text)
        # Should handle gracefully
        assert isinstance(result, list)

    def test_very_long_text(self) -> None:
        """Test with very long text."""
        long_text: str = "Dette er en test. " * 1000
        result: list[str] = split_text_by_semantic_chunks(long_text, chunk_size=100)
        assert len(result) > 1

    def test_unicode_characters(self) -> None:
        """Test with Danish special characters."""
        text: str = "# Æbler og Øl\n\nDette handler om æbler, øl og åben luft."
        result: list[str] = split_text_by_semantic_chunks(text)
        assert len(result) > 0

    def test_mixed_markdown_levels(self) -> None:
        """Test with mixed heading levels."""
        text: str = "# Level 1\n## Level 2\n### Level 3\n#### Level 4"
        result: list[str] = split_text_by_semantic_chunks(text)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])