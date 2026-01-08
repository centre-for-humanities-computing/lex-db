"""Tests for HTML to Markdown conversion functionality."""

import pytest
from lex_db.utils import convert_article_json_to_markdown


class TestConvertArticleJsonToMarkdown:
    """Test the convert_article_json_to_markdown function."""

    def test_convert_accepts_dict_input(self) -> None:
        """Test that function accepts dictionary input."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Test content</p>",
        }
        result = convert_article_json_to_markdown(article_data)
        assert isinstance(result, str)
        assert result  # Should return non-empty string

    def test_convert_accepts_json_string_input(self) -> None:
        """Test that function accepts JSON string input."""
        json_string = '{"id": 12345, "title": "Test Article", "xhtml_body": "<p>Test content</p>"}'
        result = convert_article_json_to_markdown(json_string)
        assert isinstance(result, str)
        assert result

    def test_convert_raises_on_malformed_json(self) -> None:
        """Test that malformed JSON raises ValueError."""
        malformed_json = '{"id": 12345, "title": "Test"'  # Missing closing brace
        with pytest.raises(ValueError, match="Malformed JSON"):
            convert_article_json_to_markdown(malformed_json)

    def test_convert_raises_on_missing_id(self) -> None:
        """Test that missing 'id' field raises ValueError."""
        article_data = {
            "title": "Test Article",
            "xhtml_body": "<p>Test content</p>",
        }
        with pytest.raises(ValueError, match="Missing required fields.*id"):
            convert_article_json_to_markdown(article_data)

    def test_convert_raises_on_missing_title(self) -> None:
        """Test that missing 'title' field raises ValueError."""
        article_data = {
            "id": 12345,
            "xhtml_body": "<p>Test content</p>",
        }
        with pytest.raises(ValueError, match="Missing required fields.*title"):
            convert_article_json_to_markdown(article_data)

    def test_convert_raises_on_missing_xhtml_body(self) -> None:
        """Test that missing 'xhtml_body' field raises ValueError."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
        }
        with pytest.raises(ValueError, match="Missing required fields.*xhtml_body"):
            convert_article_json_to_markdown(article_data)

    def test_convert_raises_on_multiple_missing_fields(self) -> None:
        """Test that multiple missing fields are reported."""
        article_data = {"id": 12345}
        with pytest.raises(ValueError, match="Missing required fields"):
            convert_article_json_to_markdown(article_data)

    def test_convert_handles_optional_fields(self) -> None:
        """Test that optional fields (url, changed_at, metadata) are handled gracefully."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Test content</p>",
            "url": "https://lex.dk/test",
            "changed_at": "2025-06-23T09:59:04.573+02:00",
            "metadata": {"key": "value"},
        }
        result = convert_article_json_to_markdown(article_data)
        assert isinstance(result, str)

    def test_convert_basic_html_paragraph(self) -> None:
        """Test conversion of simple HTML paragraph to Markdown."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<p>This is a test paragraph.</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert result == "This is a test paragraph."

    def test_convert_empty_html(self) -> None:
        """Test that empty HTML returns empty string."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert result == ""

    def test_convert_whitespace_only_html(self) -> None:
        """Test that whitespace-only HTML returns empty string."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "   \n\t  ",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert result == ""

    def test_convert_with_bold_text(self) -> None:
        """Test conversion of bold text."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<p>This is <strong>bold</strong> text.</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "**bold**" in result

    def test_convert_with_italic_text(self) -> None:
        """Test conversion of italic text."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<p>This is <em>italic</em> text.</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "*italic*" in result

    def test_convert_with_headings(self) -> None:
        """Test conversion of HTML headings to Markdown."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<h2>Section 1</h2><p>Content</p><h3>Subsection</h3>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "## Section 1" in result
        assert "### Subsection" in result

    def test_convert_with_unordered_list(self) -> None:
        """Test conversion of unordered list."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<ul><li>Item 1</li><li>Item 2</li></ul>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "- Item 1" in result or "* Item 1" in result
        assert "- Item 2" in result or "* Item 2" in result

    def test_convert_with_ordered_list(self) -> None:
        """Test conversion of ordered list."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<ol><li>First</li><li>Second</li></ol>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "1. First" in result or "1) First" in result
        assert "2. Second" in result or "2) Second" in result

    def test_convert_with_links(self) -> None:
        """Test conversion of HTML links to Markdown."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": '<p>Visit <a href="https://lex.dk/test">this link</a>.</p>',
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "[this link](https://lex.dk/test)" in result

    def test_convert_with_internal_crossref_links(self) -> None:
        """Test that internal crossref links are preserved as full URLs."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": '<p>See <a class="crossref" href="https://lex.dk/Socialdemokratiet">Socialdemokratiet</a>.</p>',
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "[Socialdemokratiet](https://lex.dk/Socialdemokratiet)" in result

    def test_convert_with_internal_class_links(self) -> None:
        """Test that internal class links are preserved as full URLs."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": '<p>See <a class="internal" href="https://lex.dk/analytisk_geometri">analytisk geometri</a>.</p>',
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "[analytisk geometri](https://lex.dk/analytisk_geometri)" in result

    def test_convert_with_danish_characters(self) -> None:
        """Test that Danish special characters are preserved."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<p>Æble, øl, and å are Danish letters.</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "Æble" in result
        assert "øl" in result
        assert "å" in result

    def test_convert_with_inline_latex(self) -> None:
        """Test that inline LaTeX notation is preserved."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": r"<p>Krumningen defineres som \(\kappa(p) =\frac{1}{\rho(p)}\).</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert r"\(\kappa(p) =\frac{1}{\rho(p)}\)" in result

    def test_convert_with_display_latex(self) -> None:
        """Test that display LaTeX notation is preserved."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": r"<p>\[\int_{\mathcal{F}} \text{K}(p) d\mu=2\pi\chi .\]</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert r"\[\int_{\mathcal{F}} \text{K}(p) d\mu=2\pi\chi .\]" in result

    def test_convert_with_nested_formatting(self) -> None:
        """Test conversion of nested HTML formatting."""
        article_data = {
            "id": 12345,
            "title": "Test",
            "xhtml_body": "<p>This is <strong><em>bold and italic</em></strong> text.</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        # Should contain both bold and italic markers
        assert (
            "***bold and italic***" in result
            or "**_bold and italic_**" in result
            or "_**bold and italic**_" in result
        )


class TestMetadataFormatting:
    """Test metadata appendix formatting."""

    def test_metadata_appendix_included_by_default(self) -> None:
        """Test that metadata appendix is included by default."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
        }
        result = convert_article_json_to_markdown(article_data)
        assert "## Article Metadata" in result
        assert "**Article ID:** 12345" in result
        assert "**Title:** Test Article" in result

    def test_metadata_appendix_excluded_when_disabled(self) -> None:
        """Test that metadata appendix can be excluded."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
        }
        result = convert_article_json_to_markdown(article_data, include_metadata=False)
        assert "## Article Metadata" not in result
        assert "**Article ID:**" not in result

    def test_metadata_with_url(self) -> None:
        """Test that URL is included in metadata when present."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
            "url": "https://lex.dk/test_article",
        }
        result = convert_article_json_to_markdown(article_data)
        assert "**URL:** https://lex.dk/test_article" in result

    def test_metadata_with_changed_at(self) -> None:
        """Test that last modified timestamp is included when present."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
            "changed_at": "2025-06-23T09:59:04.573+02:00",
        }
        result = convert_article_json_to_markdown(article_data)
        assert "**Last Modified:** 2025-06-23T09:59:04.573+02:00" in result

    def test_metadata_with_additional_fields(self) -> None:
        """Test that additional metadata fields are formatted correctly."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
            "metadata": {
                "gender": "k",
                "firstname": "Mette",
                "lastname": "Frederiksen",
                "birth_date": "19.11.1977",
            },
        }
        result = convert_article_json_to_markdown(article_data)
        assert "**Additional Metadata:**" in result
        assert "- **Gender:** k" in result
        assert "- **Firstname:** Mette" in result
        assert "- **Lastname:** Frederiksen" in result
        assert "- **Birth Date:** 19.11.1977" in result

    def test_metadata_key_formatting(self) -> None:
        """Test that metadata keys are formatted from snake_case to Title Case."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
            "metadata": {
                "birth_date": "01.01.2000",
                "some_long_key_name": "value",
            },
        }
        result = convert_article_json_to_markdown(article_data)
        assert "- **Birth Date:**" in result
        assert "- **Some Long Key Name:**" in result

    def test_metadata_with_empty_dict(self) -> None:
        """Test that empty metadata dict doesn't add additional metadata section."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
            "metadata": {},
        }
        result = convert_article_json_to_markdown(article_data)
        assert "## Article Metadata" in result
        assert "**Additional Metadata:**" not in result

    def test_metadata_separator_format(self) -> None:
        """Test that metadata section has proper separator."""
        article_data = {
            "id": 12345,
            "title": "Test Article",
            "xhtml_body": "<p>Content</p>",
        }
        result = convert_article_json_to_markdown(article_data)
        assert "\n---\n" in result

    def test_full_metadata_example(self) -> None:
        """Test complete metadata appendix with all fields."""
        article_data = {
            "id": 136425,
            "title": "Mette Frederiksen",
            "xhtml_body": "<p>er en dansk socialdemokratisk politiker</p>",
            "url": "https://lex.dk/Mette_Frederiksen",
            "changed_at": "2025-06-23T09:59:04.573+02:00",
            "metadata": {
                "gender": "k",
                "lastname": "Frederiksen",
                "firstname": "Mette",
                "birth_date": "19.11.1977",
                "birthplace": "Aalborg",
            },
        }
        result = convert_article_json_to_markdown(article_data)

        # Check content
        assert "er en dansk socialdemokratisk politiker" in result

        # Check metadata section structure
        assert "---" in result
        assert "## Article Metadata" in result
        assert "**Article ID:** 136425" in result
        assert "**Title:** Mette Frederiksen" in result
        assert "**URL:** https://lex.dk/Mette_Frederiksen" in result
        assert "**Last Modified:** 2025-06-23T09:59:04.573+02:00" in result

        # Check additional metadata
        assert "**Additional Metadata:**" in result
        assert "- **Gender:** k" in result
        assert "- **Lastname:** Frederiksen" in result
        assert "- **Firstname:** Mette" in result
        assert "- **Birth Date:** 19.11.1977" in result
        assert "- **Birthplace:** Aalborg" in result
