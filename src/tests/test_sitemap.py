"""Unit tests for sitemap module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lex_db.sitemap import (
    derive_encyclopedia_id,
    derive_permalink,
    fetch_all_sitemaps,
    fetch_article_json,
    fetch_sitemap,
    parse_sitemap,
)


class TestDeriveEncyclopediaId:
    """Tests for derive_encyclopedia_id function."""

    def test_all_valid_subdomains(self) -> None:
        """Test all 20 valid encyclopedia subdomains."""
        test_cases = [
            ("https://denstoredanske.lex.dk/article", 1),
            ("https://trap.lex.dk/article", 2),
            ("https://biografiskleksikon.lex.dk/article", 3),
            ("https://gyldendalogpolitikensdanmarkshistorie.lex.dk/article", 4),
            ("https://danmarksoldtid.lex.dk/article", 5),
            ("https://teaterleksikon.lex.dk/article", 6),
            ("https://mytologi.lex.dk/article", 7),
            ("https://pattedyratlas.lex.dk/article", 8),
            ("https://dansklitteraturshistorie.lex.dk/article", 9),
            ("https://bornelitteratur.lex.dk/article", 10),
            ("https://symbolleksikon.lex.dk/article", 11),
            ("https://naturenidanmark.lex.dk/article", 12),
            ("https://om.lex.dk/article", 14),
            ("https://lex.dk/article", 15),
            ("https://kvindebiografiskleksikon.lex.dk/article", 16),
            ("https://medicin.lex.dk/article", 17),
            ("https://trap-groenland.lex.dk/article", 18),
            ("https://trap-faeroeerne.lex.dk/article", 19),
            ("https://danmarkshistorien.lex.dk/article", 20),
        ]

        for url, expected_id in test_cases:
            assert derive_encyclopedia_id(url) == expected_id

    def test_invalid_subdomain_raises_error(self) -> None:
        """Test that unknown subdomain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown subdomain"):
            derive_encyclopedia_id("https://unknown.lex.dk/article")

    def test_invalid_url_raises_error(self) -> None:
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL"):
            derive_encyclopedia_id("not-a-url")


class TestDerivePermalink:
    """Tests for derive_permalink function."""

    def test_simple_permalink(self) -> None:
        """Test simple permalink extraction."""
        url = "https://lex.dk/test-article"
        assert derive_permalink(url) == "test-article"

    def test_url_decoding(self) -> None:
        """Test URL decoding of special characters."""
        url = "https://lex.dk/eksteri%C3%B8r"
        assert derive_permalink(url) == "eksteriør"

    def test_multiple_special_characters(self) -> None:
        """Test multiple special characters."""
        url = "https://lex.dk/%C3%A6%C3%B8%C3%A5"
        assert derive_permalink(url) == "æøå"

    def test_trailing_slash_removed(self) -> None:
        """Test that trailing slash is removed."""
        url = "https://lex.dk/article/"
        assert derive_permalink(url) == "article"

    def test_nested_path(self) -> None:
        """Test nested path (should keep full path)."""
        url = "https://lex.dk/category/article"
        assert derive_permalink(url) == "category/article"

    def test_empty_path(self) -> None:
        """Test empty path."""
        url = "https://lex.dk/"
        assert derive_permalink(url) == ""


class TestParseSitemap:
    """Tests for parse_sitemap function."""

    def test_parse_valid_sitemap(self, sample_sitemap_xml: str) -> None:
        """Test parsing valid sitemap XML."""
        entries = parse_sitemap(sample_sitemap_xml)

        assert len(entries) == 2

        # Check first entry
        assert entries[0].url == "https://lex.dk/test-article"
        assert entries[0].lastmod == datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert entries[0].encyclopedia_id == 15
        assert entries[0].permalink == "test-article"

        # Check second entry
        assert entries[1].url == "https://om.lex.dk/another-article"
        assert entries[1].lastmod == datetime(
            2025, 1, 2, 15, 30, 0, tzinfo=timezone.utc
        )
        assert entries[1].encyclopedia_id == 14
        assert entries[1].permalink == "another-article"

    def test_parse_empty_sitemap(self) -> None:
        """Test parsing empty sitemap."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>"""
        entries = parse_sitemap(xml)
        assert entries == []

    def test_parse_sitemap_missing_loc(self) -> None:
        """Test parsing sitemap with missing <loc> element."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""
        entries = parse_sitemap(xml)
        assert entries == []

    def test_parse_sitemap_missing_lastmod(self) -> None:
        """Test parsing sitemap with missing <lastmod> element."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/test</loc>
    </url>
</urlset>"""
        entries = parse_sitemap(xml)
        assert entries == []

    def test_parse_sitemap_invalid_url(self) -> None:
        """Test parsing sitemap with invalid URL (unknown subdomain)."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://invalid.lex.dk/test</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""
        entries = parse_sitemap(xml)
        assert entries == []  # Should skip invalid entries

    def test_parse_invalid_xml(self) -> None:
        """Test parsing invalid XML raises error."""
        with pytest.raises(Exception):  # ET.ParseError
            parse_sitemap("not valid xml")


@pytest.mark.asyncio
class TestFetchSitemap:
    """Tests for fetch_sitemap function."""

    async def test_fetch_success(self, sample_sitemap_xml: str) -> None:
        """Test successful sitemap fetch."""
        mock_response = MagicMock()
        mock_response.text = sample_sitemap_xml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await fetch_sitemap("https://lex.dk/.sitemap/sitemap1.xml")
            assert result == sample_sitemap_xml

    async def test_fetch_retry_logic(self, sample_sitemap_xml: str) -> None:
        """Test retry logic on failure."""
        mock_response = MagicMock()
        mock_response.text = sample_sitemap_xml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            # Fail twice, then succeed
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=[
                    httpx.RequestError("Error 1"),
                    httpx.RequestError("Error 2"),
                    mock_response,
                ]
            )

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_sitemap("https://lex.dk/.sitemap/sitemap1.xml")
                assert result == sample_sitemap_xml

    async def test_fetch_max_retries_exceeded(self) -> None:
        """Test that max retries raises error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Persistent error")
            )

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.RequestError):
                    await fetch_sitemap("https://lex.dk/.sitemap/sitemap1.xml")


@pytest.mark.asyncio
class TestFetchArticleJson:
    """Tests for fetch_article_json function."""

    async def test_fetch_success(self, sample_article_json: dict) -> None:
        """Test successful article JSON fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_article_json
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await fetch_article_json("https://lex.dk/test")
            assert result == sample_article_json

    async def test_fetch_404(self) -> None:
        """Test handling 404 response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "404", request=MagicMock(), response=MagicMock()
                )
            )

            with pytest.raises(httpx.HTTPStatusError):
                await fetch_article_json("https://lex.dk/nonexistent")


@pytest.mark.asyncio
class TestFetchAllSitemaps:
    """Tests for fetch_all_sitemaps function."""

    async def test_default_encyclopedia_filter(self) -> None:
        """Test that default filter includes only IDs 14, 15, 18, 19, 20."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/article1</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
    <url>
        <loc>https://denstoredanske.lex.dk/article2</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
    <url>
        <loc>https://om.lex.dk/article3</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""

        with patch(
            "lex_db.sitemap.fetch_sitemap", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = sitemap_xml

            entries = await fetch_all_sitemaps()

            # Should only include encyclopedia IDs 14 and 15 (not 1)
            assert len(entries) == 2
            assert all(e.encyclopedia_id in {14, 15, 18, 19, 20} for e in entries)

    async def test_custom_encyclopedia_filter(self) -> None:
        """Test filtering by custom encyclopedia IDs."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/article1</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
    <url>
        <loc>https://om.lex.dk/article2</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""

        with patch(
            "lex_db.sitemap.fetch_sitemap", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = sitemap_xml

            entries = await fetch_all_sitemaps(encyclopedia_ids={15})

            # Should only include encyclopedia ID 15
            assert len(entries) == 1
            assert entries[0].encyclopedia_id == 15

    async def test_deduplication_keeps_most_recent(self) -> None:
        """Test that deduplication keeps entry with most recent lastmod."""
        sitemap1_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/duplicate</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""

        sitemap2_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/duplicate</loc>
        <lastmod>2025-01-02T12:00:00Z</lastmod>
    </url>
</urlset>"""

        with patch(
            "lex_db.sitemap.fetch_sitemap", new_callable=AsyncMock
        ) as mock_fetch:
            # Return different XML for different sitemap URLs
            mock_fetch.side_effect = [
                sitemap1_xml,
                sitemap2_xml,
                sitemap1_xml,
                sitemap1_xml,
                sitemap1_xml,
                sitemap1_xml,
            ]

            entries = await fetch_all_sitemaps()

            # Should have only one entry with the most recent date
            assert len(entries) == 1
            assert entries[0].lastmod == datetime(
                2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc
            )

    async def test_partial_failure_continues(self) -> None:
        """Test that failure of one sitemap doesn't stop others."""
        # Create different XML for each sitemap to avoid deduplication
        xml_templates = [
            """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://lex.dk/article{}</loc>
        <lastmod>2025-01-01T12:00:00Z</lastmod>
    </url>
</urlset>"""
        ]

        with patch(
            "lex_db.sitemap.fetch_sitemap", new_callable=AsyncMock
        ) as mock_fetch:
            # First sitemap fails, others succeed with different articles
            mock_fetch.side_effect = [
                httpx.RequestError("Error"),
                xml_templates[0].format("1"),
                xml_templates[0].format("2"),
                xml_templates[0].format("3"),
                xml_templates[0].format("4"),
                xml_templates[0].format("5"),
            ]

            entries = await fetch_all_sitemaps()

            # Should still get entries from successful sitemaps (5 different articles)
            assert len(entries) == 5

    async def test_invalid_encyclopedia_ids_raises_error(self) -> None:
        """Test that invalid encyclopedia IDs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid encyclopedia IDs"):
            await fetch_all_sitemaps(encyclopedia_ids={13, 99})

    async def test_fetches_all_six_sitemaps(self) -> None:
        """Test that all 6 sitemaps are fetched."""
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>"""

        with patch(
            "lex_db.sitemap.fetch_sitemap", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = empty_xml

            await fetch_all_sitemaps()

            # Should have called fetch_sitemap 6 times
            assert mock_fetch.call_count == 6

            # Verify URLs
            called_urls = [call[0][0] for call in mock_fetch.call_args_list]
            expected_urls = [
                "https://lex.dk/.sitemap/sitemap1.xml",
                "https://lex.dk/.sitemap/sitemap2.xml",
                "https://lex.dk/.sitemap/sitemap3.xml",
                "https://lex.dk/.sitemap/sitemap4.xml",
                "https://lex.dk/.sitemap/sitemap5.xml",
                "https://lex.dk/.sitemap/sitemap6.xml",
            ]
            assert sorted(called_urls) == sorted(expected_urls)
