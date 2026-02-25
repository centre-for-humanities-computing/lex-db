"""Unit tests for sync_articles script."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from scripts.sync_articles import (
    categorize_articles,
    fetch_articles_batch,
    parse_encyclopedia_ids,
)
from lex_db.sitemap import SitemapEntry


class TestParseEncyclopediaIds:
    """Tests for parse_encyclopedia_ids function."""

    def test_parse_valid_ids(self) -> None:
        """Test parsing valid comma-separated IDs."""
        result = parse_encyclopedia_ids("14,15,18")
        assert result == {14, 15, 18}

    def test_parse_single_id(self) -> None:
        """Test parsing a single ID."""
        result = parse_encyclopedia_ids("14")
        assert result == {14}

    def test_parse_with_spaces(self) -> None:
        """Test parsing IDs with spaces."""
        result = parse_encyclopedia_ids("14, 15, 18")
        assert result == {14, 15, 18}

    def test_parse_none_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_encyclopedia_ids(None)
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = parse_encyclopedia_ids("")
        assert result is None

    def test_invalid_id_raises_error(self) -> None:
        """Test that invalid encyclopedia ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid encyclopedia IDs"):
            parse_encyclopedia_ids("13,14")  # 13 is not valid

    def test_non_numeric_raises_error(self) -> None:
        """Test that non-numeric input raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_encyclopedia_ids("14,abc,15")


class TestCategorizeArticles:
    """Tests for categorize_articles function."""

    def test_new_article(self) -> None:
        """Test that article in sitemap but not in DB is categorized as new."""
        sitemap_entries = [
            SitemapEntry(
                url="https://lex.dk/new-article",
                lastmod=datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="new-article",
            )
        ]
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {}
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == ["https://lex.dk/new-article"]
        assert modified_urls == []
        assert deleted_ids == []
        assert unchanged_count == 0

    def test_modified_article(self) -> None:
        """Test that article with newer sitemap timestamp is categorized as modified."""
        sitemap_entries = [
            SitemapEntry(
                url="https://lex.dk/article",
                lastmod=datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="article",
            )
        ]
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("article", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc))
        }
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == []
        assert modified_urls == ["https://lex.dk/article"]
        assert deleted_ids == []
        assert unchanged_count == 0

    def test_unchanged_article(self) -> None:
        """Test that article with same timestamp is categorized as unchanged."""
        timestamp = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        sitemap_entries = [
            SitemapEntry(
                url="https://lex.dk/article",
                lastmod=timestamp,
                encyclopedia_id=15,
                permalink="article",
            )
        ]
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("article", 15, timestamp)
        }
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == []
        assert modified_urls == []
        assert deleted_ids == []
        assert unchanged_count == 1

    def test_deleted_article(self) -> None:
        """Test that article in DB but not in sitemap is categorized as deleted."""
        sitemap_entries: list[SitemapEntry] = []
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("article", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc))
        }
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == []
        assert modified_urls == []
        assert deleted_ids == [1]
        assert unchanged_count == 0

    def test_deleted_article_filtered_by_encyclopedia(self) -> None:
        """Test that deletion is filtered by encyclopedia_ids."""
        sitemap_entries: list[SitemapEntry] = []
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("article1", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)),
            2: ("article2", 14, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)),
        }
        encyclopedia_ids = {15}  # Only syncing encyclopedia 15

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        # Only article from encyclopedia 15 should be marked for deletion
        assert deleted_ids == [1]

    def test_none_timestamp_treated_as_modified(self) -> None:
        """Test that None timestamp in DB is treated as needing update."""
        sitemap_entries = [
            SitemapEntry(
                url="https://lex.dk/article",
                lastmod=datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="article",
            )
        ]
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("article", 15, None)
        }
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == []
        assert modified_urls == ["https://lex.dk/article"]
        assert deleted_ids == []
        assert unchanged_count == 0

    def test_mixed_categorization(self) -> None:
        """Test categorization with mix of new, modified, unchanged, and deleted."""
        sitemap_entries = [
            SitemapEntry(
                url="https://lex.dk/new",
                lastmod=datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="new",
            ),
            SitemapEntry(
                url="https://lex.dk/modified",
                lastmod=datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="modified",
            ),
            SitemapEntry(
                url="https://lex.dk/unchanged",
                lastmod=datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc),
                encyclopedia_id=15,
                permalink="unchanged",
            ),
        ]
        db_metadata: dict[int, tuple[str, int, datetime | None]] = {
            1: ("modified", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)),
            2: ("unchanged", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)),
            3: ("deleted", 15, datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)),
        }
        encyclopedia_ids = None

        new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
            sitemap_entries, db_metadata, encyclopedia_ids
        )

        assert new_urls == ["https://lex.dk/new"]
        assert modified_urls == ["https://lex.dk/modified"]
        assert deleted_ids == [3]
        assert unchanged_count == 1


class TestFetchArticlesBatch:
    """Tests for fetch_articles_batch function."""

    @pytest.mark.asyncio
    async def test_fetch_empty_list(self) -> None:
        """Test fetching with empty URL list."""
        result = await fetch_articles_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_successful(self) -> None:
        """Test successful article fetching."""
        mock_article = {"id": 1, "title": "Test", "url": "https://lex.dk/test"}

        with patch(
            "scripts.sync_articles.fetch_article_json", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_article

            result = await fetch_articles_batch(
                ["https://lex.dk/test"], max_concurrent=1
            )

            assert len(result) == 1
            assert result[0] == mock_article
            mock_fetch.assert_called_once_with("https://lex.dk/test")

    @pytest.mark.asyncio
    async def test_fetch_with_failures(self) -> None:
        """Test that failures are handled gracefully."""
        mock_article = {"id": 1, "title": "Test", "url": "https://lex.dk/test"}

        with patch(
            "scripts.sync_articles.fetch_article_json", new_callable=AsyncMock
        ) as mock_fetch:
            # First call succeeds, second fails
            mock_fetch.side_effect = [mock_article, Exception("Network error")]

            result = await fetch_articles_batch(
                ["https://lex.dk/test1", "https://lex.dk/test2"], max_concurrent=2
            )

            # Only successful fetch should be in results
            assert len(result) == 1
            assert result[0] == mock_article

    @pytest.mark.asyncio
    async def test_rate_limiting(self) -> None:
        """Test that rate limiting is applied."""
        mock_article = {"id": 1, "title": "Test"}

        with patch(
            "scripts.sync_articles.fetch_article_json", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_article

            # Fetch 5 articles with max 2 concurrent
            urls = [f"https://lex.dk/test{i}" for i in range(5)]
            result = await fetch_articles_batch(urls, max_concurrent=2)

            assert len(result) == 5
            assert mock_fetch.call_count == 5

    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        """Test that 429 errors trigger retry with backoff."""
        mock_article = {"id": 1, "title": "Test"}

        with patch(
            "scripts.sync_articles.fetch_article_json", new_callable=AsyncMock
        ) as mock_fetch:
            # First call raises 429, second succeeds
            mock_fetch.side_effect = [
                Exception("Client error '429 Too Many Requests'"),
                mock_article,
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_articles_batch(
                    ["https://lex.dk/test"], max_concurrent=1
                )

                # Should succeed after retry
                assert len(result) == 1
                assert result[0] == mock_article
                assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test that article is skipped after max retries."""
        with patch(
            "scripts.sync_articles.fetch_article_json", new_callable=AsyncMock
        ) as mock_fetch:
            # Always raise 429
            mock_fetch.side_effect = Exception("Client error '429 Too Many Requests'")

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_articles_batch(
                    ["https://lex.dk/test"], max_concurrent=1
                )

                # Should return empty list after all retries fail
                assert len(result) == 0
                assert mock_fetch.call_count == 7  # max_retries = 7
