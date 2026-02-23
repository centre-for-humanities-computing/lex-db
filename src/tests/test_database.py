"""Unit tests for database module."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from lex_db.database import (
    get_articles_by_ids,
    search_lex_fts,
    SearchResults,
)


class TestGetArticlesByIds:
    """Tests for get_articles_by_ids function."""

    def test_empty_ids_list(self) -> None:
        """Test empty IDs list returns empty results."""
        result = get_articles_by_ids([], limit=50)

        assert isinstance(result, SearchResults)
        assert result.entries == []
        assert result.total == 0
        assert result.limit == 50

    @patch("lex_db.database.get_db_connection")
    def test_single_article(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test fetching single article by ID."""
        # Setup mock
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        # Mock count query
        count_result = {"count": 1}
        # Mock article query
        article_row = {
            "id": 1,
            "xhtml_md": "Test content",
            "rank": 0.0,
            "permalink": "test-article",
            "headword": "Test Article",
            "encyclopedia_id": 1,
        }

        mock_db_connection.execute.return_value.fetchone.return_value = count_result
        mock_db_connection.execute.return_value.fetchall.return_value = [article_row]

        # Execute
        result = get_articles_by_ids([1], limit=50)

        # Verify
        assert result.total == 1
        assert len(result.entries) == 1
        assert result.entries[0].id == 1
        assert result.entries[0].title == "Test Article"
        assert result.entries[0].url == "https://denstoredanske.lex.dk/test-article"

    @patch("lex_db.database.get_db_connection")
    def test_multiple_articles(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test fetching multiple articles by IDs."""
        # Setup mock
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        count_result = {"count": 2}
        article_rows = [
            {
                "id": 1,
                "xhtml_md": "Content 1",
                "rank": 0.0,
                "permalink": "article-1",
                "headword": "Article 1",
                "encyclopedia_id": 1,
            },
            {
                "id": 2,
                "xhtml_md": "Content 2",
                "rank": 0.0,
                "permalink": "article-2",
                "headword": "Article 2",
                "encyclopedia_id": 2,
            },
        ]

        mock_db_connection.execute.return_value.fetchone.return_value = count_result
        mock_db_connection.execute.return_value.fetchall.return_value = article_rows

        # Execute
        result = get_articles_by_ids([1, 2], limit=50)

        # Verify
        assert result.total == 2
        assert len(result.entries) == 2
        assert result.entries[0].id == 1
        assert result.entries[1].id == 2


class TestSearchLexFts:
    """Tests for search_lex_fts function."""

    @patch("lex_db.database.get_db_connection")
    def test_empty_query_no_ids(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test empty query without IDs returns empty results."""
        result = search_lex_fts("", ids=None, limit=50)

        assert isinstance(result, SearchResults)
        assert result.entries == []
        assert result.total == 0

    @patch("lex_db.database.get_articles_by_ids")
    def test_empty_query_with_ids(self, mock_get_articles: MagicMock) -> None:
        """Test empty query with IDs calls get_articles_by_ids."""
        mock_get_articles.return_value = SearchResults(entries=[], total=0, limit=50)

        search_lex_fts("", ids=[1, 2], limit=50)

        mock_get_articles.assert_called_once_with([1, 2], limit=50)

    @patch("lex_db.database.get_db_connection")
    def test_whitespace_query(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test whitespace-only query returns empty results."""
        result = search_lex_fts("   ", ids=None, limit=50)

        assert result.entries == []
        assert result.total == 0

    @patch("lex_db.database.get_db_connection")
    def test_simple_search(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test simple FTS search."""
        # Setup mock
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        count_result = {"count": 1}
        search_rows = [
            {
                "id": 1,
                "xhtml_md": "Test content about Denmark",
                "rank": 0.95,
                "permalink": "denmark",
                "headword": "Denmark",
                "encyclopedia_id": 1,
            }
        ]

        mock_db_connection.execute.return_value.fetchone.return_value = count_result
        mock_db_connection.execute.return_value.fetchall.return_value = search_rows

        # Execute
        result = search_lex_fts("Denmark", limit=50)

        # Verify
        assert result.total == 1
        assert len(result.entries) == 1
        assert result.entries[0].id == 1
        assert result.entries[0].rank == 0.95
        assert result.entries[0].title == "Denmark"

    @patch("lex_db.database.get_db_connection")
    def test_search_with_id_filter(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test FTS search with ID filtering."""
        # Setup mock
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        count_result = {"count": 1}
        search_rows = [
            {
                "id": 1,
                "xhtml_md": "Content",
                "rank": 0.8,
                "permalink": "test",
                "headword": "Test",
                "encyclopedia_id": 1,
            }
        ]

        mock_db_connection.execute.return_value.fetchone.return_value = count_result
        mock_db_connection.execute.return_value.fetchall.return_value = search_rows

        # Execute
        result = search_lex_fts("test", ids=[1, 2, 3], limit=50)

        # Verify execute was called (ID filter should be in query)
        assert mock_db_connection.execute.called
        assert result.total == 1

    @patch("lex_db.database.get_db_connection")
    def test_no_results(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test search with no matching results."""
        # Setup mock
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        count_result = {"count": 0}
        mock_db_connection.execute.return_value.fetchone.return_value = count_result
        mock_db_connection.execute.return_value.fetchall.return_value = []

        # Execute
        result = search_lex_fts("nonexistent", limit=50)

        # Verify
        assert result.total == 0
        assert result.entries == []


class TestFetchArticleTimestamps:
    """Tests for fetch_article_timestamps function."""

    @patch("lex_db.database.get_db_connection")
    def test_empty_database(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test empty database returns empty dict."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.execute.return_value.fetchall.return_value = []

        from lex_db.database import fetch_article_timestamps

        result = fetch_article_timestamps()

        assert result == {}
        assert mock_db_connection.execute.call_count == 1

    @patch("lex_db.database.get_db_connection")
    def test_multiple_articles(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test fetching timestamps for multiple articles."""
        from datetime import datetime, timezone

        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        timestamp1 = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2025, 1, 16, 14, 45, 0, tzinfo=timezone.utc)

        mock_db_connection.execute.return_value.fetchall.return_value = [
            {"id": 1, "changed_at": timestamp1},
            {"id": 2, "changed_at": timestamp2},
            {"id": 3, "changed_at": None},
        ]

        from lex_db.database import fetch_article_timestamps

        result = fetch_article_timestamps()

        assert len(result) == 3
        assert result[1] == timestamp1
        assert result[2] == timestamp2
        assert result[3] == datetime.fromtimestamp(0, tz=timezone.utc)


class TestUpsertArticle:
    """Tests for upsert_article function."""

    @patch("lex_db.database.convert_article_json_to_markdown")
    @patch("lex_db.database.derive_permalink")
    @patch("lex_db.database.derive_encyclopedia_id")
    @patch("lex_db.database.get_db_connection")
    def test_insert_new_article(
        self,
        mock_get_conn: MagicMock,
        mock_derive_enc_id: MagicMock,
        mock_derive_permalink: MagicMock,
        mock_convert_md: MagicMock,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test inserting new article."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.transaction.return_value.__enter__.return_value = None
        mock_db_connection.transaction.return_value.__exit__.return_value = None

        mock_derive_enc_id.return_value = 15
        mock_derive_permalink.return_value = "test-article"
        mock_convert_md.return_value = "# Test\n\nContent"

        article_data = {
            "id": 12345,
            "title": "Test Article",
            "url": "https://lex.dk/test-article",
            "xhtml_body": "<h1>Test</h1><p>Content</p>",
            "changed_at": "2025-01-15T10:30:00Z",
            "created_at": "2025-01-01T00:00:00Z",
        }

        from lex_db.database import upsert_article

        result = upsert_article(article_data)

        assert result is True
        assert mock_db_connection.execute.call_count == 1

        # Getting the arguments to the sql execute call
        call_args = mock_db_connection.execute.call_args
        assert call_args[0][1][0] == 12345
        assert call_args[0][1][1] == "Test Article"
        assert call_args[0][1][2] == "<h1>Test</h1><p>Content</p>"
        assert call_args[0][1][3] == "# Test\n\nContent"
        assert call_args[0][1][4] == "test-article"
        assert call_args[0][1][5] == 15
        assert call_args[0][1][6] == datetime.fromisoformat("2025-01-15T10:30:00Z")
        assert call_args[0][1][7] == datetime.fromisoformat("2025-01-01T00:00:00Z")

    @patch("lex_db.database.get_db_connection")
    def test_missing_required_fields(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test missing required fields returns False."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection

        article_data = {
            "id": 12345,
            "title": "Test",
        }

        from lex_db.database import upsert_article

        result = upsert_article(article_data)

        assert result is False
        assert mock_db_connection.execute.call_count == 0

    @patch("lex_db.database.derive_encyclopedia_id")
    @patch("lex_db.database.get_db_connection")
    def test_invalid_url(
        self,
        mock_get_conn: MagicMock,
        mock_derive_enc_id: MagicMock,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test invalid URL returns False."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_derive_enc_id.side_effect = ValueError("Unknown subdomain")

        article_data = {
            "id": 12345,
            "title": "Test",
            "url": "https://invalid.example.com/test",
            "xhtml_body": "<p>Content</p>",
        }

        from lex_db.database import upsert_article

        result = upsert_article(article_data)

        assert result is False

class TestDeleteArticles:
    """Tests for delete_articles function."""

    @patch("lex_db.database.get_db_connection")
    def test_empty_list(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test empty list returns 0."""
        from lex_db.database import delete_articles

        result = delete_articles([])

        assert result == 0
        assert mock_get_conn.call_count == 0

    @patch("lex_db.database.get_db_connection")
    def test_delete_single_article(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test deleting single article."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.transaction.return_value.__enter__.return_value = None
        mock_db_connection.transaction.return_value.__exit__.return_value = None

        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_db_connection.execute.return_value = mock_cursor

        from lex_db.database import delete_articles

        result = delete_articles([12345])

        assert result == 1
        assert mock_db_connection.execute.call_count == 1

        call_args = mock_db_connection.execute.call_args
        assert "ANY" in call_args[0][0]
        assert call_args[0][1] == [[12345]]

    @patch("lex_db.database.get_db_connection")
    def test_delete_multiple_articles(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test deleting multiple articles."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.transaction.return_value.__enter__.return_value = None
        mock_db_connection.transaction.return_value.__exit__.return_value = None

        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_db_connection.execute.return_value = mock_cursor

        from lex_db.database import delete_articles

        result = delete_articles([1, 2, 3])

        assert result == 3

    @patch("lex_db.database.get_db_connection")
    def test_delete_nonexistent_articles(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test deleting non-existent articles returns 0."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.transaction.return_value.__enter__.return_value = None
        mock_db_connection.transaction.return_value.__exit__.return_value = None

        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_db_connection.execute.return_value = mock_cursor

        from lex_db.database import delete_articles

        result = delete_articles([99999])

        assert result == 0

    @patch("lex_db.database.get_db_connection")
    def test_database_error(
        self, mock_get_conn: MagicMock, mock_db_connection: MagicMock
    ) -> None:
        """Test database error returns 0."""
        mock_get_conn.return_value.__enter__.return_value = mock_db_connection
        mock_db_connection.transaction.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        from lex_db.database import delete_articles

        result = delete_articles([1, 2, 3])

        assert result == 0
