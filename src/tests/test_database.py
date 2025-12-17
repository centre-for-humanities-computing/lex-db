"""Unit tests for database module."""

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
