"""Configure pytest to recognize src directory."""

from unittest.mock import MagicMock
from datetime import datetime
import pytest

from lex_db.embeddings import EmbeddingModel
from lex_db.database import SearchResult, SearchResults


# Database Fixtures


@pytest.fixture
def mock_db_connection():
    """Mock PostgreSQL connection for testing."""
    conn = MagicMock()
    conn.execute = MagicMock()
    conn.commit = MagicMock()
    conn.rollback = MagicMock()

    # Mock cursor context manager
    cursor = MagicMock()
    cursor.execute = MagicMock()
    cursor.executemany = MagicMock()
    cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=[])
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn.cursor = MagicMock(return_value=cursor)
    return conn


@pytest.fixture
def mock_db_result():
    """Mock database query result."""
    result = MagicMock()
    result.fetchone = MagicMock(return_value=None)
    result.fetchall = MagicMock(return_value=[])
    return result


# Embedding Fixtures


@pytest.fixture
def mock_embedding_model():
    """Use MOCK_MODEL for testing."""
    return EmbeddingModel.MOCK_MODEL


@pytest.fixture
def sample_texts():
    """Sample texts for embedding tests."""
    return [
        "Dette er en test.",
        "Her er mere tekst.",
        "Endnu en sætning.",
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing (dimension 4 for MOCK_MODEL)."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]


# Vector Store Fixtures


@pytest.fixture
def sample_chunks_data():
    """Sample chunks data for vector index."""
    return [
        ("1", "0", "First chunk text"),
        ("1", "1", "Second chunk text"),
        ("2", "0", "Another article chunk"),
    ]


@pytest.fixture
def sample_embeddings_data():
    """Sample pre-computed embeddings data."""
    return [
        ("1", "0", "Chunk text", [0.1, 0.2, 0.3, 0.4]),
        ("1", "1", "More text", [0.2, 0.3, 0.4, 0.5]),
        ("2", "0", "Another chunk", [0.3, 0.4, 0.5, 0.6]),
    ]


@pytest.fixture
def vector_index_metadata():
    """Sample vector index metadata."""
    return {
        "index_name": "test_index",
        "source_table": "articles",
        "source_column": "xhtml_md",
        "embedding_model": "mock_model",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "chunking_strategy": "sections",
        "updated_at_column": "changed_at",
    }


# Database Data Fixtures


@pytest.fixture
def sample_articles():
    """Sample article data."""
    return [
        {
            "id": 1,
            "headword": "Test Article",
            "xhtml_md": "# Test\n\nThis is test content.",
            "changed_at": datetime(2025, 1, 1),
            "encyclopedia_id": 1,
            "permalink": "test-article",
        },
        {
            "id": 2,
            "headword": "Another Article",
            "xhtml_md": "# Another\n\nMore content here.",
            "changed_at": datetime(2025, 1, 2),
            "encyclopedia_id": 2,
            "permalink": "another-article",
        },
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results."""
    return SearchResults(
        entries=[
            SearchResult(
                id=1,
                xhtml_md="Content",
                rank=0.95,
                url="https://denstoredanske.lex.dk/test",
                title="Test",
            )
        ],
        total=1,
        limit=50,
    )
