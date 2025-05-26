import pytest
import sqlite3
import json
from pathlib import Path
from typing import Generator

import sqlite_vec

from src.lex_db.config import Settings
from src.lex_db.vector_store import (
    create_vector_index,
    add_single_article_to_vector_index,
    remove_article_from_vector_index,
    search_vector_index,
)
from src.lex_db.embeddings import EmbeddingModel, get_embedding_dimensions


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Create a temporary database for testing."""
    return str(tmp_path / "test_lex.db")


@pytest.fixture
def db_conn(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> Generator[sqlite3.Connection, None, None]:
    """Create a temporary database and connection, ensuring Settings uses a test .env file."""

    test_env_file_path = Path(db_path).parent / ".env.test"
    env_content = f"DATABASE_URL=sqlite:///{db_path}\nDEBUG=True\n"
    with open(test_env_file_path, "w") as f:
        f.write(env_content)

    test_settings = Settings(DATABASE_URL=Path(db_path), DEBUG=True)

    monkeypatch.setattr("src.lex_db.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("src.lex_db.database.get_settings", lambda: test_settings)
    monkeypatch.setattr("src.lex_db.database.get_db_path", lambda: db_path)

    # Ensure the db file exists
    Path(db_path).touch()

    # Create a direct connection to the test database to ensure we're using the right one
    test_conn = sqlite3.connect(db_path)

    # Enable loading extensions and load sqlite_vec, mirroring database.py
    test_conn.enable_load_extension(True)
    sqlite_vec.load(test_conn)

    # Set row factory as in the real connection
    test_conn.row_factory = sqlite3.Row

    # Patch get_db_connection to return our test connection
    def mock_get_db_connection() -> sqlite3.Connection:
        return test_conn

    monkeypatch.setattr("src.lex_db.database.get_db_connection", mock_get_db_connection)
    # Also patch the contextmanager directly
    monkeypatch.setattr("src.lex_db.database.create_connection", lambda: test_conn)

    try:
        # Set up test tables
        cursor = test_conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS articles;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                updated_at REAL
            )
        """)
        cursor.execute(
            "INSERT INTO articles (title, content, updated_at) VALUES (?, ?, ?)",
            (
                "Test Article 1",
                "This is the first test article. It has some content.",
                1678886400.0,
            ),
        )
        cursor.execute(
            "INSERT INTO articles (title, content, updated_at) VALUES (?, ?, ?)",
            (
                "Test Article 2",
                "Second article for testing purposes. More content here.",
                1678886500.0,
            ),
        )
        test_conn.commit()
        yield test_conn
    finally:
        test_conn.close()
        if test_env_file_path.exists():
            test_env_file_path.unlink()  # Clean up dummy .env


def test_create_vector_index_mock(db_conn: sqlite3.Connection) -> None:
    """Test creating a vector index with mock model."""
    index_name = "test_index_mock"
    model_choice = EmbeddingModel.MOCK_MODEL

    # Create the vector index
    create_vector_index(db_conn, index_name, model_choice, force=True)

    # Verify the index was created
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (index_name,)
    )
    table_exists = cursor.fetchone() is not None
    assert table_exists

    # Add some data to test the structure
    article_id = "1"
    article_text = "This is a test article for verifying the vector index structure."

    add_single_article_to_vector_index(
        db_conn,
        index_name,
        article_id,
        article_text,
        model_choice,
        chunk_size=20,
        chunk_overlap=5,
    )

    # Check the structure and content
    cursor.execute(f"SELECT * FROM {index_name} LIMIT 1")
    row = cursor.fetchone()

    assert row is not None
    assert "embedding" in row.keys()
    assert "source_article_id" in row.keys()
    assert "chunk_sequence_id" in row.keys()
    assert "chunk_text" in row.keys()
    assert "last_updated" in row.keys()

    # Verify embedding format and dimensions
    embedding_data = row["embedding"]
    # Check if it's binary or JSON format
    if isinstance(embedding_data, bytes):
        # For binary format, each float is 4 bytes
        embedding_length = len(embedding_data) // 4
        assert embedding_length == get_embedding_dimensions(model_choice)
    else:
        # If it's JSON format
        try:
            embedding = json.loads(embedding_data)
            assert len(embedding) == get_embedding_dimensions(model_choice)
        except (json.JSONDecodeError, TypeError):
            # If neither binary nor JSON, we have a problem
            assert False, f"Embedding format not recognized: {type(embedding_data)}"

    assert row["source_article_id"] == article_id


def test_add_and_remove_article_mock(db_conn: sqlite3.Connection) -> None:
    """Test adding and removing articles from a vector index."""
    index_name = "article_maintenance_index"
    model_choice = EmbeddingModel.MOCK_MODEL

    # Create a new vector index
    create_vector_index(db_conn, index_name, model_choice, force=True)

    article_rowid = "test_article_123"
    article_text = "A new article to be added and then removed."

    # Add the article to the vector index
    add_single_article_to_vector_index(
        db_conn,
        index_name,
        article_rowid,
        article_text,
        model_choice,
        chunk_size=10,
        chunk_overlap=2,
    )

    # Verify the article was added
    cursor = db_conn.cursor()
    cursor.execute(
        f"SELECT COUNT(*) FROM {index_name} WHERE source_article_id = ?",
        (article_rowid,),
    )
    count_after_add = cursor.fetchone()[0]
    assert count_after_add > 0

    # Remove the article and verify it's gone
    remove_article_from_vector_index(db_conn, index_name, article_rowid)
    cursor.execute(
        f"SELECT COUNT(*) FROM {index_name} WHERE source_article_id = ?",
        (article_rowid,),
    )
    count_after_remove = cursor.fetchone()[0]
    assert count_after_remove == 0


def test_search_vector_index_mock(db_conn: sqlite3.Connection) -> None:
    """Test searching a vector index."""
    index_name = "search_index_mock"
    model_choice = EmbeddingModel.MOCK_MODEL

    # Create the vector index
    create_vector_index(db_conn, index_name, model_choice, force=True)

    # Add some articles to search
    article1_id = "1"
    article1_text = "This is the first test article for vector search."

    article2_id = "2"
    article2_text = "This is the second article with different content."

    article3_id = "3"
    article3_text = "This is the third article with completely different content."

    # Add articles to index
    add_single_article_to_vector_index(
        db_conn,
        index_name,
        article1_id,
        article1_text,
        model_choice,
        chunk_size=20,
        chunk_overlap=5,
    )

    add_single_article_to_vector_index(
        db_conn,
        index_name,
        article2_id,
        article2_text,
        model_choice,
        chunk_size=20,
        chunk_overlap=5,
    )

    add_single_article_to_vector_index(
        db_conn,
        index_name,
        article3_id,
        article3_text,
        model_choice,
        chunk_size=20,
        chunk_overlap=5,
    )

    # Search the vector index
    query_text = "test article"
    top_k = 2
    results = search_vector_index(db_conn, index_name, query_text, model_choice, top_k)

    assert len(results) == top_k
    if results:
        for item in results:
            assert item["source_article_id"] in [article1_id, article2_id, article3_id]
            assert "id_in_index" in item
            assert "source_article_id" in item
            assert "chunk_seq" in item
            assert "distance" in item  # sqlite-vec provides this


def test_updated_article_in_vector_index(db_conn: sqlite3.Connection) -> None:
    """Test updating an existing article in the vector index."""
    index_name = "update_test_index"
    model_choice = EmbeddingModel.MOCK_MODEL

    # Create the vector index
    create_vector_index(db_conn, index_name, model_choice, force=True)

    article_id = "update_test_123"
    original_text = "Original article text."
    updated_text = "Updated article text with completely different content."

    # Add the original article
    add_single_article_to_vector_index(
        db_conn, index_name, article_id, original_text, model_choice
    )

    # Get the original embedding
    cursor = db_conn.cursor()
    cursor.execute(
        f"SELECT embedding FROM {index_name} WHERE source_article_id = ?", (article_id,)
    )
    original_embedding = cursor.fetchone()[0]

    # Remove and re-add with new content (simulating an update)
    remove_article_from_vector_index(db_conn, index_name, article_id)

    # Use a distinctly different text to ensure different embedding
    add_single_article_to_vector_index(
        db_conn, index_name, article_id, updated_text, model_choice
    )

    # Verify the embedding changed
    cursor.execute(
        f"SELECT embedding FROM {index_name} WHERE source_article_id = ?", (article_id,)
    )
    row = cursor.fetchone()
    updated_embedding = row[0]

    if isinstance(original_embedding, bytes) and isinstance(updated_embedding, bytes):
        embedding_length = len(original_embedding) // 4  # 4 bytes per float
        expected_length = get_embedding_dimensions(model_choice)
        assert embedding_length == expected_length, (
            "Embedding should have correct dimensions"
        )
        assert updated_embedding != original_embedding, (
            "Embedding should change after update"
        )
