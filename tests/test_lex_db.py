import struct
import pytest
import sqlite3
import json
from pathlib import Path
from typing import Generator

from lex_db.config import Settings, get_settings
from lex_db.database import (
    create_vector_index,
    add_single_article_to_vector_index,
    remove_article_from_vector_index,
    search_vector_index,
    get_db_connection
)
from lex_db.embeddings import EmbeddingModel, get_embedding_dimensions
from lex_db.utils import split_document_into_chunks


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Create a temporary database for testing."""
    return str(tmp_path / "test_lex.db")

@pytest.fixture
def db_conn(db_path: str, monkeypatch) -> Generator[sqlite3.Connection, None, None]:
    """Create a temporary database and connection, ensuring Settings uses a test .env file."""
    
    test_env_file_path = Path(db_path).parent / ".env.test"
    env_content = f"DATABASE_URL=sqlite:///{db_path}\nDEBUG=True\n" # Ensure DATABASE_URL is a valid URI
    with open(test_env_file_path, "w") as f:
        f.write(env_content)

    # Get the current model_config from the Settings class
    original_model_config = Settings.model_config
    
    # Create a new config dict for testing, pointing to the test .env file
    test_model_config = original_model_config.copy() 
    test_model_config['env_file'] = str(test_env_file_path)
    
    # Use monkeypatch to temporarily replace Settings.model_config
    monkeypatch.setattr(Settings, "model_config", test_model_config)
    
    # Clear the lru_cache on get_settings so it re-evaluates with the new Settings.model_config
    if hasattr(get_settings, 'cache_clear'):
        get_settings.cache_clear()
    
    # Ensure the db file itself exists
    Path(db_path).touch()

    try:
        # Now, get_db_connection will use the settings from .env.test
        with get_db_connection() as conn:
            # Create a dummy source table
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS articles;") # Add this line
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
                ("Test Article 1", "This is the first test article. It has some content.", 1678886400.0)
            )
            cursor.execute(
                "INSERT INTO articles (title, content, updated_at) VALUES (?, ?, ?)",
                ("Test Article 2", "Second article for testing purposes. More content here.", 1678886500.0)
            )
            conn.commit()
            yield conn
    finally:
        # monkeypatch will automatically restore Settings.model_config to its original value.
        # Clear cache again in case other tests need a fresh start or if get_settings is called during teardown.
        if hasattr(get_settings, 'cache_clear'):
            get_settings.cache_clear()
        
        if test_env_file_path.exists():
            test_env_file_path.unlink() # Clean up dummy .env



def test_create_vector_index_mock(db_conn: sqlite3.Connection):
    source_table = "articles"
    text_column = "content"
    index_name = "test_index_mock"
    model_choice = EmbeddingModel.MOCK_MODEL.value

    create_vector_index(db_conn, source_table, text_column, index_name, model_choice, chunk_size=20, chunk_overlap=5)

    cursor = db_conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {index_name}")
    count = cursor.fetchone()[0]
    assert count > 0  # Check that some embeddings were inserted

    cursor.execute(f"SELECT embedding, source_article_id, chunk_sequence_id FROM {index_name} LIMIT 1")
    row = cursor.fetchone()
    assert row is not None
    embedding_bytes = row["embedding"]
    num_dimensions = get_embedding_dimensions(model_choice)
    # sqlite-vec stores embeddings as BLOBs of 4-byte little-endian floats.
    # Unpack the bytes into a list of floats.
    embedding_data = list(struct.unpack(f'<{num_dimensions}f', embedding_bytes))
    assert len(embedding_data) == num_dimensions # Ensure the length matches dimensions
    assert row["source_article_id"] is not None


def test_add_and_remove_article_mock(db_conn: sqlite3.Connection):
    index_name = "article_maintenance_index"
    model_choice = EmbeddingModel.MOCK_MODEL.value
    # Create an empty index first
    embedding_dim = get_embedding_dimensions(model_choice)
    db_conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {index_name} USING vec0(
            embedding FLOAT[{embedding_dim}], source_article_id TEXT, chunk_sequence_id INTEGER
        );
    """)
    db_conn.commit()

    article_rowid = "test_article_123"
    article_text = "A new article to be added and then removed."

    add_single_article_to_vector_index(db_conn, index_name, article_rowid, article_text, model_choice, chunk_size=10, chunk_overlap=2)
    
    cursor = db_conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {index_name} WHERE source_article_id = ?", (article_rowid,))
    count_after_add = cursor.fetchone()[0]
    assert count_after_add > 0

    remove_article_from_vector_index(db_conn, index_name, article_rowid)
    cursor.execute(f"SELECT COUNT(*) FROM {index_name} WHERE source_article_id = ?", (article_rowid,))
    count_after_remove = cursor.fetchone()[0]
    assert count_after_remove == 0


def test_search_vector_index_mock(db_conn: sqlite3.Connection):
    source_table = "articles"
    text_column = "content"
    index_name = "search_index_mock"
    model_choice = EmbeddingModel.MOCK_MODEL.value

    create_vector_index(db_conn, source_table, text_column, index_name, model_choice, chunk_size=20, chunk_overlap=5)

    query_text = "test query"
    top_k = 2
    results = search_vector_index(db_conn, index_name, query_text, model_choice, top_k)

    assert len(results) <= top_k
    if results:
        for item in results:
            assert "id_in_index" in item
            assert "source_article_id" in item
            assert "chunk_seq" in item
            assert "distance" in item # sqlite-vec provides this