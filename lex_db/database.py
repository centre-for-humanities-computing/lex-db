"""Database connection and operations for Lex DB."""

import json
from pathlib import Path
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from typing import Generator

from lex_db.config import get_settings
from lex_db.utils import get_logger, split_document_into_chunks
from lex_db.embeddings import (
    EmbeddingModel,
    generate_embeddings,
    get_embedding_dimensions,
    generate_query_embedding,
)

logger = get_logger()


def get_db_path() -> Path:
    """Get the path to the SQLite database file."""
    settings = get_settings()
    return settings.DATABASE_URL


def verify_db_exists() -> bool:
    """Verify that the database file exists."""
    db_path = get_db_path()
    return db_path.exists()


def create_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    db_path = get_db_path()

    if not verify_db_exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # Enable loading extensions
        conn.enable_load_extension(True)

        # Load the sqlite-vec extension
        try:
            sqlite_vec.load(conn)
        except sqlite3.Error as e:
            conn.close()
            raise sqlite3.Error(f"Failed to load sqlite-vec extension: {e}")

        # Configure connection
        conn.row_factory = sqlite3.Row

        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error connecting to database: {e}")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a connection to the SQLite database."""
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_db_info() -> dict:
    """Get information about the database."""
    with get_db_connection() as conn:
        # Get the list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Get the SQLite version
        cursor.execute("SELECT sqlite_version();")
        sqlite_version = cursor.fetchone()[0]

        # Check if sqlite-vec is loaded
        try:
            cursor.execute("SELECT vec_version();")
            vector_version = cursor.fetchone()[0]
        except sqlite3.Error:
            vector_version = "Not loaded"

        return {
            "path": get_db_path(),
            "tables": tables,
            "sqlite_version": sqlite_version,
            "vector_version": vector_version,
        }


def create_vector_index(
    db_conn: sqlite3.Connection,
    source_table_name: str,
    source_text_column: str,
    vector_index_name: str,
    embedding_model_choice: EmbeddingModel,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """Create a new vector index from a source table's text column."""
    cursor = db_conn.cursor()
    embedding_dim = get_embedding_dimensions(embedding_model_choice)

    # Create the sqlite-vec virtual table
    create_table_sql = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {vector_index_name} USING vec0(
        embedding FLOAT[{embedding_dim}],
        source_article_id TEXT,
        chunk_sequence_id INTEGER,
        chunk_text TEXT,
    );
    """
    cursor.execute(create_table_sql)
    logger.info(f"Virtual table {vector_index_name} created.")

    # Fetch articles from the source table
    cursor.execute(f"SELECT rowid, {source_text_column} FROM {source_table_name}")
    articles = cursor.fetchall()
    total_articles = len(articles)
    logger.info(f"Found {total_articles} articles to process")

    # Process each article: chunk, embed, and insert
    for idx, (article_rowid, text_content) in enumerate(articles, 1):
        if not text_content:
            continue

        logger.info(f"Processing article {idx}/{total_articles} (ID: {article_rowid})")

        chunks = split_document_into_chunks(
            text_content, chunk_size=chunk_size, overlap=chunk_overlap
        )
        if not chunks:
            continue

        logger.info(f"  Generated {len(chunks)} chunks")
        chunk_embeddings = generate_embeddings(
            chunks, model_choice=embedding_model_choice
        )

        for i, (chunk_text, chunk_embedding) in enumerate(
            zip(chunks, chunk_embeddings)
        ):
            # Insert the embedding with metadata
            insert_sql = f"""
            INSERT INTO {vector_index_name} (embedding, source_article_id, chunk_sequence_id, chunk_text)
            VALUES (?, ?, ?, ?);
            """
            # Convert embedding to JSON string for sqlite-vec
            cursor.execute(
                insert_sql,
                (json.dumps(chunk_embedding), str(article_rowid), i, chunk_text),
            )

        # Commit after each article to avoid losing work if there's an error
        db_conn.commit()

    logger.info(f"Finished populating {vector_index_name} from {source_table_name}.")


def add_single_article_to_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    article_rowid: str,
    article_text: str,
    embedding_model_choice: EmbeddingModel,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """Add a single article to an existing vector index."""
    if not article_text:
        return

    cursor = db_conn.cursor()
    chunks = split_document_into_chunks(
        article_text, chunk_size=chunk_size, overlap=chunk_overlap
    )

    if not chunks:
        return

    chunk_embeddings = generate_embeddings(chunks, model_choice=embedding_model_choice)

    for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
        insert_sql = f"""
        INSERT INTO {vector_index_name} (embedding, source_article_id, chunk_sequence_id, chunk_text)
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(
            insert_sql, (json.dumps(chunk_embedding), article_rowid, i, chunk_text)
        )

    db_conn.commit()
    logger.info(f"Added article {article_rowid} to index {vector_index_name}.")


def remove_article_from_vector_index(
    db_conn: sqlite3.Connection, vector_index_name: str, article_rowid: str
) -> None:
    """Remove an article from a vector index."""
    cursor = db_conn.cursor()
    delete_sql = f"DELETE FROM {vector_index_name} WHERE source_article_id = ?;"
    cursor.execute(delete_sql, (article_rowid,))
    db_conn.commit()
    logger.info(f"Removed article {article_rowid} from index {vector_index_name}.")


def search_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    query_text: str,
    embedding_model_choice: EmbeddingModel,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Search a vector index for similar content to the query text."""
    cursor = db_conn.cursor()

    # Generate embedding for the query text
    query_vector = generate_query_embedding(
        query_text, model_choice=embedding_model_choice
    )
    query_vector_json = json.dumps(query_vector)

    search_sql = f"""
    SELECT rowid, source_article_id, chunk_sequence_id, chunk_text, distance
    FROM {vector_index_name}
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT ?;
    """

    cursor.execute(search_sql, (query_vector_json, top_k))
    results = cursor.fetchall()

    # Format results
    formatted_results = [
        {
            "id_in_index": res[0],
            "source_article_id": res[1],
            "chunk_seq": res[2],
            "chunk_text": res[3],
            "distance": res[4],
        }
        for res in results
    ]

    return formatted_results
