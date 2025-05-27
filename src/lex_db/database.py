"""Database connection and operations for Lex DB."""

from pathlib import Path
import re
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from typing import Generator

from src.lex_db.config import get_settings
from src.lex_db.utils import get_logger

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


def search_lex_fts(
    query: str, limit: int = 50, offset: int = 0
) -> dict[str, object]:
    """
    Perform full-text search on lex entries using FTS5.
    """
    if not query or not query.strip():
        return {"entries": [], "total": 0, "query": query}

    # Keep Danish characters (æ, ø, å) and basic punctuation
    sanitized_query = re.sub(r'["\-:*^()\[\]{}|+&]', ' ', query.strip())
    # Remove multiple spaces and strip
    sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Count total results
        cursor.execute(
            """
            SELECT COUNT(*) FROM fts_articles
            WHERE fts_articles MATCH ?
        """,
            (sanitized_query,),
        )
        total = cursor.fetchone()[0]

        # Get paginated results with ranking
        cursor.execute(
            """
            SELECT 
                rowid,
                xhtml_md,
                bm25(fts_articles) as rank
            FROM fts_articles
            WHERE fts_articles MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
        """,
            (sanitized_query, limit, offset),
        )

        entries = []
        for row in cursor.fetchall():
            entries.append(
                {
                    "id": row[0],
                    "xhtml_md": row[1],
                    "rank": row[2],
                }
            )

        return {
            "entries": entries,
            "total": total,
            "query": query,
            "limit": limit,
            "offset": offset,
        }
