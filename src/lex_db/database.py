"""Database connection and operations for Lex DB."""

from pathlib import Path
import re
import sqlite3
from pydantic import BaseModel
import sqlite_vec
from contextlib import contextmanager
from typing import Generator, Optional

from lex_db.config import get_settings
from lex_db.utils import get_logger

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
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)

        # Load the sqlite-vec extension
        try:
            sqlite_vec.load(conn)
        except sqlite3.Error as e:
            conn.close()
            raise sqlite3.Error(f"Failed to load sqlite-vec extension: {e}")
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
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
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


class SearchResult(BaseModel):
    """Single result from a search."""

    id: int
    xhtml_md: str
    rank: float
    url: Optional[str] = None
    title: str
    headword: Optional[str] = None


class SearchResults(BaseModel):
    """Results of a search."""

    entries: list[SearchResult]
    total: int
    limit: int


def get_url_base(encyclopedia_id: int) -> str:
    match encyclopedia_id:
        case 1:
            return "https://denstoredanske.lex.dk/"
        case 2:
            return "https://trap.lex.dk/"
        case 3:
            return "https://biografiskleksikon.lex.dk/"
        case 4:
            return "https://gyldendalogpolitikensdanmarkshistorie.lex.dk/"
        case 5:
            return "https://danmarksoldtid.lex.dk/"
        case 6:
            return "https://teaterleksikon.lex.dk/"
        case 7:
            return "https://mytologi.lex.dk/"
        case 8:
            return "https://pattedyratlas.lex.dk/"
        case 9:
            return "https://dansklitteraturshistorie.lex.dk/"
        case 10:
            return "https://bornelitteratur.lex.dk/"
        case 11:
            return "https://symbolleksikon.lex.dk/"
        case 12:
            return "https://naturenidanmark.lex.dk/"
        case 14:
            return "https://om.lex.dk/"
        case 15:
            return "https://lex.dk/"
        case 16:
            return "https://kvindebiografiskleksikon.lex.dk/"
        case 17:
            return "https://medicin.lex.dk/"
        case 18:
            return "https://trap-groenland.lex.dk/"
        case 19:
            return "https://trap-faeroeerne.lex.dk/"
        case 20:
            return "https://danmarkshistorien.lex.dk/"
        case _:
            logger.warning(
                f"Encyclopedia id {encyclopedia_id} used, but there is no valid URL base for that ID."
            )
            return "https://lex.dk/"


def get_articles_by_ids(ids: list[int], limit: int = 50) -> SearchResults:
    """
    Fetch articles by a list of IDs.
    """
    if not ids:
        return SearchResults(entries=[], total=0, limit=limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"SELECT COUNT(*) FROM articles WHERE rowid IN ({placeholders})",
            ids,
        )
        total = cursor.fetchone()[0]
        cursor.execute(
            f"""
            SELECT rowid, xhtml_md, 0.0 as rank, permalink, headword, encyclopedia_id
            FROM articles
            WHERE rowid IN ({placeholders})
            LIMIT ?
            """,
            (*ids, limit),
        )
        entries = [
            SearchResult(
                id=row[0],
                xhtml_md=row[1],
                rank=row[2],
                url=get_url_base(int(row[5])) + row[3],
                title=row[4],
                headword=row[3],
            )
            for row in cursor.fetchall()
        ]
        return SearchResults(
            entries=entries,
            total=total,
            limit=limit,
        )


def batch_search_lex_fts(queries: list[str], limit: int = 50) -> list[SearchResults]:
    """
    Perform batch full-text search on lex entries using FTS5.
    """
    results = []
    for query in queries:
        result = search_lex_fts(query=query, limit=limit)
        results.append(result)
    return results


def search_lex_fts(
    query: str, ids: Optional[list[int]] = None, limit: int = 50
) -> SearchResults:
    """
    Perform full-text search on lex entries using FTS5.
    Optionally restrict search to a set of IDs.
    """
    if not query or not query.strip():
        if ids:
            return get_articles_by_ids(ids, limit=limit)
        return SearchResults(entries=[], total=0, limit=limit)

    # Keep Danish characters (æ, ø, å) and basic punctuation
    sanitized_query = re.sub(r'["\-:*^()\[\]{}|+&]', " ", query.strip())
    sanitized_query = re.sub(r"\s+", " ", sanitized_query).strip()

    if not sanitized_query:
        return SearchResults(entries=[], total=0, limit=limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Build WHERE clause for ids if provided
        where_clause = "fts_articles MATCH ?"
        params = [sanitized_query]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            where_clause += f" AND f.rowid IN ({placeholders})"
            params.extend([str(id) for id in ids])
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM fts_articles f
            WHERE {where_clause}
            """,
            params,
        )
        total = cursor.fetchone()[0]
        cursor.execute(
            f"""
            SELECT 
                f.rowid,
                f.xhtml_md,
                bm25(fts_articles) as rank,
                a.permalink,
                a.headword,
                a.encyclopedia_id
            FROM fts_articles f
            JOIN articles a ON a.id = f.rowid
            WHERE {where_clause}
            ORDER BY rank
            LIMIT ?
            """,
            (*params, limit),
        )
        entries = [
            SearchResult(
                id=row[0],
                xhtml_md=row[1],
                rank=row[2],
                url=get_url_base(int(row[5])) + row[3],
                title=row[4],
                headword=row[4],
            )
            for row in cursor.fetchall()
        ]
        return SearchResults(
            entries=entries,
            total=total,
            limit=limit,
        )
