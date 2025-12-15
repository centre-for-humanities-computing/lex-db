"""Database connection and operations for Lex DB."""

import re
import psycopg
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from contextlib import contextmanager
from typing import Generator, Optional

from lex_db.config import get_settings
from lex_db.utils import get_logger

logger = get_logger()

# Global connection pool
_connection_pool: ConnectionPool | None = None

def get_connection_pool() -> ConnectionPool:
    """Get or create connection pool."""
    global _connection_pool
    if _connection_pool is None:
        settings = get_settings()
        _connection_pool = ConnectionPool(
            settings.DATABASE_URL,
            min_size=settings.DB_POOL_MIN_SIZE,
            max_size=settings.DB_POOL_MAX_SIZE,
            kwargs={"row_factory": dict_row},
        )
    return _connection_pool

@contextmanager
def get_db_connection():
    """Get a connection from the pool."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        yield conn


def get_db_info() -> dict:
    """Get information about the database."""
    with get_db_connection() as conn:
        # Get list of tables from PostgreSQL system catalog
        tables = [
            row[0] for row in conn.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';"
            ).fetchall()
        ]
        
        # Get PostgreSQL version
        pg_version, = conn.execute("SELECT version();").fetchone() or ("Unknown",)

        return {
            "tables": tables,
            "postgres_version": pg_version,
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
        placeholders = ",".join("%s" for _ in ids)
        
        total, = conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE id IN ({placeholders})",
            ids,
        ).fetchone() or (0,)
        
        rows = conn.execute(
            f"""
            SELECT id, xhtml_md, 0.0 as rank, permalink, headword, encyclopedia_id
            FROM articles
            WHERE id IN ({placeholders})
            LIMIT %s
            """,
            (*ids, limit),
        ).fetchall()
        
        entries = [
            SearchResult(
                id=row[0],
                xhtml_md=row[1],
                rank=row[2],
                url=get_url_base(int(row[5])) + row[3],
                title=row[4],
                headword=row[3],
            )
            for row in rows
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
    Perform full-text search on lex entries.
    
    NOTE: Full-text search migration to PostgreSQL is handled in Sub-Issue 4.
    This function is temporarily disabled.
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
    raise NotImplementedError(
        "Full-text search is not yet implemented for PostgreSQL. "
        "This will be migrated in Sub-Issue 4 of the PostgreSQL migration."
    )
