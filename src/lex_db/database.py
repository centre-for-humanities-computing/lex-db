"""Database connection and operations for Lex DB."""

from datetime import datetime, timezone
from pydantic import BaseModel
from psycopg import Connection
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from contextlib import contextmanager
from typing import Optional, Generator

from lex_db.config import get_settings
from lex_db.sitemap import derive_encyclopedia_id, derive_permalink
from lex_db.utils import convert_article_json_to_markdown, get_logger

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
def get_db_connection() -> Generator[Connection, None, None]:
    """Get a connection from the pool."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        yield conn


def get_db_info() -> dict:
    """Get information about the database."""
    with get_db_connection() as conn:
        # Get list of tables from PostgreSQL system catalog
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';"
            ).fetchall()
        ]

        # Get PostgreSQL version
        (pg_version,) = conn.execute("SELECT version();").fetchone() or ("Unknown",)

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
        # Use PostgreSQL's ANY() operator with array parameter
        # More efficient than IN with multiple placeholders
        count_result = conn.execute(
            "SELECT COUNT(*) as count FROM articles WHERE id = ANY(%s)",
            [ids],
        ).fetchone()
        total = count_result["count"] if count_result else 0  # type: ignore[call-overload]

        rows = conn.execute(
            """
            SELECT id, xhtml_md, 0.0 as rank, permalink, headword, encyclopedia_id
            FROM articles
            WHERE id = ANY(%s)
            LIMIT %s
            """,
            [ids, limit],
        ).fetchall()  # type: ignore[misc]

        entries = [
            SearchResult(
                id=row["id"],  # type: ignore[call-overload]
                xhtml_md=row["xhtml_md"],  # type: ignore[call-overload]
                rank=row["rank"],  # type: ignore[call-overload]
                url=get_url_base(int(row["encyclopedia_id"])) + row["permalink"],  # type: ignore[call-overload]
                title=row["headword"],  # type: ignore[call-overload]
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
    Perform full-text search on lex entries using PostgreSQL native FTS.

    Args:
        query: Search query string (natural language, questions, or keywords)
        ids: Optional list of article IDs to restrict search to
        limit: Maximum number of results to return

    Returns:
        SearchResults with ranked entries
    """
    # Handle empty query
    if not query or not query.strip():
        if ids:
            return get_articles_by_ids(ids, limit=limit)
        return SearchResults(entries=[], total=0, limit=limit)

    with get_db_connection() as conn:
        # Build the WHERE clause
        # Use plainto_tsquery for natural language queries (better for RAG)
        # Automatically filters stop words and handles conversational queries
        where_clause = "xhtml_md_tsv @@ plainto_tsquery('danish', %s)"
        params: list = [query]

        # Add ID filter if provided
        if ids:
            where_clause += " AND a.id = ANY(%s)"
            params.append(ids)

        # Count total matching results
        count_result = conn.execute(
            f"SELECT COUNT(*) as count FROM articles a WHERE {where_clause}", params
        ).fetchone()
        total = count_result["count"] if count_result else 0  # type: ignore[call-overload]

        # Get ranked results
        # ts_rank() scores results by relevance (higher = more relevant)
        rows = conn.execute(
            f"""
            SELECT 
                a.id,
                a.xhtml_md,
                ts_rank(a.xhtml_md_tsv, plainto_tsquery('danish', %s)) as rank,
                a.permalink,
                a.headword,
                a.encyclopedia_id
            FROM articles a
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT %s
            """,
            [query] + params + [limit],
        ).fetchall()  # type: ignore[misc]

        # Format results
        entries = [
            SearchResult(
                id=row["id"],  # type: ignore[call-overload]
                xhtml_md=row["xhtml_md"],  # type: ignore[call-overload]
                rank=float(row["rank"]),  # type: ignore[call-overload]
                url=get_url_base(int(row["encyclopedia_id"])) + row["permalink"],  # type: ignore[call-overload]
                title=row["headword"],  # type: ignore[call-overload]
            )
            for row in rows
        ]

        return SearchResults(entries=entries, total=total, limit=limit)


def fetch_article_timestamps() -> dict[int, datetime]:
    """Fetch all article IDs and their last modified timestamps."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, changed_at FROM articles").fetchall()

        return {
            row["id"]: row["changed_at"] or datetime.fromtimestamp(0, tz=timezone.utc)  # type: ignore[call-overload]
            for row in rows
        }


def upsert_article(article_data: dict) -> bool:
    """Insert or update article with field mapping from lex.dk JSON API."""
    try:
        required_fields = ["id", "title", "url", "xhtml_body"]
        missing = [f for f in required_fields if f not in article_data]
        if missing:
            logger.error(f"Missing required fields: {', '.join(missing)}")
            return False

        article_id = article_data["id"]
        headword = article_data["title"]
        url = article_data["url"]
        xhtml_body = article_data["xhtml_body"]

        encyclopedia_id = derive_encyclopedia_id(url)
        permalink = derive_permalink(url)
        xhtml_md = convert_article_json_to_markdown(article_data)

        changed_at = None
        if "changed_at" in article_data and article_data["changed_at"]:
            changed_at_str = article_data["changed_at"]
            changed_at = datetime.fromisoformat(changed_at_str.replace("Z", "+00:00"))

        created_at = datetime.fromisoformat(
            article_data.get("created_at", "1970-01-01T00:00:00Z").replace(
                "Z", "+00:00"
            )
        )

        with get_db_connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO articles (
                        id, headword, xhtml, xhtml_md, permalink, 
                        encyclopedia_id, changed_at, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        headword = EXCLUDED.headword,
                        xhtml = EXCLUDED.xhtml,
                        xhtml_md = EXCLUDED.xhtml_md,
                        permalink = EXCLUDED.permalink,
                        encyclopedia_id = EXCLUDED.encyclopedia_id,
                        changed_at = EXCLUDED.changed_at,
                        created_at = EXCLUDED.created_at
                    """,
                    [
                        article_id,
                        headword,
                        xhtml_body,
                        xhtml_md,
                        permalink,
                        encyclopedia_id,
                        changed_at,
                        created_at,
                    ],
                )

        logger.info(f"Upserted article {article_id}: {headword}")
        return True

    except ValueError as e:
        logger.error(f"Invalid article data: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to upsert article: {e}")
        return False


def delete_articles(article_ids: list[int]) -> int:
    """Delete articles by ID list with cascade to vector indexes."""
    if not article_ids:
        logger.debug("No article IDs provided for deletion")
        return 0

    try:
        with get_db_connection() as conn:
            with conn.transaction():
                cursor = conn.execute(
                    "DELETE FROM articles WHERE id = ANY(%s)",
                    [article_ids],
                )
                deleted_count = cursor.rowcount

        logger.info(
            f"Deleted {deleted_count} articles (requested {len(article_ids)} IDs)"
        )
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete articles: {e}")
        return 0
