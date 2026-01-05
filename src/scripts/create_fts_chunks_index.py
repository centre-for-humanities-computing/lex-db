"""Script to create and populate FTS5 (Full-Text Search) index for chunks in a vec0 vector index."""

import argparse
import sqlite3
import sys

from lex_db.utils import get_logger, configure_logging
from lex_db.database import create_connection
from lex_db.config import get_settings

logger = get_logger()


def create_fts_chunks_table(conn: sqlite3.Connection, fts_table_name: str) -> None:
    """Create a standalone FTS5 virtual table for chunk text search."""
    cursor = conn.cursor()

    # Standalone FTS5 table - stores its own copy of text
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table_name} USING fts5(
            chunk_text
        )
    """)

    conn.commit()
    logger.info(f"Created FTS5 table '{fts_table_name}'")


def populate_fts_chunks_table(
    conn: sqlite3.Connection,
    vec0_table_name: str,
    fts_table_name: str,
    batch_size: int = 1000,
) -> int:
    """Populate FTS5 table with chunks from vec0 table, preserving rowids.

    Args:
        conn: Database connection
        vec0_table_name: Source vec0 table containing chunks
        fts_table_name: Target FTS5 table
        batch_size: Number of rows to insert per batch

    Returns:
        Number of chunks indexed
    """
    # Use separate cursors for reading and writing to avoid interference
    read_cursor = conn.cursor()
    write_cursor = conn.cursor()

    # Get total count for progress reporting
    read_cursor.execute(f"SELECT COUNT(*) FROM {vec0_table_name}")
    total_chunks = read_cursor.fetchone()[0]

    if total_chunks == 0:
        logger.info("No chunks found to index")
        return 0

    logger.info(f"Indexing {total_chunks} chunks from '{vec0_table_name}'...")

    # Read all chunks with their rowids
    read_cursor.execute(
        f"SELECT rowid, chunk_text FROM {vec0_table_name} ORDER BY rowid"
    )

    indexed_count = 0
    batch = []

    for row in read_cursor:
        rowid, chunk_text = row
        batch.append((rowid, chunk_text))

        if len(batch) >= batch_size:
            # Insert batch with explicit rowids using separate cursor
            write_cursor.executemany(
                f"INSERT INTO {fts_table_name}(rowid, chunk_text) VALUES (?, ?)", batch
            )
            indexed_count += len(batch)
            logger.info(f"  Indexed {indexed_count}/{total_chunks} chunks...")
            batch = []

    # Insert remaining rows
    if batch:
        write_cursor.executemany(
            f"INSERT INTO {fts_table_name}(rowid, chunk_text) VALUES (?, ?)", batch
        )
        indexed_count += len(batch)

    conn.commit()
    logger.info(f"Populated FTS index with {indexed_count} chunks")
    return indexed_count


def optimize_fts_index(conn: sqlite3.Connection, fts_table_name: str) -> None:
    """Optimize the FTS5 index for better query performance."""
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {fts_table_name}({fts_table_name}) VALUES('optimize')")
    conn.commit()
    logger.info(f"Optimized FTS5 index '{fts_table_name}'")


def verify_fts_setup(
    conn: sqlite3.Connection,
    vec0_table_name: str,
    fts_table_name: str,
) -> None:
    """Verify that the FTS setup is working correctly."""
    cursor = conn.cursor()

    # Check that the FTS table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (fts_table_name,),
    )
    if not cursor.fetchone():
        raise RuntimeError(f"FTS table '{fts_table_name}' was not created")

    # Check row counts match
    cursor.execute(f"SELECT COUNT(*) FROM {vec0_table_name}")
    vec0_count = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM {fts_table_name}")
    fts_count = cursor.fetchone()[0]

    if vec0_count != fts_count:
        raise RuntimeError(
            f"Row count mismatch: {vec0_table_name} has {vec0_count} rows, "
            f"{fts_table_name} has {fts_count} rows"
        )

    # Test a simple search query
    cursor.execute(f"""
        SELECT rowid, chunk_text 
        FROM {fts_table_name} 
        WHERE {fts_table_name} MATCH 'den' 
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        logger.info(f"Test query successful (found rowid {result[0]})")
    else:
        logger.warning(
            "Test query returned no results (this may be okay if no chunks contain 'den')"
        )

    logger.info(f"FTS setup verification passed ({fts_count} chunks indexed)")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create and populate FTS5 full-text search index for chunks in a vec0 table"
    )

    parser.add_argument(
        "--vec0-table",
        "-v",
        default="article_embeddings_e5",
        help="Name of the vec0 table containing chunks (default: article_embeddings_e5)",
    )
    parser.add_argument(
        "--fts-table",
        "-f",
        default=None,
        help="Name for the FTS5 table (default: fts_<vec0_table>)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1000,
        help="Batch size for inserting chunks (default: 1000)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--force", action="store_true", help="Recreate index even if it already exists"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.debug)

    # Derive FTS table name if not specified
    fts_table_name = args.fts_table or f"fts_{args.vec0_table}"

    try:
        settings = get_settings()
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")

        with create_connection() as conn:
            cursor = conn.cursor()

            # Check if vec0 table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (args.vec0_table,),
            )
            if not cursor.fetchone():
                logger.error(f"Vec0 table '{args.vec0_table}' does not exist.")
                sys.exit(1)

            # Check if FTS table already exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (fts_table_name,),
            )
            if cursor.fetchone() and not args.force:
                logger.error(
                    f"FTS index '{fts_table_name}' already exists. Use --force to recreate it."
                )
                sys.exit(1)

            if args.force:
                logger.info(f"Dropping existing FTS table '{fts_table_name}'...")
                cursor.execute(f"DROP TABLE IF EXISTS {fts_table_name}")
                conn.commit()

            logger.info(
                f"Creating FTS5 index '{fts_table_name}' for chunks in '{args.vec0_table}'..."
            )

            create_fts_chunks_table(conn, fts_table_name)
            populate_fts_chunks_table(
                conn, args.vec0_table, fts_table_name, args.batch_size
            )
            optimize_fts_index(conn, fts_table_name)
            verify_fts_setup(conn, args.vec0_table, fts_table_name)

            logger.info("FTS5 chunk index creation completed successfully!")
            logger.info(
                f"You can now search chunks using:\n"
                f"  SELECT rowid, bm25({fts_table_name}), chunk_text\n"
                f"  FROM {fts_table_name}\n"
                f"  WHERE {fts_table_name} MATCH 'your search term'\n"
                f"  ORDER BY bm25({fts_table_name})"
            )

    except Exception as e:
        logger.error(f"Error creating FTS index: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
