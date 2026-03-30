"""CLI tool to dump articles from the database to JSONL format.

This script exports articles from the PostgreSQL database to a JSONL file
that can be processed on a remote server for embedding generation.

Two modes are supported:
- full: Dump all articles (default)
- incremental: Only dump articles that have changed since last vector index update
"""

import argparse
import json
from datetime import datetime, timezone

from lex_db.database import get_db_connection
from lex_db.config import get_settings
from lex_db.utils import get_logger, configure_logging
from lex_db.vector_store import get_vector_index_metadata

logger = get_logger()


def dump_articles(
    output_file: str,
    mode: str = "full",
    index_name: str | None = None,
) -> dict[str, int]:
    """
    Dump articles from the database to a JSONL file.

    Args:
        output_file: Path to the output JSONL file
        mode: "full" to dump all articles, "incremental" to dump only changed articles
        index_name: Name of the vector index (required for incremental mode)

    Returns:
        Dictionary with stats: {"total_articles": N, "dumped_articles": M}
    """
    stats = {"total_articles": 0, "dumped_articles": 0}

    with get_db_connection() as db_conn:
        # Get total article count
        result = db_conn.execute("SELECT COUNT(*) as count FROM articles").fetchone()
        stats["total_articles"] = result["count"] if result else 1  # type: ignore
        logger.info(f"Total articles in database: {stats['total_articles']}")

        # Build query based on mode
        if mode == "incremental":
            if not index_name:
                raise ValueError("index_name is required for incremental mode")

            # Get metadata for the vector index
            metadata = get_vector_index_metadata(db_conn, index_name)
            if not metadata:
                raise ValueError(f"No metadata found for vector index '{index_name}'")

            updated_at_column = metadata.get("updated_at_column", "changed_at")
            logger.info(f"Using incremental mode with column: {updated_at_column}")

            # Find articles that have changed since the vector index was last updated
            # or articles that don't exist in the vector index at all
            query = f"""
                SELECT DISTINCT a.id, a.headword, a.xhtml_md, a.permalink, 
                       a.encyclopedia_id, a.changed_at, a.created_at
                FROM articles a
                LEFT JOIN {index_name} vi ON a.id = vi.source_article_id
                WHERE vi.source_article_id IS NULL
                   OR a.{updated_at_column} > vi.last_updated
                ORDER BY a.id
            """
        else:
            # Full dump - get all articles
            query = """
                SELECT id, headword, xhtml_md, permalink, 
                       encyclopedia_id, changed_at, created_at
                FROM articles
                ORDER BY id
            """

        # Open output file and write metadata header
        with open(output_file, "w", encoding="utf-8") as f:
            # Write metadata header
            metadata_header = {
                "_metadata": {
                    "dump_date": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                    "index_name": index_name,
                    "total_articles_in_db": stats["total_articles"],
                }
            }
            f.write(json.dumps(metadata_header, ensure_ascii=False) + "\n")

            # Stream articles to file
            logger.info(f"Starting {mode} dump to {output_file}...")
            cursor = db_conn.execute(query)  # type: ignore

            for row in cursor:
                # Handle datetime fields - they may be datetime objects or strings
                changed_at = row["changed_at"]  # type: ignore[call-overload]
                if changed_at and hasattr(changed_at, "isoformat"):
                    changed_at = changed_at.isoformat()
                created_at = row["created_at"]  # type: ignore[call-overload]
                if created_at and hasattr(created_at, "isoformat"):
                    created_at = created_at.isoformat()

                article = {
                    "id": row["id"],  # type: ignore[call-overload]
                    "headword": row["headword"],  # type: ignore[call-overload]
                    "xhtml_md": row["xhtml_md"],  # type: ignore[call-overload]
                    "permalink": row["permalink"],  # type: ignore[call-overload]
                    "encyclopedia_id": row["encyclopedia_id"],  # type: ignore[call-overload]
                    "changed_at": changed_at,
                    "created_at": created_at,
                }
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                stats["dumped_articles"] += 1

                # Log progress every 10000 articles
                if stats["dumped_articles"] % 10000 == 0:
                    logger.info(f"Dumped {stats['dumped_articles']} articles...")

    logger.info(
        f"Dump complete: {stats['dumped_articles']} articles written to {output_file}"
    )
    return stats


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Dump articles from database to JSONL format for remote processing"
    )
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Dump mode: 'full' for all articles, 'incremental' for changed articles only (default: full)",
    )
    parser.add_argument(
        "--index-name",
        "-i",
        help="Name of the vector index (required for incremental mode)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    # Validate arguments
    if args.mode == "incremental" and not args.index_name:
        parser.error("--index-name is required when using incremental mode")

    settings = get_settings()
    try:
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")
        stats = dump_articles(
            output_file=args.output,
            mode=args.mode,
            index_name=args.index_name,
        )

        logger.info(f"Successfully dumped {stats['dumped_articles']} articles")
        if stats["dumped_articles"] == 0:
            logger.warning("No articles were dumped. Check your filters.")

    except Exception as e:
        logger.error(f"Error dumping articles: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
