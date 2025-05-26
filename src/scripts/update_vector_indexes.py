"""CLI tool to update vector indexes with new, modified, or deleted content."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.lex_db.database import create_connection
from src.lex_db.config import get_settings
from src.lex_db.utils import get_logger, configure_logging
from src.lex_db.vector_store import update_vector_index

logger = get_logger()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Populate and update vector indexes with content"
    )

    parser.add_argument(
        "--index-name", "-i", required=True, help="Name of the vector index to update"
    )

    parser.add_argument(
        "--source-table",
        "-s",
        required=True,
        help="Source table containing text to embed",
    )

    parser.add_argument(
        "--source-column",
        "-c",
        required=True,
        help="Column in source table containing text to embed",
    )

    parser.add_argument(
        "--embedding-model",
        "-e",
        required=True,
        help="Embedding model to use (must match the one used to create the index)",
    )

    parser.add_argument(
        "--updated-column",
        default="changed_at",
        help="Column name for the last-updated timestamp in the source table",
    )

    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Size of text chunks for embedding"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of articles to process in one batch",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.debug)

    # Connect to database and update the vector index
    try:
        settings = get_settings()
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")
        db_conn = create_connection()

        update_vector_index(
            db_conn=db_conn,
            vector_index_name=args.index_name,
            source_table=args.source_table,
            text_column=args.source_column,
            embedding_model_choice=args.embedding_model,
            updated_at_column=args.updated_column,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )

        logger.info("Vector index update complete!")
    except Exception as e:
        logger.error(f"Error updating vector index: {str(e)}", exc_info=True)
        exit(1)
    finally:
        if "db_conn" in locals():
            db_conn.close()


if __name__ == "__main__":
    main()
