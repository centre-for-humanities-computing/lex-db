"""Script to create a vector index from a source table in the database."""

import argparse
import sqlite3
from pathlib import Path

from lex_db.utils import get_logger, configure_logging
from lex_db.embeddings import EmbeddingModel
from lex_db.database import create_vector_index

logger = get_logger()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create a vector index from a source table"
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
        "--index-name", "-i", required=True, help="Name for the new vector index"
    )

    parser.add_argument(
        "--embedding-model",
        "-e",
        choices=[m.value for m in EmbeddingModel],
        default=EmbeddingModel.LOCAL_E5_MULTILINGUAL.value,
        help="Embedding model to use",
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
        "--db-path",
        default=str(Path.home() / "lex-db" / "lex.db"),
        help="Path to SQLite database",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.debug)

    # Connect to database
    try:
        logger.info(f"Connecting to database at {args.db_path}")
        db_conn = sqlite3.connect(args.db_path)

        logger.info(
            f"Creating vector index '{args.index_name}' using {args.embedding_model} model"
        )
        create_vector_index(
            db_conn=db_conn,
            source_table_name=args.source_table,
            source_text_column=args.source_column,
            vector_index_name=args.index_name,
            embedding_model_choice=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        logger.info("Vector index creation complete!")
    except Exception as e:
        logger.error(f"Error creating vector index: {str(e)}", exc_info=True)
    finally:
        if "db_conn" in locals():
            db_conn.close()


if __name__ == "__main__":
    main()
