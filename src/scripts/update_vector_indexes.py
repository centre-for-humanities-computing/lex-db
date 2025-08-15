"""CLI tool to update vector indexes with new, modified, or deleted content."""

import argparse
import sys
from pathlib import Path

# Adjust path to import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.lex_db.database import create_connection
from src.lex_db.config import get_settings
from src.lex_db.utils import get_logger, configure_logging
from src.lex_db.vector_store import update_vector_index, get_vector_index_metadata
from src.lex_db.embeddings import EmbeddingModel

logger = get_logger()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update vector index using stored configuration from metadata"
    )
    parser.add_argument(
        "--index-name", "-i", required=True, help="Name of the vector index to update"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for processing"
    )

    args = parser.parse_args()
    configure_logging(args.debug)

    settings = get_settings()
    try:
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")
        db_conn = create_connection()

        # Fetch metadata for the index
        metadata = get_vector_index_metadata(db_conn, args.index_name)
        if not metadata:
            logger.error(f"No metadata found for vector index '{args.index_name}'.")
            exit(1)

        # Validate required fields
        required_fields = [
            "source_table",
            "source_column",
            "embedding_model",
            "chunk_size",
            "chunk_overlap",
            "chunking_strategy",
        ]
        missing = [f for f in required_fields if not metadata.get(f)]
        if missing:
            logger.error(f"Missing metadata fields: {missing}")
            exit(1)

        # Use metadata values
        update_vector_index(
            db_conn=db_conn,
            vector_index_name=args.index_name,
            source_table=metadata["source_table"],
            text_column=metadata["source_column"],
            embedding_model_choice=EmbeddingModel(metadata["embedding_model"]),
            updated_at_column=metadata.get(
                "updated_at_column", "changed_at"
            ),  # Optional override field
            chunk_size=int(metadata["chunk_size"]),
            chunk_overlap=int(metadata["chunk_overlap"]),
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
