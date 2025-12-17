"""Import vector embeddings from JSONL format into PostgreSQL.

This script imports vector embeddings that were exported from SQLite
or generated externally into a PostgreSQL vector index.
"""

import argparse
import json

from lex_db.database import get_db_connection
from lex_db.utils import get_logger, configure_logging
from lex_db.vector_store import (
    add_precomputed_embeddings_to_vector_index,
    get_vector_index_metadata,
)
from lex_db.embeddings import get_embedding_dimensions, EmbeddingModel

logger = get_logger()


def import_embeddings(
    jsonl_file: str,
    index_name: str,
    batch_size: int = 1000,
) -> dict[str, int]:
    """
    Import embeddings from JSONL file into PostgreSQL vector index.

    Args:
        jsonl_file: Path to JSONL file with embeddings
        index_name: Name of the target vector index table
        batch_size: Number of embeddings to import per batch

    Returns:
        Dictionary with stats: {"created": N, "errors": K}
    """

    # Read and validate metadata
    logger.info(f"Reading embeddings from {jsonl_file}")
    with open(jsonl_file, "r") as f:
        first_line = f.readline()
        metadata = json.loads(first_line)

        if "_metadata" not in metadata:
            raise ValueError("First line must contain metadata with '_metadata' key")

        meta = metadata["_metadata"]
        total_chunks = meta.get("total_chunks", "unknown")
        source_index = meta.get("index_name", "unknown")

        logger.info(f"Source index: {source_index}")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Export date: {meta.get('export_date', 'unknown')}")

    # Verify target index exists and get its metadata
    with get_db_connection() as db_conn:
        index_metadata = get_vector_index_metadata(db_conn, index_name)
        if not index_metadata:
            raise ValueError(
                f"Vector index '{index_name}' not found. "
                f"Create it first using create_vector_index.py"
            )

        # Get expected embedding dimensions
        embedding_model = EmbeddingModel(index_metadata["embedding_model"])
        expected_dims = get_embedding_dimensions(embedding_model)
        logger.info(
            f"Target index '{index_name}' expects {expected_dims}-dimensional embeddings"
        )

    # Read all embeddings from JSONL file
    logger.info(f"Loading embeddings from {jsonl_file}")
    embeddings_data = []
    line_num = 0

    with open(jsonl_file, "r") as f:
        for line in f:
            line_num += 1

            # Skip metadata line
            if line_num == 1:
                continue

            try:
                data = json.loads(line)
                article_id = data["article_id"]
                chunk_idx = data["chunk_idx"]
                chunk_text = data["chunk_text"]
                embedding = data["embedding"]

                # Validate embedding dimensions
                if len(embedding) != expected_dims:
                    logger.error(
                        f"Line {line_num}: Embedding dimension mismatch "
                        f"(expected {expected_dims}, got {len(embedding)})"
                    )
                    continue

                embeddings_data.append((article_id, chunk_idx, chunk_text, embedding))

                # Log progress every 10000 lines
                if len(embeddings_data) % 10000 == 0:
                    logger.info(f"Loaded {len(embeddings_data)} embeddings...")

            except Exception as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(embeddings_data)} embeddings from file")

    # Import embeddings into PostgreSQL
    logger.info(f"Importing embeddings into index '{index_name}'")
    with get_db_connection() as db_conn:
        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=db_conn,
            vector_index_name=index_name,
            embeddings_data=embeddings_data,
            batch_size=batch_size,
        )

    return stats


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Import vector embeddings from JSONL into PostgreSQL"
    )
    parser.add_argument(
        "--jsonl-file", required=True, help="Path to JSONL file with embeddings"
    )
    parser.add_argument(
        "--index-name", required=True, help="Name of the target vector index table"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of embeddings to import per batch (default: 1000)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    try:
        stats = import_embeddings(
            jsonl_file=args.jsonl_file,
            index_name=args.index_name,
            batch_size=args.batch_size,
        )

        logger.info(f"Successfully imported {stats['created']} embeddings")
        if stats["errors"] > 0:
            logger.warning(f"Encountered {stats['errors']} errors during import")
            exit(1)

    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
