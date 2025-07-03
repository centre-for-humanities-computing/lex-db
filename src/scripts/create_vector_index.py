"""Script to create a vector index structure (without populating it) in the database."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.lex_db.utils import get_logger, configure_logging, ChunkingStrategy
from src.lex_db.embeddings import EmbeddingModel
from src.lex_db.vector_store import create_vector_index
from src.lex_db.database import create_connection
from src.lex_db.config import get_settings

logger = get_logger()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create an empty vector index structure (without populating it)"
    )

    parser.add_argument(
        "--index-name", "-i", required=True, help="Name for the new vector index"
    )

    parser.add_argument(
        "--embedding-model",
        "-e",
        choices=[m.value for m in EmbeddingModel],
        default=EmbeddingModel.OPENAI_SMALL_003.value,
        help="Embedding model to use",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--force-drop",
        action="store_true",
        help="Force drop of existing index before creation",
    )

    parser.add_argument(
        "--source-table",
        required=True,
        help="Source table containing text to embed (for metadata)",
    )
    parser.add_argument(
        "--source-column",
        required=True,
        help="Column in source table containing text to embed (for metadata)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks for embedding (for metadata)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks (for metadata)",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=ChunkingStrategy.TOKENS.value,
        choices=[s.value for s in ChunkingStrategy],
        help="Name of the chunking strategy (for metadata)",
    )

    args = parser.parse_args()
    settings = get_settings()

    # Configure logging
    configure_logging(args.debug)

    # Connect to database
    try:
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")
        db_conn = create_connection()

        logger.info(
            f"Creating empty vector index '{args.index_name}' using {args.embedding_model} model"
        )
        create_vector_index(
            db_conn=db_conn,
            vector_index_name=args.index_name,
            embedding_model_choice=EmbeddingModel(args.embedding_model),
            force=args.force_drop,
            source_table=args.source_table,
            source_column=args.source_column,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunking_strategy=ChunkingStrategy(args.chunking_strategy),
        )
        logger.info(
            "Vector index structure created! Use update_vector_indexes.py to populate it."
        )
    except Exception as e:
        logger.error(f"Error creating vector index structure: {str(e)}", exc_info=True)
    finally:
        if "db_conn" in locals():
            db_conn.close()


if __name__ == "__main__":
    main()
