"""
Script to populate vector_index_metadata for existing vector indexes.
"""

import sqlite3

from lex_db.vector_store import insert_vector_index_metadata
from lex_db.embeddings import EmbeddingModel
from lex_db.utils import ChunkingStrategy
from lex_db.config import get_settings
from lex_db.database import create_connection
from lex_db.utils import get_logger

logger = get_logger()

# List of existing vector indexes to add metadata for
INDEXES = [
    {"name": "e5_index", "embedding_model": EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE.value},
    {"name": "mock_index", "embedding_model": EmbeddingModel.MOCK_MODEL.value},
    {"name": "ada_002", "embedding_model": EmbeddingModel.OPENAI_ADA_002.value},
    {"name": "small_003", "embedding_model": EmbeddingModel.OPENAI_SMALL_003.value},
    {"name": "large_003", "embedding_model": EmbeddingModel.OPENAI_LARGE_003.value},
]

# Defaults for all indexes
SOURCE_TABLE = "articles"
SOURCE_COLUMN = "text"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHUNKING_STRATEGY = ChunkingStrategy.TOKENS.value


def main() -> None:
    settings = get_settings()
    logger.info(f"Connecting to database at {settings.DATABASE_URL}")
    db_conn = create_connection()
    for idx in INDEXES:
        try:
            logger.info(f"Inserting metadata for index: {idx['name']}")
            insert_vector_index_metadata(
                db_conn=db_conn,
                index_name=idx["name"],
                source_table=SOURCE_TABLE,
                source_column=SOURCE_COLUMN,
                embedding_model=idx["embedding_model"],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                chunking_strategy=CHUNKING_STRATEGY,
            )
            logger.info(f"Inserted metadata for {idx['name']}")
        except sqlite3.IntegrityError:
            logger.warning(f"Metadata for {idx['name']} already exists. Skipping.")
        except Exception as e:
            logger.error(f"Error inserting metadata for {idx['name']}: {e}")
    db_conn.close()
    logger.info("Done populating vector_index_metadata.")


if __name__ == "__main__":
    main()
