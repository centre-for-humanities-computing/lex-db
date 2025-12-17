"""Export vector embeddings from SQLite database to JSONL format.

This script exports vector indexes from the old SQLite database format
to JSONL files that can be imported into PostgreSQL using import_vector_embeddings.py.
"""

import argparse
import json
import sqlite3
import struct
from datetime import datetime

import sqlite_vec

from lex_db.utils import get_logger, configure_logging

logger = get_logger()


def export_vector_index(
    sqlite_db_path: str,
    index_name: str,
    output_file: str,
) -> dict[str, int]:
    """
    Export a vector index from SQLite to JSONL format.

    Args:
        sqlite_db_path: Path to SQLite database file
        index_name: Name of the vector index table to export
        output_file: Path to output JSONL file

    Returns:
        Dictionary with stats: {"exported": N, "errors": K}
    """
    stats = {"exported": 0, "errors": 0}

    # Connect to SQLite database (read-only)
    conn = sqlite3.connect(f"file:{sqlite_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (index_name,),
        )
        if not cursor.fetchone():
            raise ValueError(f"Vector index table '{index_name}' not found in database")

        # Get total count
        cursor.execute(f"SELECT COUNT(*) as count FROM {index_name}")
        total_chunks = cursor.fetchone()["count"]
        logger.info(f"Found {total_chunks} chunks in index '{index_name}'")

        # Open output file for writing
        with open(output_file, "w") as f:
            # Write metadata header
            metadata = {
                "_metadata": {
                    "index_name": index_name,
                    "total_chunks": total_chunks,
                    "export_date": datetime.now().isoformat(),
                    "source_db": sqlite_db_path,
                }
            }
            f.write(json.dumps(metadata) + "\n")

            # Query all chunks from the vector index
            cursor.execute(
                f"""
                SELECT 
                    source_article_id,
                    chunk_sequence_id,
                    chunk_text,
                    embedding
                FROM {index_name}
                ORDER BY source_article_id, chunk_sequence_id
                """
            )

            # Process chunks one at a time (memory efficient)
            for row in cursor:
                article_id = str(row["source_article_id"])
                chunk_idx = str(row["chunk_sequence_id"])

                try:
                    chunk_text = row["chunk_text"]
                    embedding_blob = row["embedding"]

                    # Deserialize binary vector data from sqlite_vec
                    # sqlite_vec stores vectors as binary blobs of float32 values
                    # Each float32 is 4 bytes, so we can calculate the number of dimensions
                    num_floats = len(embedding_blob) // 4

                    # Unpack binary data as little-endian float32 values
                    embedding_list = list(
                        struct.unpack(f"<{num_floats}f", embedding_blob)
                    )

                    # Validate embedding is a list of numbers
                    if not isinstance(embedding_list, list) or not all(
                        isinstance(x, (int, float)) for x in embedding_list
                    ):
                        logger.error(
                            f"Invalid embedding format for article {article_id}, chunk {chunk_idx}"
                        )
                        stats["errors"] += 1
                        continue

                    # Write JSONL entry
                    entry = {
                        "article_id": article_id,
                        "chunk_idx": chunk_idx,
                        "chunk_text": chunk_text,
                        "embedding": embedding_list,
                    }
                    f.write(json.dumps(entry) + "\n")
                    stats["exported"] += 1

                    # Log progress every 1000 chunks
                    if stats["exported"] % 1000 == 0:
                        logger.info(
                            f"Exported {stats['exported']}/{total_chunks} chunks..."
                        )

                except Exception as e:
                    logger.error(
                        f"Error exporting article {article_id}, chunk {chunk_idx}: {e}"
                    )
                    stats["errors"] += 1

        logger.info(f"Export complete: {stats}")
        return stats

    finally:
        conn.close()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Export vector index from SQLite to JSONL format"
    )
    parser.add_argument(
        "--sqlite-db", required=True, help="Path to SQLite database file"
    )
    parser.add_argument(
        "--index-name", required=True, help="Name of the vector index table to export"
    )
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    try:
        stats = export_vector_index(
            sqlite_db_path=args.sqlite_db,
            index_name=args.index_name,
            output_file=args.output,
        )

        logger.info(
            f"Successfully exported {stats['exported']} chunks to {args.output}"
        )
        if stats["errors"] > 0:
            logger.warning(f"Encountered {stats['errors']} errors during export")
            exit(1)

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
