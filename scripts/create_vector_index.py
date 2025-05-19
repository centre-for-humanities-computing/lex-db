"""Script to create a vector index for a SQLite table."""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add the parent directory to sys.path to import lex_db modules
sys.path.append(str(Path(__file__).parent.parent))

from lex_db.database import create_vector_index, get_db_connection
from lex_db.embeddings import EmbeddingModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a vector index for a SQLite table")
    
    parser.add_argument(
        "--source-table",
        "-t",
        required=True,
        help="Source table name containing text to index",
    )
    
    parser.add_argument(
        "--source-column",
        "-c",
        required=True,
        help="Column name in the source table containing text to index",
    )
    
    parser.add_argument(
        "--index-name",
        "-i",
        required=True,
        help="Name to give the vector index table",
    )
    
    parser.add_argument(
        "--embedding-model",
        "-e",
        choices=[m.value for m in EmbeddingModel],
        default=EmbeddingModel.LOCAL_SENTENCE_TRANSFORMER,
        help="Embedding model to use",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum size of each text chunk in characters",
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of characters to overlap between chunks",
    )

    return parser.parse_args()


def main():
    """Main function to create a vector index."""
    args = parse_args()
    
    try:
        with get_db_connection() as conn:
            create_vector_index(
                db_conn=conn,
                source_table_name=args.source_table,
                source_text_column=args.source_column,
                vector_index_name=args.index_name,
                embedding_model_choice=args.embedding_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            print(f"Successfully created vector index '{args.index_name}'")
    except (sqlite3.Error, ValueError) as e:
        print(f"Error creating vector index: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
