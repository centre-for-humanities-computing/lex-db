"""Script to update vector indexes with new, modified, or deleted content."""

import argparse
import json
import time
import sqlite3
import os
from pathlib import Path
from typing import Dict, Any

from lex_db.utils import get_logger, configure_logging
from lex_db.database import (
    get_db_connection, add_single_article_to_vector_index, 
    remove_article_from_vector_index
)

logger = get_logger()

def get_state_file_path(index_name: str) -> str:
    """Get the path to the state file for a vector index."""
    state_dir = Path.home() / ".lex-db" / "vector_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    return str(state_dir / f"{index_name}.json")


def get_last_processed_timestamp(index_name: str) -> float:
    """Get the last processed timestamp for a vector index."""
    state_file = get_state_file_path(index_name)
    if not os.path.exists(state_file):
        return 0.0
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        return state.get('last_timestamp', 0.0)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0.0


def set_last_processed_timestamp(index_name: str, timestamp: float):
    """Set the last processed timestamp for a vector index."""
    state_file = get_state_file_path(index_name)
    
    state = {
        'last_timestamp': timestamp,
        'index_name': index_name,
        'updated_at': time.time()
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def update_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    source_table: str,
    text_column: str,
    embedding_model_choice: str,
    updated_at_column: str = "updated_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> Dict[str, Any]:
    """Update a vector index with new, modified, or deleted content."""
    cursor = db_conn.cursor()
    last_timestamp = get_last_processed_timestamp(vector_index_name)
    current_timestamp = time.time()
    
    logger.info(f"Updating vector index '{vector_index_name}' from source table '{source_table}'")
    logger.info(f"Processing changes since timestamp: {last_timestamp}")
    
    stats = {
        "new_or_updated": 0,
        "deleted": 0,
        "errors": 0
    }
    
    # Check if the vector index exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (vector_index_name,)
    )
    if not cursor.fetchone():
        raise ValueError(f"Vector index '{vector_index_name}' does not exist")
    
    # Check if the source table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (source_table,)
    )
    if not cursor.fetchone():
        raise ValueError(f"Source table '{source_table}' does not exist")
    
    # Check if the updated_at column exists in the source table
    try:
        cursor.execute(f"SELECT {updated_at_column} FROM {source_table} LIMIT 1")
    except sqlite3.Error:
        raise ValueError(f"Column '{updated_at_column}' not found in table '{source_table}'")
    
    # Process new or updated articles
    try:
        logger.info("Processing new or updated articles...")
        cursor.execute(
            f"SELECT rowid, {text_column} FROM {source_table} WHERE {updated_at_column} > ?",
            (last_timestamp,)
        )
        new_or_updated_articles = cursor.fetchall()
        
        if not new_or_updated_articles:
            logger.info("No new or updated articles found")
        
        for article in new_or_updated_articles:
            article_rowid = article[0]
            text_content = article[1]
            
            try:
                # Remove any existing entries for this article (in case it's an update)
                remove_article_from_vector_index(db_conn, vector_index_name, article_rowid)
                
                # Add the article to the vector index
                if text_content:
                    add_single_article_to_vector_index(
                        db_conn=db_conn,
                        vector_index_name=vector_index_name,
                        article_rowid=article_rowid,
                        article_text=text_content,
                        embedding_model_choice=embedding_model_choice,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    stats["new_or_updated"] += 1
            except Exception as e:
                logger.error(f"Error processing article {article_rowid}: {str(e)}")
                stats["errors"] += 1
    except Exception as e:
        logger.error(f"Error processing new/updated articles: {str(e)}")
        stats["errors"] += 1
    
    # Process deleted articles by comparing vector index entries with source table
    try:
        logger.info("Checking for deleted articles...")
        # Get all unique source_article_ids from the vector index
        cursor.execute(f"SELECT DISTINCT source_article_id FROM {vector_index_name}")
        indexed_ids = {row[0] for row in cursor.fetchall()}
        
        # Get all rowids from the source table
        cursor.execute(f"SELECT rowid FROM {source_table}")
        source_ids = {str(row[0]) for row in cursor.fetchall()}
        
        # Find IDs that are in the index but not in the source table (deleted)
        ids_to_delete = indexed_ids - source_ids
        
        if not ids_to_delete:
            logger.info("No deleted articles found")
        
        # Remove deleted articles from the vector index
        for article_id_to_delete in ids_to_delete:
            try:
                remove_article_from_vector_index(db_conn, vector_index_name, article_id_to_delete)
                stats["deleted"] += 1
            except Exception as e:
                logger.error(f"Error removing article {article_id_to_delete}: {str(e)}")
                stats["errors"] += 1
    except Exception as e:
        logger.error(f"Error processing deleted articles: {str(e)}")
        stats["errors"] += 1
    
    # Update the timestamp only if there were no errors
    if stats["errors"] == 0:
        set_last_processed_timestamp(vector_index_name, current_timestamp)
        logger.info(f"Updated timestamp for index '{vector_index_name}' to {current_timestamp}")
    
    logger.info(f"Update summary: {stats}")
    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Update vector indexes with new content")
    
    parser.add_argument(
        "--index-name",
        "-i",
        required=True,
        help="Name of the vector index to update"
    )
    
    parser.add_argument(
        "--source-table",
        "-s",
        required=True,
        help="Source table containing text to embed"
    )
    
    parser.add_argument(
        "--source-column",
        "-c",
        required=True,
        help="Column in source table containing text to embed"
    )
    
    parser.add_argument(
        "--embedding-model",
        "-e",
        required=True,
        help="Embedding model to use (must match the one used to create the index)"
    )
    
    parser.add_argument(
        "--updated-column",
        default="updated_at",
        help="Column name for the last-updated timestamp in the source table"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks for embedding"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks"
    )
    
    parser.add_argument(
        "--db-path",
        default=str(Path.home() / ".lex-db" / "lex.db"),
        help="Path to SQLite database"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Connect to database and update the vector index
    try:
        logger.info(f"Connecting to database at {args.db_path}")
        db_conn = sqlite3.connect(args.db_path)
        
        update_vector_index(
            db_conn=db_conn,
            vector_index_name=args.index_name,
            source_table=args.source_table,
            text_column=args.source_column,
            embedding_model_choice=args.embedding_model,
            updated_at_column=args.updated_column,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        logger.info("Vector index update complete!")
    except Exception as e:
        logger.error(f"Error updating vector index: {str(e)}", exc_info=True)
        exit(1)
    finally:
        if 'db_conn' in locals():
            db_conn.close()


if __name__ == "__main__":
    main()
