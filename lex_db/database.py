"""Database connection and operations for Lex DB."""

import os
import json
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from typing import Generator, List

from lex_db.config import get_settings
from lex_db.embeddings import generate_embeddings, get_embedding_dimensions
from lex_db.utils import split_document_into_chunks


def get_db_path() -> str:
    """Get the path to the SQLite database file.
    
    Returns:
        str: Path to the SQLite database file.
    """
    settings = get_settings()
    return settings.DATABASE_URL


def verify_db_exists() -> bool:
    """Verify that the database file exists.
    
    Returns:
        bool: True if the database file exists, False otherwise.
    """
    db_path = get_db_path()
    return os.path.exists(db_path)


def create_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Connection to the SQLite database.
        
    Raises:
        FileNotFoundError: If the database file does not exist.
        sqlite3.Error: If there is an error connecting to the database.
    """
    db_path = get_db_path()
    
    if not verify_db_exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Enable loading extensions
        conn.enable_load_extension(True)
        
        # Load the sqlite-vec extension
        try:
            sqlite_vec.load(conn)
        except sqlite3.Error as e:
            conn.close()
            raise sqlite3.Error(f"Failed to load sqlite-vec extension: {e}")
        
        # Configure connection
        conn.row_factory = sqlite3.Row
        
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error connecting to database: {e}")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a connection to the SQLite database.
    
    Yields:
        sqlite3.Connection: Connection to the SQLite database.
        
    Raises:
        FileNotFoundError: If the database file does not exist.
        sqlite3.Error: If there is an error connecting to the database.
    """
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_db_info() -> dict:
    """Get information about the database.
    
    Returns:
        dict: Information about the database.
    """
    with get_db_connection() as conn:
        # Get the list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get the SQLite version
        cursor.execute("SELECT sqlite_version();")
        sqlite_version = cursor.fetchone()[0]
        
        # Check if sqlite-vec is loaded
        try:
            cursor.execute("SELECT vec_version();")
            vector_version = cursor.fetchone()[0]
        except sqlite3.Error:
            vector_version = "Not loaded"
        
        return {
            "path": get_db_path(),
            "tables": tables,
            "sqlite_version": sqlite_version,
            "vector_version": vector_version,
        }


def create_vector_index(
    db_conn: sqlite3.Connection,
    source_table_name: str,
    source_text_column: str,
    vector_index_name: str,
    embedding_model_choice: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    factory: str = None
):
    """Create a vector index for a SQLite table.
    
    Args:
        db_conn: SQLite database connection
        source_table_name: Name of the table containing text to index
        source_text_column: Name of the column containing text to index
        vector_index_name: Name to give the vector index table
        embedding_model_choice: Which embedding model to use
        chunk_size: Maximum size of each text chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Raises:
        sqlite3.Error: If there's an error creating or populating the index
    """
    cursor = db_conn.cursor()
    embedding_dim = get_embedding_dimensions(embedding_model_choice)
    
    # Create the sqlite-vec virtual table
    create_table_sql = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {vector_index_name} USING vec0(
        embedding {embedding_dim}D,
        source_article_id TEXT,
        chunk_sequence_id INTEGER,
        chunk_text TEXT
    );
    """
    cursor.execute(create_table_sql)
    print(f"Virtual table {vector_index_name} created or already exists.")
    
    # Fetch articles from the source table
    cursor.execute(f"SELECT rowid, {source_text_column} FROM {source_table_name}")
    articles = cursor.fetchall()
    
    # Process each article: chunk, embed, and insert
    for article_rowid, text_content in articles:
        if not text_content:
            continue  # Skip if no text
            
        chunks = split_document_into_chunks(
            text_content, 
            chunk_size=chunk_size, 
            overlap=chunk_overlap
        )
        
        if not chunks:
            continue
            
        chunk_embeddings = generate_embeddings(chunks, model_choice=embedding_model_choice)
        
        for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
            # Convert embedding to JSON string for sqlite-vec
            embedding_json = json.dumps(chunk_embedding)
            
            insert_sql = f"""
            INSERT INTO {vector_index_name} (embedding, source_article_id, chunk_sequence_id, chunk_text)
            VALUES (?, ?, ?, ?);
            """
            cursor.execute(
                insert_sql, 
                (embedding_json, str(article_rowid), i, chunk_text)
            )
    
    db_conn.commit()
    print(f"Finished populating {vector_index_name} from {source_table_name}.")


def search_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    query_text: str,
    embedding_model_choice: str,
    top_k: int = 5
) -> List[dict]:
    """Search a vector index for similar chunks to the query text.
    
    Args:
        db_conn: SQLite database connection
        vector_index_name: Name of the vector index to search
        query_text: The query text to search for
        embedding_model_choice: Which embedding model to use (must match the index)
        top_k: Maximum number of results to return
        
    Returns:
        List of dictionaries containing search results
        
    Raises:
        sqlite3.Error: If there's an error searching the index
    """
    cursor = db_conn.cursor()
    
    # Generate embedding for the query text
    query_vector = generate_embeddings([query_text], model_choice=embedding_model_choice)[0]
    query_vector_json = json.dumps(query_vector)
    
    # Perform the search using sqlite-vec's MATCH operator
    search_sql = f"""
    SELECT rowid, source_article_id, chunk_sequence_id, chunk_text, distance
    FROM {vector_index_name}
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT ?;
    """
    
    cursor.execute(search_sql, (query_vector_json, top_k))
    results = cursor.fetchall()
    
    # Format results
    formatted_results = [
        {
            "id_in_index": row[0],
            "source_article_id": row[1],
            "chunk_seq": row[2],
            "chunk_text": row[3],
            "distance": row[4]
        }
        for row in results
    ]
    
    return formatted_results