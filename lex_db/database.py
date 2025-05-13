"""Database connection and operations for Lex DB."""

import os
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from typing import Generator

from lex_db.config import get_settings


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