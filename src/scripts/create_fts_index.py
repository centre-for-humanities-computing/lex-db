"""Script to create and populate FTS5 (Full-Text Search) tables for the lexicon database."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.lex_db.utils import get_logger, configure_logging
from src.lex_db.database import create_connection
from src.lex_db.config import get_settings

logger = get_logger()

def create_fts_tables(conn) -> None:
    """Create FTS5 virtual tables for full-text search with external content."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_articles USING fts5(
            xhtml_md,
            content='articles',
            content_rowid='id'
        )
    """)
    
    # Create triggers to keep FTS table in sync with the articles table
    # INSERT trigger
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS articles_fts_insert AFTER INSERT ON articles BEGIN
            INSERT INTO fts_articles(rowid, xhtml_md) VALUES (new.id, new.xhtml_md);
        END
    """)
    
    # DELETE trigger
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS articles_fts_delete AFTER DELETE ON articles BEGIN
            INSERT INTO fts_articles(fts_articles, rowid, xhtml_md) VALUES ('delete', old.id, old.xhtml_md);
        END
    """)
    
    # UPDATE trigger
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS articles_fts_update AFTER UPDATE ON articles BEGIN
            INSERT INTO fts_articles(fts_articles, rowid, xhtml_md) VALUES ('delete', old.id, old.xhtml_md);
            INSERT INTO fts_articles(rowid, xhtml_md) VALUES (new.id, new.xhtml_md);
        END
    """)
    
    conn.commit()
    logger.info("✓ Created FTS5 view, table and triggers")


def populate_fts_tables(conn) -> None:
    """Populate FTS5 tables with existing data."""
    cursor = conn.cursor()
    
    # Get count of existing articles
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_entries = cursor.fetchone()[0]
    
    if total_entries == 0:
        logger.info("No articles found to index")
        return
    
    # Use the FTS5 'rebuild' command to populate from external content
    cursor.execute("INSERT INTO fts_articles(fts_articles) VALUES('rebuild')")
    
    conn.commit()
    logger.info(f"✓ Populated FTS index with {total_entries} articles using rebuild")


def optimize_fts_index(conn) -> None:
    """Optimize the FTS5 index for better performance."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO fts_articles(fts_articles) VALUES('optimize')")
    conn.commit()
    logger.info("✓ Optimized FTS5 index")


def verify_fts_setup(conn) -> None:
    """Verify that the FTS setup is working correctly."""
    cursor = conn.cursor()
    
    # Check that the FTS table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='fts_articles'
    """)
    if not cursor.fetchone():
        raise RuntimeError("fts_articles table was not created")
    
    # Check that triggers exist
    expected_triggers = [
        'articles_fts_insert',
        'articles_fts_delete',
        'articles_fts_update'
    ]
    
    for trigger_name in expected_triggers:
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='trigger' AND name=?
        """, (trigger_name,))
        if not cursor.fetchone():
            raise RuntimeError(f"Trigger {trigger_name} was not created")
    
    logger.info("✓ FTS setup verification passed")


if __name__ == "__main__":
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create and populate FTS5 full-text search index for articles"
    )
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Recreate index even if it already exists"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.debug)
    
    try:
        settings = get_settings()
        logger.info(f"Connecting to database at {settings.DATABASE_URL}")
        
        with create_connection() as conn:
            cursor = conn.cursor()
            
            # Check if FTS table already exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_articles'"            
            )                        
            if cursor.fetchone() and not args.force:                
                logger.error("FTS index already exists. Use --force to recreate it.")                
                sys.exit(1)                        
            if args.force:                
                logger.info("Dropping existing FTS tables...")                
                cursor.execute("DROP TABLE IF EXISTS fts_articles")                
                cursor.execute("DROP TRIGGER IF EXISTS articles_fts_insert")                
                cursor.execute("DROP TRIGGER IF EXISTS articles_fts_delete")                 
                cursor.execute("DROP TRIGGER IF EXISTS articles_fts_update")                
                cursor.execute("DROP VIEW IF EXISTS articles_fts_view")                
                conn.commit()                        
            logger.info("Creating FTS5 full-text search index for articles...")                        
            create_fts_tables(conn)            
            populate_fts_tables(conn)            
            optimize_fts_index(conn)            
            verify_fts_setup(conn)                        
            logger.info("✅ FTS5 index creation completed successfully!")            
            logger.info("You can now search articles using: SELECT * FROM fts_articles WHERE fts_articles MATCH 'your search term'")                
    except Exception as e:        
        logger.error(f"❌ Error creating FTS index: {str(e)}", exc_info=True)        
        sys.exit(1)

    