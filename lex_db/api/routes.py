"""API routes for Lex DB."""

from fastapi import APIRouter, HTTPException

from lex_db.database import get_db_connection


router = APIRouter(prefix="/api", tags=["API"])


@router.get("/tables")
async def get_tables():
    """Get a list of tables in the database.
    
    Returns:
        dict: List of tables in the database.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")