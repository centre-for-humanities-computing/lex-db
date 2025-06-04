"""API routes for Lex DB."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from src.lex_db.utils import get_logger
from src.lex_db.embeddings import EmbeddingModel
from src.lex_db.database import (
    FullTextSearchResults,
    get_db_connection,
    search_lex_fts,
)
from src.lex_db.vector_store import VectorSearchResults, search_vector_index

logger = get_logger()
router = APIRouter(prefix="/api", tags=["lex-db"])


@router.get("/tables")
async def get_tables() -> dict[str, list[str]]:
    """Get a list of tables in the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


class VectorSearchRequest(BaseModel):
    """Vector search request model."""

    vector_index_name: str
    query_text: str
    embedding_model_choice: str
    top_k: int = 5


@router.post("/vector-search")
async def vector_search(request: VectorSearchRequest) -> VectorSearchResults:
    """Search a vector index for similar content to the query text."""
    try:
        # Validate embedding model choice
        if request.embedding_model_choice not in [m.value for m in EmbeddingModel]:
            raise ValueError(
                f"Unsupported embedding model: {request.embedding_model_choice}"
            )

        with get_db_connection() as conn:
            logger.info(
                f"Searching index '{request.vector_index_name}' for: {request.query_text}"
            )
            results = search_vector_index(
                db_conn=conn,
                vector_index_name=request.vector_index_name,
                query_text=request.query_text,
                embedding_model_choice=EmbeddingModel(request.embedding_model_choice),
                top_k=request.top_k,
            )
            return results
    except ValueError as e:
        logger.error(f"Validation error in vector search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


class FullTextSearchRequest(BaseModel):
    """Full-text search request model."""

    query: str
    limit: int = 50


@router.post("/search")
async def full_text_search(request: FullTextSearchRequest) -> FullTextSearchResults:
    """Search lexicon entries using full-text search."""
    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        # Validate pagination parameters
        if request.limit < 1 or request.limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        logger.info(f"Full-text search for: {request.query}")
        results = search_lex_fts(query=request.query, limit=request.limit)
        return FullTextSearchResults(
            entries=results.entries,
            total=results.total,
            query=request.query,
            limit=request.limit,
        )
    except ValueError as e:
        logger.error(f"Validation error in full-text search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in full-text search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
