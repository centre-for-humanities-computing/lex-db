"""API routes for Lex DB."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from src.lex_db.utils import get_logger
from src.lex_db.embeddings import EmbeddingModel
from src.lex_db.database import get_db_connection
from src.lex_db.vector_store import search_vector_index

logger = get_logger()
router = APIRouter(prefix="/api", tags=["API"])


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
async def vector_search(request: VectorSearchRequest) -> dict[str, list]:
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
            return {"results": results}
    except ValueError as e:
        logger.error(f"Validation error in vector search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
