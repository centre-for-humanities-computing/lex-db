"""API routes for Lex DB."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Union

from src.lex_db.utils import get_logger
from src.lex_db.embeddings import EmbeddingModel
import src.lex_db.database as db
from src.lex_db.vector_store import VectorSearchResults, search_vector_index

logger = get_logger()
router = APIRouter(prefix="/api", tags=["lex-db"])


@router.get("/tables")
async def get_tables() -> dict[str, list[str]]:
    """Get a list of tables in the database."""
    try:
        with db.get_db_connection() as conn:
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

        with db.get_db_connection() as conn:
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


@router.get(
    "/articles",
    summary="An endpoint for filtering articles based on metadata such as id, text search, etc. Query parameters are used for filtering (e.g. GET /articles?query=RoundTower, or GET /articles?id=[1,2,5])",
)
async def get_articles(
    query: Optional[str] = Query(None, description="Text search in articles"),
    id: Union[List[int], int, None] = Query(None, description="List of article IDs"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
) -> db.SearchResults:
    """
    Filtrér artikler baseret på tekstsøgning og/eller id.
    """
    try:
        # Accept both single and multiple ids
        ids = id if id is not None else None
        if ids is not None and not isinstance(ids, list):
            ids = [ids]
        # Remove duplicates and None values
        if ids:
            ids = [i for i in ids if i is not None]
            if len(ids) == 0:
                ids = None

        # If only id filter is provided, fetch by id(s)
        if ids and (not query or not query.strip()):
            logger.info(f"Fetching articles by id: {ids}")
            results = db.get_articles_by_ids(ids, limit=limit)
            return results

        # If only query or both query and id are provided
        if query and query.strip():
            logger.info(f"Full-text search for: {query}, id filter: {ids}")
            results = db.search_lex_fts(query=query, ids=ids, limit=limit)
            return results
        raise ValueError(
            "At least one filter parameter ('query' or 'id') must be provided"
        )
    except ValueError as e:
        logger.error(f"Validation error in article search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in article search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
