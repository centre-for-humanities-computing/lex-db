"""API routes for Lex DB."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional

from lex_db.embeddings import EmbeddingModel
from lex_db.utils import get_logger
import lex_db.database as db
from lex_db.vector_store import (
    VectorSearchResults,
    search_vector_index,
    get_all_vector_index_metadata,
    get_vector_index_metadata,
)

logger = get_logger()
router = APIRouter(prefix="/api", tags=["lex-db"])


@router.get(
    "/tables", operation_id="get_tables", summary="Get a list of tables in the database"
)
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

    query_text: str
    top_k: int = 5


@router.post(
    "/vector-search/indexes/{index_name}/query",
    operation_id="vector_search",
    summary="Search a vector index for similar content to the query text",
)
async def vector_search(
    index_name: str, request: VectorSearchRequest
) -> VectorSearchResults:
    """Search a vector index for similar content to the query text."""
    try:
        with db.get_db_connection() as conn:
            meta = get_vector_index_metadata(conn, index_name)
            if not meta:
                raise HTTPException(
                    status_code=404, detail=f"Vector index '{index_name}' not found"
                )
            embedding_model = meta["embedding_model"]
            logger.info(
                f"Searching index '{index_name}' for: {request.query_text} using model {embedding_model}"
            )
            results = search_vector_index(
                db_conn=conn,
                vector_index_name=index_name,
                query_text=request.query_text,
                embedding_model=embedding_model,
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
    operation_id="get_articles",
    summary="An endpoint for filtering articles based on metadata such as id, text search, etc. Query parameters are used for filtering (e.g. GET /articles?query=Rundetårn, or GET /articles?ids=1&ids=2&ids=5)",
)
async def get_articles(
    request: Request,
    query: Optional[str] = Query(None, description="Text search in articles"),
    ids: Optional[str] = Query(
        None,
        description="List of article IDs (comma-separated, JSON list, or repeated)",
    ),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
) -> db.SearchResults:
    """
    Filter articles based on metadata such as id, text search, etc.
    """
    try:
        # Try to get all 'ids' parameters (repeated or single)
        raw_ids = request.query_params.getlist("ids")
        parsed_ids = None
        if raw_ids:
            # If repeated: /api/articles?ids=1&ids=2
            if len(raw_ids) > 1:
                try:
                    parsed_ids = [int(x) for x in raw_ids]
                except Exception:
                    raise HTTPException(status_code=422, detail="Invalid ids format")
            # If single: could be comma-separated or JSON list
            else:
                ids_str = raw_ids[0].strip()
                try:
                    if ids_str.startswith("[") and ids_str.endswith("]"):
                        import json

                        parsed_ids = json.loads(ids_str)
                    else:
                        parsed_ids = [int(x) for x in ids_str.split(",") if x.strip()]
                except Exception:
                    raise HTTPException(status_code=422, detail="Invalid ids format")
        # If only id filter is provided, fetch by id(s)
        if parsed_ids and (not query or not query.strip()):
            logger.info(f"Fetching articles by id: {parsed_ids}")
            results = db.get_articles_by_ids(parsed_ids, limit=limit)
            return results

        # If only query or both query and id are provided
        if query and query.strip():
            logger.info(f"Full-text search for: {query}, id filter: {parsed_ids}")
            results = db.search_lex_fts(query=query, ids=parsed_ids, limit=limit)
            return results
        raise ValueError(
            "At least one filter parameter ('query' or 'ids') must be provided"
        )
    except ValueError as e:
        logger.error(f"Validation error in article search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in article search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get(
    "/vector-search/indexes",
    operation_id="list_vector_indexes",
    summary="List all vector indexes and their metadata",
)
async def list_vector_indexes() -> list[dict]:
    """Return a list of all vector indexes and their metadata."""
    try:
        with db.get_db_connection() as conn:
            return get_all_vector_index_metadata(conn)
    except Exception as e:
        logger.error(f"Error listing vector indexes: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing vector indexes: {str(e)}"
        )


@router.get(
    "/vector-search/indexes/{index_name}",
    operation_id="get_vector_index",
    summary="Get metadata for a specific vector index",
)
async def get_vector_index(index_name: str) -> dict:
    """Return metadata for a specific vector index."""
    try:
        with db.get_db_connection() as conn:
            meta = get_vector_index_metadata(conn, index_name)
            if not meta:
                raise HTTPException(
                    status_code=404, detail=f"Vector index '{index_name}' not found"
                )
            return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vector index metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting vector index metadata: {str(e)}"
        )


class BenchmarkEmbeddingsRequest(BaseModel):
    model_choice: EmbeddingModel = EmbeddingModel.LOCAL_E5_MULTILINGUAL
    num_texts: int = 50
    text_length: int = 200


class BenchmarkEmbeddingsResponse(BaseModel):
    num_texts: int
    avg_text_length: int
    total_time_seconds: float
    texts_per_second: float
    ms_per_text: float
    embedding_dimension: int


@router.post(
    "/benchmark/embeddings",
    operation_id="benchmark_embeddings",
    summary="Benchmark embedding generation performance",
)
async def benchmark_embeddings(
    request: BenchmarkEmbeddingsRequest,
) -> BenchmarkEmbeddingsResponse:
    """Benchmark embedding generation with configurable parameters."""
    import time
    import random
    import string
    from lex_db.embeddings import generate_embeddings

    # Generate test texts
    texts = []
    for _ in range(request.num_texts):
        words = []
        for _ in range(request.text_length // 5):
            word = "".join(
                random.choices(string.ascii_lowercase, k=random.randint(3, 10))
            )
            words.append(word)
        texts.append(" ".join(words))

    start = time.time()
    embeddings = generate_embeddings(texts, request.model_choice)
    elapsed = time.time() - start

    return BenchmarkEmbeddingsResponse(
        num_texts=request.num_texts,
        avg_text_length=request.text_length,
        total_time_seconds=round(elapsed, 3),
        texts_per_second=round(request.num_texts / elapsed, 2),
        ms_per_text=round((elapsed / request.num_texts) * 1000, 2),
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
    )
