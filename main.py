"""Main entry point for the Lex DB application."""

import asyncio
import os
import multiprocessing
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from lex_db.config import get_settings
from lex_db.database import get_db_info, get_connection_pool
from lex_db.api.routes import router as api_router


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database connection pool and preload embedding model on startup."""
    # Initialize connection pool at startup
    pool = get_connection_pool()
    print(
        f"✓ Database connection pool initialized (min={settings.DB_POOL_MIN_SIZE}, max={settings.DB_POOL_MAX_SIZE})"
    )

    # Preload the default embedding model so the first request doesn't pay
    # the cold-start cost (model download/export/quantization).
    try:
        from lex_db.embeddings import EmbeddingModel, get_local_embedding_model

        default_model = EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE
        print(f"⏳ Preloading embedding model: {default_model.value} ...")
        # Run in a thread to avoid blocking the event loop during startup
        await asyncio.to_thread(get_local_embedding_model, default_model)
        print(f"✓ Embedding model preloaded: {default_model.value}")
    except Exception as e:
        print(f"⚠ Could not preload embedding model: {e}")

    yield

    # Close connection pool on shutdown
    pool.close()
    print("✓ Database connection pool closed")


app = FastAPI(
    title=settings.APP_NAME,
    description="A PostgreSQL database API for encyclopedia articles with vector and full-text search",
    lifespan=lifespan,
    version="0.1.0",
    debug=settings.DEBUG,
)

# Compress large responses (xhtml_md payloads can be substantial)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/", tags=["Health"])
def health_check() -> dict:
    """Health check endpoint.

    Returns:
        dict: Health check information.
    """
    try:
        db_info = get_db_info()
        return {
            "status": "ok",
            "app_name": settings.APP_NAME,
            "database": db_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# Include API routes
app.include_router(api_router)


def main() -> None:
    """Run the application.

    Note: This function is intended to be run using UV:
    uv run main.py
    """
    host = os.getenv("DEPLOY_DOMAIN", "0.0.0.0")
    port = int(os.getenv("DEPLOY_PORT", "8000"))
    settings = get_settings()

    # In DEBUG mode, use reload for development (incompatible with workers>1).
    # In production, use multiple workers for throughput.
    if settings.DEBUG:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        workers = settings.WEB_CONCURRENCY or multiprocessing.cpu_count()
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
        )


if __name__ == "__main__":
    main()
