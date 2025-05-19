"""Main entry point for the Lex DB application."""

import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from lex_db.config import get_settings
from lex_db.database import get_db_info, verify_db_exists
from lex_db.api.routes import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: The configured FastAPI application.
    """
    settings = get_settings()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Verify database connection on startup."""
        if not verify_db_exists():
            raise Exception(f"Database file not found at {settings.DATABASE_URL}")
        yield

    app = FastAPI(
        title=settings.APP_NAME,
        description="A wrapper around a SQLite database for encyclopedia articles with vector and full-text search",
        lifespan=lifespan,
        version="0.1.0",
        debug=settings.DEBUG,
    )   
    
    @app.get("/", tags=["Health"])
    async def health_check():
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
    
    return app


app = create_app()


def main():
    """Run the application.
    
    Note: This function is intended to be run using UV:
    uv run main.py
    """
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
