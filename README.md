# lex-db
A repository for interacting with the lex database for the Lex AI project. This project provides a wrapper around a SQLite database to enable querying encyclopedia articles via API requests, supporting both **vector (semantic) search** and **full-text/keyword search**.

## Features
- SQLite database access with `sqlite-vec` for vector search
- Full-text search capabilities using FTS5
- FastAPI-based REST API with automatic OpenAPI documentation
- Hybrid querying via metadata filtering and text search
- Vector index management and semantic search

## Requirements
- Python 3.12+
- [Astral UV](https://github.com/astral-sh/uv) for package management
- SQLite compiled with the `sqlite-vec` extension (required for vector search)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lex-db.git
   cd lex-db
   ```

2. Install dependencies using Make:
   ```bash
   make install
   ```

3. Create a `.env` file (or modify the existing one) to set the database path:
   ```
   # Database settings
   DATABASE_URL=PATH/TO/DB_FILE.db
   ```

## Usage

### Scripts
- `create_fts_index.py`: Creates a full-text search index on a specified column in a table.
  ```bash
  uv run src/scripts/create_fts_index.py <table_name> <column_name>
  ```

- `create_vector_index.py`: Creates a new vector index for semantic search on a given column.
  ```bash
  uv run src/scripts/create_vector_index.py <table_name> <column_name>
  ```

- `update_vector_indexes.py`: Populates vector indexes with embeddings using OpenAI or another embedding provider.
  ```bash
  uv run src/scripts/update_vector_indexes.py
  ```

> ⚠️ Note: While `create_openai_embedding_batches.py` and `add_batch_embeddings_to_index.py` exist for batch processing, they are not recommended due to reliability issues with the OpenAI Batch API. Use `update_vector_indexes.py` instead.

### Running the API Server
Start the FastAPI server:
```bash
make run
```
The server will be available at `http://0.0.0.0:8000`.

### API Endpoints

#### List Database Tables
- **`GET /api/tables`**
  - Returns a list of all tables in the database.
  - Example response:
    ```json
    {
      "tables": ["articles", "vector_index", "metadata"]
    }
    ```

#### Filter Articles by Metadata
- **`GET /api/articles`**
  - Retrieve articles filtered by ID or full-text search query.
  - Supports optional query parameters:
    - `query`: Text-based search in article content.
    - `ids`: Filter by article IDs (supports comma-separated string, JSON array, or repeated `ids` parameter).
    - `limit`: Maximum number of results (1–100, default: 50).

  **Examples:**
  - By IDs only (comma-separated):
    ```
    GET /api/articles?ids=1,2,5
    ```
  - By IDs (repeated parameters):
    ```
    GET /api/articles?ids=1&ids=2&ids=5
    ```
  - By IDs and text search:
    ```
    GET /api/articles?query=Rundetårn&ids=1,2&limit=10
    ```
  - Full-text search only:
    ```
    GET /api/articles?query=Denmark
    ```

  **Response:**  
  Returns structured search results including matched articles with metadata and scores.

#### Vector Search
- **`POST /api/vector-search/indexes/{index_name}/query`**
  - Perform semantic search on a specific vector index.
  - **Path Parameter:**
    - `index_name`: Name of the vector index to search.
  - **Request Body (JSON):**
    ```json
    {
      "query_text": "What is the capital of Denmark?",
      "top_k": 5
    }
    ```
    - `query_text`: The search query (required).
    - `top_k`: Number of top results to return (optional, default: 5).

  **Example Request:**
  ```http
  POST /api/vector-search/indexes/article_embeddings/query
  Content-Type: application/json

  {
    "query_text": "Scandinavian history",
    "top_k": 3
  }
  ```

  **Response:**  
  Returns a list of semantically similar documents with metadata and similarity scores.

#### List All Vector Indexes
- **`GET /api/vector-search/indexes`**
  - Retrieve metadata for all available vector indexes.
  - Example response:
    ```json
    [
      {
        "index_name": "article_embeddings",
        "embedding_model": "text-embedding-3-small",
        "dimension": 1536,
        "created_at": "2025-04-05T12:00:00Z"
      }
    ]
    ```

#### Get Metadata for a Specific Vector Index
- **`GET /api/vector-search/indexes/{index_name}`**
  - Retrieve metadata for a specific vector index.
  - **Path Parameter:**
    - `index_name`: The name of the vector index.
  - Returns details such as model used, dimension, and creation timestamp.

---

### API Documentation
Once the server is running, access auto-generated API documentation at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

The OpenAPI 3.1 specification is available at:
```
/openapi/openapi.yaml
```

You can generate clients in various languages using OpenAPI Generator:
```bash
openapi-generator-cli generate -i openapi/openapi.yaml -g <language> -o ./client
```
Replace `<language>` with your target (e.g., `python`, `typescript-fetch`, `java`).

## Development

This project uses a `Makefile` to streamline development tasks:

| Command                 | Description |
|-------------------------|-----------|
| `make install`          | Install dependencies |
| `make run`              | Start the API server |
| `make lint`             | Format code and fix lint issues (using Ruff) |
| `make lint-check`       | Check formatting and linting without applying fixes |
| `make static-type-check`| Run static type checking with Mypy |
| `make test`             | Run tests using Pytest |
| `make pr`               | Run all pre-PR checks (linting, type checking, testing) |
| `make help`             | Show all available commands |

## License
N/A
