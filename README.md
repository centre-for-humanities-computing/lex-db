# lex-db

A repository for interacting with the lex database for the Lex AI project. This project provides a wrapper around a SQLite database that is used to read from the database using API requests, implementing both vector search and normal full-text/keyword search for encyclopedia articles.

## Features

- SQLite database access with `sqlite-vec` for vector search
- Full-text search capabilities
- FastAPI endpoints for querying the database
- Hybrid search combining vector and keyword search

## Requirements

- Python 3.12+
- Astral UV for package management
- SQLite with `sqlite-vec` extension

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

3. Create a `.env` file (or modify the existing one):
   ```
   # Database settings
   DATABASE_URL=PATH TO DB
   ```

## Usage

### Scripts

- `create_fts_index.py`: Creates a new full-text search index for a specified column in a given table.
  ```bash
  uv run src/scripts/create_fts_index.py <table_name> <column_name>
  ```

- `create_vector_index.py`: Creates a new vector index for semantic search.
  ```bash
  uv run src/scripts/create_vector_index.py <table_name> <column_name>
  ```

- `update_vector_indexes.py`: Updates or populates vector indexes with embeddings using the OpenAI API.
  ```bash
  uv run src/scripts/update_vector_indexes.py
  ```

Note: While `create_openai_embedding_batches.py` and `add_batch_embeddings_to_index.py` exist for batch processing with OpenAI's API, they are not recommended due to service reliability issues. Instead, use the scripts mentioned above.

### Running the API Server

```bash
make run
```

This will start the FastAPI server at `http://0.0.0.0:8000`.

### API Endpoints

- `GET /api/tables`: List all tables in the database
  
- `POST /api/search`: Full-text search for articles
  ```
  {
    "query": string,
    "limit": int (optional, default: 50, max: 100)
  }
  ```
  Returns an array of matching documents sorted by relevance.

- `POST /api/vector-search`: Vector search using embeddings
  ```
  {
    "vector_index_name": string,
    "query_text": string,
    "embedding_model_choice": string,
    "top_k": int (optional, default: 5)
  }
  ```
  Returns semantically similar documents using vector search.

### API Documentation

Once the server is running, you can access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The API specification is available in OpenAPI 3.1 format at `/openapi/openapi.yaml`. You can use this specification with the OpenAPI Generator to create API clients in various languages:

```bash
openapi-generator-cli generate -i openapi/openapi.yaml -g <language> -o ./client
```

Replace `<language>` with your target language (e.g., typescript-fetch, python, java).

## Development

This project uses a `Makefile` to simplify common development tasks. Here are some of the available commands:

- `make install`: Install project dependencies.
- `make run`: Run the FastAPI application.
- `make lint`: Format code with Ruff and apply lint fixes.
- `make lint-check`: Check formatting and linting without applying fixes.
- `make static-type-check`: Run static type checking with Mypy.
- `make test`: Run tests with Pytest.
- `make pr`: Run all pre-PR checks (linting, type checking, testing).
- `make help`: Show all available Makefile commands.

## License

N/A
