# lex-db
A repository for interacting with the lex database for the Lex AI project. This project provides a wrapper around a PostgreSQL database to enable querying encyclopedia articles via API requests, supporting both **vector (semantic) search** and **full-text/keyword search**.

## Features
- PostgreSQL database with pgvector extension for vector search
- Full-text search capabilities using PostgreSQL native FTS with Danish language support
- FastAPI-based REST API with automatic OpenAPI documentation
- Hybrid querying via metadata filtering and text search
- Vector index management and semantic search
- Connection pooling for efficient database access

## Requirements
- Python 3.12+
- [Astral UV](https://github.com/astral-sh/uv) for package management
- PostgreSQL 15+ with pgvector extension installed

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/centre-for-humanities-computing/lex-db 
cd lex-db
```

### 2. Install dependencies
```bash
make install
```

### 3. Set up PostgreSQL database
Ensure PostgreSQL is installed and the pgvector extension is available:
```bash
# Install PostgreSQL and pgvector (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-15-pgvector

# Or using Homebrew (macOS)
brew install postgresql pgvector
```

### 4. Configure environment variables
Create a `.env` file with your PostgreSQL connection details:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lex_db
DB_USER=lex_user
DB_PASSWORD=your_password_here
DB_SSLMODE=prefer

# Connection Pooling
DB_POOL_MIN_SIZE=2
DB_POOL_MAX_SIZE=10

# Application Settings
APP_NAME=Lex DB API
DEBUG=false

# OpenAI API (for embeddings, optional)
OPENAI_API_KEY=your_api_key_here
```

### 5. Initialize the database
Run the database setup script to create tables and indexes:
```bash
# Create database and user (run as postgres user)
sudo -u postgres psql <<EOF
CREATE DATABASE lex_db;
CREATE USER lex_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE lex_db TO lex_user;
\c lex_db
GRANT ALL ON SCHEMA public TO lex_user;
CREATE EXTENSION vector;
EOF
```

## Usage

### Database Migration from SQLite

If you're migrating from an existing SQLite database, follow these steps:

#### 1. Migrate Articles Table
Use pgloader to migrate the articles table:
```bash
# Run the migration script
cd db
./run_migration.sh
```

This will:
- Migrate all articles from SQLite to PostgreSQL
- Add full-text search (FTS) column and GIN index
- Verify the migration was successful

#### 2. Export Vector Indexes from SQLite (Optional)
If you have existing vector indexes in SQLite that you want to preserve:
```bash
# Export a vector index to JSONL format
uv run python -m scripts.export_sqlite_vector_index \
    --sqlite-db db/lex.db \
    --index-name e5_small \
    --output exports/e5_small.jsonl
```

#### 3. Create Vector Index in PostgreSQL
```bash
# Create an empty vector index structure
uv run python -m scripts.create_vector_index \
    --index-name e5_small \
    --embedding-model intfloat/multilingual-e5-small \
    --source-table articles \
    --source-column xhtml_md \
    --chunking-strategy semantic_chunks \
    --chunk-size 250 \
    --chunk-overlap 30
```

#### 4. Import Embeddings (Optional)
If you exported embeddings from SQLite:
```bash
# Import embeddings into PostgreSQL
uv run python -m scripts.import_vector_embeddings \
    --jsonl-file exports/e5_small.jsonl \
    --index-name e5_small \
    --batch-size 1000
```

Or generate new embeddings:
```bash
# Generate and populate embeddings from scratch
uv run python -m scripts.update_vector_indexes \
    --index-name e5_small \
    --batch-size 64
```

### Managing Vector Indexes

#### Create a New Vector Index
```bash
uv run python -m scripts.create_vector_index \
    --index-name my_index \
    --embedding-model text-embedding-3-small \
    --source-table articles \
    --source-column xhtml_md \
    --chunking-strategy semantic_chunks
```

Available embedding models:
- `intfloat/multilingual-e5-small` (384 dimensions, local)
- `intfloat/multilingual-e5-large` (1024 dimensions, local)
- `text-embedding-ada-002` (1536 dimensions, OpenAI)
- `text-embedding-3-small` (1536 dimensions, OpenAI)
- `text-embedding-3-large` (3072 dimensions, OpenAI)

Available chunking strategies:
- `tokens` - Token-based chunking using tiktoken
- `characters` - Simple character-based chunking
- `sections` - Markdown section-based chunking
- `semantic_chunks` - Danish sentence-aware semantic chunking (recommended)

#### Update Vector Index with New/Modified Articles
```bash
uv run python -m scripts.update_vector_indexes \
    --index-name my_index \
    --batch-size 64
```

This will:
- Detect new, modified, and deleted articles
- Remove outdated embeddings
- Generate and add new embeddings
- Update metadata timestamps

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
