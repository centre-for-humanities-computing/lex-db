# lex-db

A PostgreSQL-backed API for the Lex AI project. Provides vector (semantic) search, full-text search, and article management for Danish encyclopedia articles from [lex.dk](https://lex.dk).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
  - [Database Migration (SQLite → PostgreSQL)](#database-migration-sqlite--postgresql)
  - [Managing Vector Indexes](#managing-vector-indexes)
  - [Syncing Articles from lex.dk](#syncing-articles-from-lexdk)
- [Development](#development)
- [Project Structure](#project-structure)

## Features

- PostgreSQL + pgvector for vector similarity search (HNSW indexes)
- PostgreSQL native full-text search with Danish language support
- FastAPI REST API with auto-generated OpenAPI 3.1 docs
- Batch vector and full-text search endpoints
- Multiple embedding models: local (E5, Jina) and OpenAI
- ONNX Runtime with INT8 quantization for CPU; native transformers for GPU
- Automatic GPU detection (CUDA)
- Connection pooling via psycopg
- Article synchronization from lex.dk sitemaps

## Requirements

- Python 3.12+
- [Astral UV](https://github.com/astral-sh/uv) for package management
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/centre-for-humanities-computing/lex-db
cd lex-db
make install

# 2. Create PostgreSQL database and enable pgvector
sudo -u postgres psql <<EOF
CREATE DATABASE lex_db;
CREATE USER lex_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE lex_db TO lex_user;
\c lex_db
GRANT ALL ON SCHEMA public TO lex_user;
CREATE EXTENSION IF NOT EXISTS vector;
EOF

# 3. Copy and edit the environment file
cp .env.example .env
# Edit .env with your database credentials

# 4. Initialize metadata tables
PGPASSWORD=your_password psql -U lex_user -d lex_db -f db/schema.sql

# 5. Run the API
make run
```

API docs available at `http://localhost:8000/docs` (Swagger) and `http://localhost:8000/redoc` (ReDoc).

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `lex_db` | Database name |
| `DB_USER` | `lex_user` | Database user |
| `DB_PASSWORD` | — | Database password |
| `DB_POOL_MIN_SIZE` | `2` | Min connection pool size |
| `DB_POOL_MAX_SIZE` | `10` | Max connection pool size |
| `GPU_BATCH_SIZE` | `64` | Batch size for GPU inference |
| `CPU_BATCH_SIZE` | `16` | Batch size for CPU inference |
| `CPU_NUM_THREADS` | `4` | ONNX Runtime thread count |
| `INFERENCE_SERVER_URL` | — | Remote inference server (optional) |
| `INFERENCE_SERVER_ENABLED` | `True` | Use remote server when available |
| `INFERENCE_SERVER_XAUTH` | — | Auth token for inference server |
| `OPENAI_API_KEY` | — | For OpenAI embedding models |
| `OPENROUTER_API_KEY` | — | For HyDE query expansion (optional) |
| `SITEMAP_RATE_LIMIT` | `10` | Max concurrent sitemap requests |
| `DEBUG` | `false` | Enable debug logging & hot reload |

**GPU support** is detected automatically via CUDA. No configuration needed — if a CUDA-capable GPU is available, models load in full-precision transformer mode. Otherwise, ONNX Runtime with INT8 quantization is used for CPU.

## Usage

### Running the API

```bash
make run          # Production mode
make run-dev      # Development mode (generates OpenAPI spec first)
```

The server listens on `0.0.0.0:8000` by default. Override with `DEPLOY_DOMAIN` and `DEPLOY_PORT` env vars.

### API Endpoints

#### Health
| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check with DB info |

#### Articles
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/articles?ids=1,2&query=Denmark&limit=10` | Full-text search with optional ID filter |

Supports `ids` as comma-separated, repeated (`?ids=1&ids=2`), or JSON array.

#### Vector Search
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/vector-search/indexes/{name}/query` | Single semantic search |
| `POST` | `/api/vector-search/indexes/{name}/batch` | Batch semantic search |
| `GET` | `/api/vector-search/indexes` | List all vector indexes |
| `GET` | `/api/vector-search/indexes/{name}` | Get index metadata |

#### Full-Text Search on Chunks
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/text-search/indexes/{name}/batch` | Batch FTS on vector index chunks |

#### Deprecated Endpoints
| Method | Path | Replacement |
|---|---|---|
| `POST` | `/api/hybrid-search/indexes/{name}/query` | Use batch semantic + batch FTS instead |
| `POST` | `/api/hyde-search/indexes/{name}/query` | Use batch semantic search instead |

#### Benchmarking
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/benchmark/embeddings` | Benchmark embedding generation speed |

#### Other
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tables` | List database tables |

### Database Migration (SQLite → PostgreSQL)

If migrating from the old SQLite database:

```bash
# 1. Migrate articles table via pgloader
cd db
./run_migration.sh

# 2. (Optional) Export vector indexes from SQLite
uv run python -m scripts.export_sqlite_vector_index \
    --sqlite-db db/lex_1.5.0.db \
    --index-name e5_small \
    --output exports/e5_small.jsonl

# 3. Create vector index in PostgreSQL
uv run python -m scripts.create_vector_index \
    --index-name e5_small \
    --embedding-model intfloat/multilingual-e5-small \
    --source-table articles \
    --source-column xhtml_md \
    --chunking-strategy semantic_chunks

# 4. Import embeddings or generate new ones
uv run python -m scripts.import_vector_embeddings \
    --jsonl-file exports/e5_small.jsonl \
    --index-name e5_small
```

### Managing Vector Indexes

#### Create an Index

```bash
uv run python -m scripts.create_vector_index \
    --index-name my_index \
    --embedding-model intfloat/multilingual-e5-small \
    --source-table articles \
    --source-column xhtml_md \
    --chunking-strategy semantic_chunks \
    --chunk-size 512 \
    --chunk-overlap 50
```

Use `--force-drop` to replace an existing index.

**Embedding models:**

| Model ID | Dims | Type |
|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | Local |
| `intfloat/multilingual-e5-large` | 1024 | Local |
| `jinaai/jina-embeddings-v5-text-small` | 1024 | Local |
| `jinaai/jina-embeddings-v5-text-nano` | 768 | Local |
| `text-embedding-ada-002` | 1536 | OpenAI |
| `text-embedding-3-small` | 1536 | OpenAI |
| `text-embedding-3-large` | 3072 | OpenAI |

**Chunking strategies:** `tokens`, `characters`, `sections`, `semantic_chunks` (Danish-aware, recommended).

#### Update an Index

```bash
uv run python -m scripts.update_vector_indexes \
    --index-name my_index \
    --batch-size 2048
```

Detects new/modified/deleted articles and updates embeddings accordingly. Uses metadata stored at index creation time — no need to re-specify chunking or model parameters.

### Syncing Articles from lex.dk

```bash
# Sync all encyclopedias
uv run python -m scripts.sync_articles

# Dry run (preview only)
uv run python -m scripts.sync_articles --dry-run

# Specific encyclopedias
uv run python -m scripts.sync_articles --encyclopedia-ids 14,15,18
```

- Deletions are skipped if not all sitemaps fetched successfully (prevents data loss)
- Vector indexes are automatically updated after article changes

## Development

```bash
make install              # Install dependencies
make install-dev          # Install with dev dependencies
make run                  # Start API server
make run-dev              # Start with OpenAPI spec generation
make lint                 # Format + fix with Ruff
make lint-check           # Check formatting/linting (no fix)
make static-type-check    # Mypy type checking
make test                 # Run pytest
make pr                   # All pre-PR checks
make generate-openapi-schema  # Write OpenAPI spec to openapi/openapi.yaml
```

## Project Structure

```
├── main.py                  # FastAPI app entry point
├── generate_openapi.py      # OpenAPI spec generator
├── pyproject.toml           # Dependencies & tool config
├── Makefile
├── .env.example             # Environment template
├── db/
│   ├── schema.sql           # Metadata tables & triggers
│   ├── migrate_articles.load # pgloader config
│   └── run_migration.sh     # SQLite→PostgreSQL migration
├── openapi/
│   └── openapi.yaml         # Generated OpenAPI 3.1 spec
├── scripts/                 # Deployment scripts
└── src/
    ├── lex_db/
    │   ├── api/routes.py    # API endpoints
    │   ├── database.py      # DB connection pool & queries
    │   ├── config.py        # Pydantic settings
    │   ├── embeddings.py    # Model loading & embedding generation
    │   ├── vector_store.py  # pgvector index CRUD & search
    │   ├── advanced_search.py # HyDE, RRF fusion (partially deprecated)
    │   ├── sitemap.py       # lex.dk sitemap parsing
    │   └── utils.py         # Chunking, logging, markdown conversion
    ├── scripts/             # CLI tools (sync, index management, etc.)
    └── tests/               # Test suite
```

## License

N/A
