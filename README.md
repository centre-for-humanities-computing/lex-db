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

### Running the API Server

```bash
make run
```

This will start the FastAPI server at `http://0.0.0.0:8000`.

### API Endpoints

- `GET /`: Health check endpoint
- `GET /api/tables`: List all tables in the database
- `POST /api/vector-search`: Does top-k search on a given vector index for a given query text

### API Documentation

Once the server is running, you can access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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
