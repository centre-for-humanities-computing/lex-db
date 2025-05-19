# Makefile for the Lex DB project

.PHONY: install run static-type-check lint lint-check test pr help

# Default target
default: help

install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv sync

run:
	@echo "--- ▶️ Running the application ---"
	uv run main.py

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	uv run mypy .

lint:
	@echo "--- 🧹 Formatting and linting codebase ---"
	uv run ruff format .
	uv run ruff check . --fix

lint-check:
	@echo "--- 🧐 Checking if project is formatted and linted ---"
	uv run ruff format . --check
	uv run ruff check .

test:
	@echo "--- 🧪 Running tests ---"
	# Pytest will discover tests in the project (e.g., in a 'tests/' directory or files named test_*.py)
	uv run pytest

pr: static-type-check lint-check test
	@echo "--- ✅ All PR checks passed successfully ---"
	@echo "Ready to make a PR!"

help:
	@echo "Makefile for the Lex DB project"
	@echo ""
	@echo "Available commands:"
	@echo "  make install             Install project dependencies using uv sync"
	@echo "  make run                 Run the FastAPI application using 'uv run main.py'"
	@echo "  make static-type-check   Run static type checking with mypy on the current directory"
	@echo "  make lint                Format code with Ruff and apply lint fixes"
	@echo "  make lint-check          Check formatting with Ruff and run lint checks without fixing"
	@echo "  make test                Run tests with pytest"
	@echo "  make pr                  Run all pre-PR checks: static-type-check, lint-check, and test"
	@echo "  make help                Show this help message"

