"""Utility functions for Lex DB."""

import logging
import tiktoken
from enum import Enum

# Configure logger
logger = logging.getLogger("lex_db")


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""

    TOKENS = "tokens"
    CHARACTERS = "characters"


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logger


def configure_logging(debug: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    root_logger.addHandler(handler)

    lex_db_logger = logging.getLogger("lex_db")
    lex_db_logger.setLevel(level)


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """Count tokens in text using OpenAI's tokenizer."""
    try:
        # Get encoding for the model
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to character-based estimation (rough approximation: 1 token ≈ 4 characters)
        logger.warning(f"Error counting tokens: {e}. Using character-based estimation.")
        return len(text) // 4


def split_text_by_tokens(
    text: str, chunk_size: int, overlap: int, model: str = "text-embedding-ada-002"
) -> list[str]:
    """Split text into chunks based on token count."""
    if not text:
        return []
    encoding = tiktoken.encoding_for_model(model)

    # Encode the entire text
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end_idx == len(tokens):
            break

        start_idx += chunk_size - overlap
        if start_idx >= len(tokens):
            break

    return chunks


def split_text_by_characters(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks based on character count."""
    if not text:
        return []

    chunks = []
    start_index = 0
    text_len = len(text)

    while start_index < text_len:
        end_index = min(start_index + chunk_size, text_len)
        chunks.append(text[start_index:end_index])

        if end_index == text_len:
            break

        start_index += chunk_size - overlap
        if start_index >= text_len:
            break

    return chunks


def split_document_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    method: ChunkingStrategy = ChunkingStrategy.TOKENS,
    model: str = "text-embedding-ada-002",
) -> list[str]:
    """Split a document into chunks with specified method, size and overlap."""
    if method == ChunkingStrategy.TOKENS:
        return split_text_by_tokens(text, chunk_size, overlap, model)
    elif method == ChunkingStrategy.CHARACTERS:
        return split_text_by_characters(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")
