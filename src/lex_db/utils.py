"""Utility functions for Lex DB."""

import logging
import tiktoken
import re
from enum import Enum

# Configure logger
logger = logging.getLogger("lex_db")


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""

    TOKENS = "tokens"
    CHARACTERS = "characters"
    SECTIONS = "sections"


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


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
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
    text: str, chunk_size: int, overlap: int, model: str = "text-embedding-3-small"
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


def split_text_by_sections(
    text: str, exclude_footer_pattern: str | None = r"(?s)#*Læs\s+mere\si\sLex.*?$"
) -> list[str]:
    """
    Split text into chunks based on Markdown sections (headings), excluding footers.
    """
    if not text.strip():
        return []

    # Step 1: Remove footer section (e.g. "Læs mere i Lex" and bullet list)
    if exclude_footer_pattern:
        text = re.sub(exclude_footer_pattern, "", text, flags=re.IGNORECASE)

    # Step 2: Split on level 1 and 2 Markdown headings (e.g., ## Section)
    # This regex captures: ## Heading\n or # Heading\n or ### Heading\n
    section_pattern = r"(#{1,3}\s+[^\n]+)"
    parts = re.split(section_pattern, text)
    chunks = []

    # Process alternating heading/content parts
    for i in range(1, len(parts)):
        if i % 2 == 1:  # It's a heading (from capture group)
            heading = parts[i].strip()
            content = "" if i + 1 >= len(parts) else parts[i + 1].strip()
            section_text = f"{heading}\n{content}".strip()
            chunks.append(section_text)

    # Clean up whitespace and duplicates
    cleaned_chunks = []
    seen = set()
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped:
            # Avoid duplicate chunks (e.g. repeated headings)
            if stripped not in seen:
                seen.add(stripped)
                cleaned_chunks.append(stripped)

    logger = get_logger()
    logger.debug(f"Split into {len(cleaned_chunks)} section chunks.")

    return cleaned_chunks


def split_document_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.TOKENS,
    model: str = "text-embedding-3-large",
) -> list[str]:
    """Split a document into chunks with specified method, size and overlap."""
    if chunking_strategy == ChunkingStrategy.TOKENS:
        return split_text_by_tokens(text, chunk_size, overlap, model)
    elif chunking_strategy == ChunkingStrategy.CHARACTERS:
        return split_text_by_characters(text, chunk_size, overlap)
    elif chunking_strategy == ChunkingStrategy.SECTIONS:
        return split_text_by_sections(text)
    else:
        raise ValueError(f"Unsupported chunking method: {chunking_strategy}")
