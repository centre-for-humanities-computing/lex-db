"""Utility functions for Lex DB."""

import logging
import tiktoken
import re
from enum import Enum
from sentence_splitter import SentenceSplitter  # type: ignore

# Configure logger
logger = logging.getLogger("lex_db")

# Initialize sentence splitter for Danish
SENTENCE_SPLITTER = SentenceSplitter(language="da")

# Markdown metadata patterns to remove
METADATA_PATTERNS = [
    r"#{2,6}\s+Læs\s+mere\s+i\s+Lex.*?(?=#{1,6}\s|$)",
    r"#{2,6}\s+Se\s+også.*?(?=#{1,6}\s|$)",
    r"#{2,6}\s+Relateret.*?(?=#{1,6}\s|$)",
    r"#{2,6}\s+Eksterne\s+links?.*?(?=#{1,6}\s|$)",
    r"#{2,6}\s+External\s+links?.*?(?=#{1,6}\s|$)",
    r"#{2,6}\s+Det\s+sker.*?(?=#{1,6}\s|$)",
    r"Læs\s+mere\s+i\s+Lex\s*:\s*",
]


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""

    TOKENS = "tokens"
    CHARACTERS = "characters"
    SECTIONS = "sections"
    SEMANTIC_CHUNKS = "semantic_chunks"


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


def clean_markdown(markdown_content: str) -> str:
    """Clean markdown by removing unwanted sections."""
    if not markdown_content:
        return ""

    cleaned = markdown_content

    for pattern in METADATA_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r"\n\n\n+", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def split_text_by_sections(
    text: str, exclude_footer_pattern: str | None = r"(?s)#*Læs\s+mere\si\sLex.*?$"
) -> list[str]:
    """
    Split text into chunks based on Markdown sections (headings), excluding footers.
    Returns list of strings with heading and content combined.
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

    logger_instance = get_logger()
    logger_instance.debug(f"Split into {len(cleaned_chunks)} section chunks.")

    return cleaned_chunks


def split_text_by_semantic_chunks(
    text: str,
    chunk_size: int = 250,
    overlap: int = 30,
    model: str = "text-embedding-3-small",
    min_chunk_size: int = 5,
) -> list[str]:
    """
    Split text into semantic chunks using Danish sentence segmentation.
    Enforces min_chunk_size and chunk_size limits, with overlap between chunks.
    """

    if not text:
        return []

    logger_instance = get_logger()

    # ---------- Helper functions inside ----------

    def clean_markdown(markdown_content: str) -> str:
        """Remove Lex metadata sections and normalize whitespace."""
        if not markdown_content:
            return ""
        cleaned = markdown_content
        for pattern in METADATA_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r"\n\n\n+", "\n\n", cleaned).strip()

    def split_text_by_sections_with_headings(md_text: str) -> list[tuple[str, str]]:
        """Split markdown into (heading, content) pairs."""
        if not md_text.strip():
            return []

        md_text = re.sub(
            r"(?s)#*Læs\s+mere\si\sLex.*?$", "", md_text, flags=re.IGNORECASE
        )
        parts = re.split(r"(#{1,6}\s+[^\n]+)", md_text)

        sections, seen = [], set()
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue
            if re.match(r"#{1,6}\s+", part):
                heading = part
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                i += 2
            else:
                heading, content = "", part
                i += 1
            key = (heading, content)
            if key not in seen:
                seen.add(key)
                sections.append(key)

        logger_instance.debug(f"Split into {len(sections)} sections.")
        return sections

    def tokenize(text_input: str) -> list[str]:
        """Tokenize text by splitting on whitespace."""
        return text_input.split() if text_input else []

    def reconstruct_text(tokens: list[str]) -> str:
        """Rebuild text with correct spacing for punctuation."""
        if not tokens:
            return ""
        NO_SPACE_BEFORE = {",", ".", "!", "?", ";", ":", ")", "]", "}", '"', "'"}
        NO_SPACE_AFTER = {"(", "[", "{", '"', "'"}
        result = tokens[0]
        for token in tokens[1:]:
            if token in NO_SPACE_BEFORE or result[-1] in NO_SPACE_AFTER:
                result += token
            else:
                result += " " + token
        return result

    def chunk_section(section_heading: str, section_text: str) -> list[str]:
        """Split one section into chunks with overlap and size limits."""
        if not section_text.strip():
            return []

        try:
            sentences = SENTENCE_SPLITTER.split(text=section_text)
            if not sentences:
                return []

            sentence_tokens = [tokenize(s) for s in sentences]
            total_tokens = sum(len(t) for t in sentence_tokens)

            if total_tokens < min_chunk_size:
                return []
            if total_tokens < chunk_size:
                chunk = reconstruct_text([t for s in sentence_tokens for t in s])
                if section_heading and chunk and chunk[0].islower():
                    chunk = section_heading + " " + chunk
                return [chunk]

            chunks: list[str] = []
            current: list[list[str]] = []
            count: int = 0
            for sent_tokens in sentence_tokens:
                sent_len = len(sent_tokens)
                if count + sent_len >= chunk_size and count >= min_chunk_size:
                    chunk_tokens = [t for s in current for t in s]
                    chunk_text = reconstruct_text(chunk_tokens)
                    if (
                        section_heading
                        and chunk_text
                        and chunk_text[0].islower()
                        and not chunks
                    ):
                        chunk_text = section_heading + " " + chunk_text
                    chunks.append(chunk_text)

                    # Overlap logic
                    overlap_sentences: list[list[str]] = []
                    overlap_count: int = 0
                    for s in reversed(current):
                        slen = len(s)
                        if overlap_count + slen <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_count += slen
                        else:
                            break
                    current = overlap_sentences
                    count = overlap_count

                current.append(sent_tokens)
                count += sent_len

            # Last chunk
            if current and count >= min_chunk_size:
                chunk_tokens = [t for s in current for t in s]
                chunk_text = reconstruct_text(chunk_tokens)
                if (
                    section_heading
                    and chunk_text
                    and chunk_text[0].islower()
                    and not chunks
                ):
                    chunk_text = section_heading + " " + chunk_text
                chunks.append(chunk_text)

            return chunks

        except Exception as e:
            logger_instance.warning(f"Error chunking section: {e}")
            return []

    # ---------- End of helper functions ----------

    cleaned_text = clean_markdown(text)
    sections = split_text_by_sections_with_headings(cleaned_text)
    if not sections:
        return []

    all_chunks = []
    for heading, content in sections:
        clean_heading = heading.lstrip("#").strip() if heading else ""
        all_chunks.extend(chunk_section(clean_heading, content))

    logger_instance.debug(f"Split into {len(all_chunks)} semantic chunks.")
    return all_chunks


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
    elif chunking_strategy == ChunkingStrategy.SEMANTIC_CHUNKS:
        return split_text_by_semantic_chunks(text, chunk_size, overlap, model)
    else:
        raise ValueError(f"Unsupported chunking method: {chunking_strategy}")
