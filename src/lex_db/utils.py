"""Utility functions for Lex DB."""

import json
import logging
import tiktoken
import re
from enum import Enum
from typing import Any
from markdownify import markdownify as md
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

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

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


# From here on, the code is for semantic chunking

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

    md_text = re.sub(r"(?s)#*Læs\s+mere\si\sLex.*?$", "", md_text, flags=re.IGNORECASE)
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

    logger_instance = get_logger()
    logger_instance.debug(f"Split into {len(sections)} sections.")
    return sections


def tokenize(text_input: str) -> list[str]:
    """Tokenize text by splitting on whitespace, keeping leading line breaks.

    Newlines that open `text_input` are carried as a prefix on the first
    token rather than emitted as a token of their own. The returned length is
    therefore identical to ``text_input.split()``, which is what keeps every
    chunk boundary in chunk_section exactly where it was before this change.
    """
    if not text_input:
        return []
    tokens = text_input.split()
    if not tokens:
        return []
    leading = text_input[: len(text_input) - len(text_input.lstrip("\n"))]
    if leading:
        tokens[0] = leading + tokens[0]
    return tokens


def reconstruct_text(tokens: list[str]) -> str:
    """Rebuild text with correct spacing for punctuation."""
    if not tokens:
        return ""
    NO_SPACE_BEFORE = {",", ".", "!", "?", ";", ":", ")", "]", "}", '"', "'"}
    NO_SPACE_AFTER = {"(", "[", "{", '"', "'"}
    result = tokens[0].lstrip("\n")
    for token in tokens[1:]:
        if token.startswith("\n"):
            result += token
        elif token in NO_SPACE_BEFORE or result[-1] in NO_SPACE_AFTER:
            result += token
        else:
            result += " " + token
    return result


def split_sentences_preserving_lines(text: str) -> list[str]:
    """Sentence-split `text` while recording where its line breaks were.

    SENTENCE_SPLITTER discards newlines and returns one bare sentence per
    line, which is what flattened every markdown table in the index onto a
    single line: a 26-row table arrived at the LLM as pipe-delimited soup
    whose only row boundary was a doubled "|".

    Splitting per line first preserves the line structure. Each sentence that
    opens a new line carries its newlines as a string prefix, which tokenize()
    folds into the first token, so sentence and token counts are unchanged and
    no chunk boundary moves.
    """
    if not text:
        return []

    sentences: list[str] = []
    pending = ""
    for line in text.split("\n"):
        if not line.strip():
            if sentences:
                pending = "\n\n"
            continue
        for position, sentence in enumerate(SENTENCE_SPLITTER.split(text=line)):
            if not sentence.strip():
                continue
            if not sentences:
                sentences.append(sentence)
            elif position == 0:
                sentences.append((pending or "\n") + sentence)
            else:
                sentences.append(sentence)
            pending = ""
    return sentences


# --- Article Metadata ("fact box") rewriting -------------------------------
#
# The metadata appendix is written at sync time by _format_metadata_appendix
# and stored inside articles.xhtml_md. It is rewritten here, at chunk time,
# rather than at the source, so that existing articles are fixed by a re-chunk
# instead of a full re-fetch of the corpus from lex.dk.

METADATA_SECTION_HEADING = "Article Metadata"

# Fields carrying no retrievable information. The id and URL already reach the
# LLM as separate fields, and the repeated Title is what made fact boxes
# outrank the article content they belong to.
METADATA_PLUMBING_FIELDS = frozenset(
    {"Article ID", "Title", "URL", "Last Modified", "Additional Metadata"}
)

_METADATA_FIELD_RE = re.compile(r"\*\*([^*]+?):\*\*[ \t]*([^\n]*)")

# Lex encodes an unknown day/month as zeros, e.g. "0.0.1723" for a year we only
# know approximately. Rendering that as a day would be wrong.
_YEAR_ONLY_DATE_RE = re.compile(r"^0\.0\.(\d{3,4})$")


def _format_danish_date(value: str) -> str | None:
    """Render a Lex date as a Danish date phrase, or None if unusable."""
    value = value.strip()
    if not value:
        return None
    year_only = _YEAR_ONLY_DATE_RE.match(value)
    if year_only:
        return f"i {year_only.group(1)}"
    return f"den {value}"


def _danish_life_sentence(title: str, fields: dict[str, str]) -> str | None:
    """Express birth and death metadata as a Danish sentence.

    The stored form ("**Death Date:** 22.12.1984") is unreachable from Danish:
    a query for "død" cannot match the English key, and for many biographical
    articles these dates appear nowhere in the prose. Danish stemming does
    match "død" against "døde", so the sentence form makes them findable.
    """
    birth = _format_danish_date(fields.get("Birth Date", ""))
    death = _format_danish_date(fields.get("Death Date", ""))
    birthplace = fields.get("Birthplace", "").strip()
    deathplace = fields.get("Place Of Death", "").strip()

    if not (birth or death or birthplace or deathplace):
        return None

    clauses = []
    if birth or birthplace:
        parts = ["blev født"]
        if birth:
            parts.append(birth)
        if birthplace:
            parts.append(f"i {birthplace}")
        clauses.append(" ".join(parts))
    if death or deathplace:
        parts = ["døde"]
        if death:
            parts.append(death)
        if deathplace:
            parts.append(f"i {deathplace}")
        clauses.append(" ".join(parts))

    return f"{title} " + " og ".join(clauses) + "."


def rewrite_metadata_section(content: str, doc_title: str) -> str:
    """Rewrite the metadata appendix into retrievable text.

    Drops the plumbing block, keeps every remaining field, and prepends the
    article title plus a Danish sentence for birth and death facts.

    Returns "" when nothing but plumbing was present, so that fact boxes
    carrying no information produce no chunk at all rather than a chunk
    echoing the article title.
    """
    fields: dict[str, str] = {}
    order: list[str] = []
    for match in _METADATA_FIELD_RE.finditer(content):
        key = match.group(1).strip()
        value = match.group(2).strip().strip("-").strip()
        if key in METADATA_PLUMBING_FIELDS or not value:
            continue
        if key not in fields:
            order.append(key)
        fields[key] = value

    if not fields:
        return ""

    lines: list[str] = [doc_title] if doc_title else []
    sentence = _danish_life_sentence(doc_title or "", fields)
    if sentence:
        lines.append(sentence)
    lines.extend(f"{key}: {fields[key]}" for key in order)
    return "\n".join(lines)


def is_table_chunk(text: str) -> bool:
    """Whether a finished chunk is tabular and needs context prepended.

    Detection is by pipe count rather than the "| --- |" separator or a
    leading "|", because:
      - the separator row appears only in the FIRST chunk of a split table
      - a leading "|" misses table chunks merged with preceding prose
      - a leading "|" stops working once this function's caller prepends
        the title and heading

    Measured on the production index: 100% of chunks containing the separator
    have >=4 pipes, versus 0.08% of chunks that do not.
    """
    return text.count("|") >= 4


def _apply_chunk_context(
    chunk: str,
    doc_title: str,
    section_heading: str,
    is_first_chunk: bool,
) -> str:
    """Prepend locating context to a chunk.

    Table chunks carry no indication of what they are about — a row like
    "| 12,09 | Masai Russell, USA | 2026 |" shares no vocabulary with the
    question it answers, so it is unreachable by both keyword and vector
    search. Those chunks get the article title and section heading prepended.

    Non-tabular chunks keep the pre-existing behaviour exactly.
    """
    if not chunk:
        return chunk

    if is_table_chunk(chunk):
        prefix = " ".join(part for part in (doc_title, section_heading) if part)
        return f"{prefix}\n{chunk}" if prefix else chunk

    # Unchanged legacy behaviour for prose: only continuation fragments
    # (those starting mid-sentence) inherit their heading.
    if section_heading and chunk[0].islower() and is_first_chunk:
        return section_heading + " " + chunk
    return chunk


def chunk_section(
    section_heading: str,
    section_text: str,
    min_chunk_size: int = 5,
    chunk_size: int = 250,
    overlap: int = 30,
    doc_title: str = "",
) -> list[str]:
    """Split one section into chunks with overlap and size limits."""
    if not section_text.strip():
        return []

    try:
        sentences = split_sentences_preserving_lines(section_text)
        if not sentences:
            return []

        sentence_tokens = [tokenize(s) for s in sentences]
        total_tokens = sum(len(t) for t in sentence_tokens)

        if total_tokens < min_chunk_size:
            return []
        if total_tokens < chunk_size:
            chunk = reconstruct_text([t for s in sentence_tokens for t in s])
            return [_apply_chunk_context(chunk, doc_title, section_heading, True)]

        chunks: list[str] = []
        current: list[list[str]] = []
        count: int = 0
        for sent_tokens in sentence_tokens:
            sent_len = len(sent_tokens)
            if count + sent_len >= chunk_size and count >= min_chunk_size:
                chunk_tokens = [t for s in current for t in s]
                chunk_text = reconstruct_text(chunk_tokens)
                chunks.append(
                    _apply_chunk_context(
                        chunk_text, doc_title, section_heading, not chunks
                    )
                )

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
            chunks.append(
                _apply_chunk_context(chunk_text, doc_title, section_heading, not chunks)
            )

        return chunks

    except Exception as e:
        logger_instance = get_logger()
        logger_instance.warning(f"Error chunking section: {e}")
        return []

    # ---------- End of helper functions ----------


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

    cleaned_text = clean_markdown(text)
    sections = split_text_by_sections_with_headings(cleaned_text)
    if not sections:
        return []

    # The article title arrives as a level-1 heading ("# " + headword, added by
    # vector_store.update_vector_index). Only the first section would otherwise
    # see it — every later "##" heading replaces it — so capture it here and
    # carry it into table chunks, which have no other way to say what they are.
    doc_title = ""
    for heading, _ in sections:
        if heading.startswith("#") and not heading.startswith("##"):
            doc_title = heading.lstrip("#").strip()
            break

    all_chunks = []
    for heading, content in sections:
        clean_heading = heading.lstrip("#").strip() if heading else ""
        if clean_heading == METADATA_SECTION_HEADING:
            content = rewrite_metadata_section(content, doc_title)
            if not content:
                continue
            # The rewritten text already opens with the article title, and
            # "Article Metadata" is an English label with no retrieval value.
            clean_heading = ""
        all_chunks.extend(
            chunk_section(
                clean_heading,
                content,
                min_chunk_size,
                chunk_size,
                overlap,
                doc_title=doc_title,
            )
        )

    logger_instance.debug(f"Split into {len(all_chunks)} semantic chunks.")
    return all_chunks


def _format_metadata_key(key: str) -> str:
    """
    Format metadata key for display.

    Converts snake_case to Title Case for better readability.
    """
    return key.replace("_", " ").title()


def _format_metadata_appendix(
    article_id: int,
    title: str,
    url: str | None,
    changed_at: str | None,
    metadata: dict[str, Any] | None,
) -> str:
    """
    Format article metadata as a Markdown section.

    Creates a structured metadata appendix with article information
    and any additional metadata fields.

    Args:
        article_id: The article ID
        title: The article title
        url: Optional article URL
        changed_at: Optional last modified timestamp
        metadata: Optional dictionary of additional metadata

    Returns:
        Formatted metadata section as Markdown string
    """
    lines = [
        "",
        "---",
        "",
        "## Article Metadata",
        "",
        f"**Article ID:** {article_id}",
        f"**Title:** {title}",
    ]

    if url:
        lines.append(f"**URL:** {url}")

    if changed_at:
        lines.append(f"**Last Modified:** {changed_at}")

    if metadata:
        lines.append("")
        lines.append("**Additional Metadata:**")
        for key, value in metadata.items():
            formatted_key = _format_metadata_key(key)
            lines.append(f"- **{formatted_key}:** {value}")

    return "\n".join(lines)


def convert_article_json_to_markdown(
    article_json: dict[str, Any] | str,
    include_metadata: bool = True,
    base_url: str = "https://lex.dk",
) -> str:
    """
    Convert lex.dk article JSON to Markdown format.

    Parses article JSON from lex.dk API responses and converts the HTML content
    to clean Markdown, preserving links, formatting, and semantic structure.
    Optionally appends article metadata as a structured section.

    Args:
        article_json: Article JSON dict or JSON string from lex.dk API
        include_metadata: Whether to append metadata section (default: True)
        base_url: Base URL for resolving relative links (default: "https://lex.dk")

    Returns:
        Markdown formatted article content with optional metadata appendix

    Raises:
        ValueError: If JSON is malformed or missing required fields (id, title, xhtml_body)

    Examples:
        >>> json_data = {"id": 12345, "title": "Test", "xhtml_body": "<p>Content</p>"}
        >>> markdown = convert_article_json_to_markdown(json_data)
        >>> print(markdown)
        Content
        <BLANKLINE>
        ---
        <BLANKLINE>
        ## Article Metadata
        ...
    """
    # Parse JSON if string provided
    if isinstance(article_json, str):
        try:
            article_data = json.loads(article_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON: {e}")
    else:
        article_data = article_json

    # Validate required fields
    required_fields = ["id", "title", "xhtml_body"]
    missing_fields = [field for field in required_fields if field not in article_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Extract fields
    article_id = article_data["id"]
    title = article_data["title"]
    xhtml_body = article_data["xhtml_body"]
    url = article_data.get("url")
    changed_at = article_data.get("changed_at")
    metadata = article_data.get("metadata")

    # Convert HTML to Markdown
    if not xhtml_body or not xhtml_body.strip():
        markdown_content = ""
    else:
        markdown_content = md(
            xhtml_body,
            heading_style="ATX",  # Use # for headings
            bullets="-",  # Use - for unordered lists
            strip=["script", "style"],  # Remove unwanted tags
            escape_asterisks=False,  # Preserve * in text
            escape_underscores=False,  # Preserve _ in text
        )

    # Append metadata if requested
    result = markdown_content
    if include_metadata:
        metadata_appendix = _format_metadata_appendix(
            article_id, title, url, changed_at, metadata
        )
        result = markdown_content + metadata_appendix

    return result.strip()


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
