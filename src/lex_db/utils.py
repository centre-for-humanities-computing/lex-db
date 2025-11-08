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


def split_text_by_semantic_chunks(
    text: str, chunk_size: int, overlap: int, model: str = "text-embedding-3-small"
) -> list[str]:
    """Split text into chunks based on semantic boundaries with sentence detection."""
    if not text:
        return []

    # Danish abbreviations for sentence detection
    abbreviations = {
        'bl.a.', 'dvs.', 'f.eks.', 'etc.', 'ca.', 'mht.', 'nr.', 'kap.', 'art.', 'stk.', 'al.', 'forts.',
        'dir.', 'vej.', 'skt.', 'sml.', 'udg.', 'red.', 'opr.', 'bearb.',
        'genv.', 'osv.', 'igennem', 'omkring',
        'u.s.', 'u.s.a.', 'u.k.', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
        'inc.', 'ltd.', 'co.', 'corp.', 'vs.', 'no.', 'pp.', 'eg.', 'ie.', 'etc',
    }

    # Configuration parameters
    min_chunk_size = 5
    rewind_limit = 50
    chunk_overlap = 30

    # Helper: Clean and normalize whitespace
    def clean_text(text_input):
        if not text_input:
            return ""
        text_input = re.sub(r'\s+', ' ', text_input).strip()
        return text_input

    # Helper: Remove unwanted sections from markdown
    def remove_unwanted_sections(markdown):
        if not markdown:
            return markdown
        
        unwanted_patterns = [
            r'#{2,6}\s+Læs\s+mere\s+i\s+Lex.*?(?=#{1,6}\s|$)',
            r'#{2,6}\s+Se\s+også.*?(?=#{1,6}\s|$)',
            r'#{2,6}\s+Relateret.*?(?=#{1,6}\s|$)',
            r'#{2,6}\s+Eksterne\s+links?.*?(?=#{1,6}\s|$)',
            r'#{2,6}\s+External\s+links?.*?(?=#{1,6}\s|$)',
            r'#{2,6}\s+Det\s+sker.*?(?=#{1,6}\s|$)',
            r'Læs\s+mere\s+i\s+Lex\s*:\s*',
        ]
        
        for pattern in unwanted_patterns:
            markdown = re.sub(pattern, '', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        return markdown

    # Helper: Split markdown by heading markers
    def split_by_headings(markdown):
        if not markdown:
            return []
        
        parts = re.split(r'(#{1,6}\s+[^\n]*)', markdown, flags=re.IGNORECASE)
        sections = []
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            if not part or not part.strip():
                i += 1
                continue
            
            if re.match(r'#{1,6}\s+', part, re.IGNORECASE):
                current_heading = clean_text(part)
                
                content = ""
                if i + 1 < len(parts):
                    content = clean_text(parts[i + 1])
                    i += 2
                else:
                    i += 1
                
                if content:
                    sections.append((current_heading, content))
            else:
                content = clean_text(part)
                if content:
                    sections.append(("", content))
                i += 1
        
        return sections

    # Helper: Check if token ends a sentence
    def is_sentence_end(token, next_token=None):
        if not token or not re.search(r'[.!?;]+$', token):
            return False
        if token.lower() in abbreviations:
            return False
        if next_token and len(next_token) > 0:
            if next_token[0].isalpha():
                if next_token[0].islower():
                    return False
            else:
                return False
        return True

    # Helper: Find nearest sentence boundary
    def find_chunk_boundary(tokens, target_idx):
        if target_idx >= len(tokens):
            return len(tokens)
        if target_idx < 0:
            return 0
        
        rewind_threshold = max(0, target_idx - rewind_limit)
        
        for i in range(target_idx, rewind_threshold - 1, -1):
            if i < 0 or i >= len(tokens):
                continue
            token = tokens[i]
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            if is_sentence_end(token, next_token):
                return i + 1
        
        minimum_overlap_start = target_idx - chunk_overlap
        return minimum_overlap_start if minimum_overlap_start >= 0 else 0

    # Helper: Reconstruct text from tokens with proper punctuation spacing
    def reconstruct_text(tokens):
        if not tokens:
            return ""
        
        no_space_before = {',', '.', '!', '?', ';', ':', ')', ']', '}', '"', "'"}
        no_space_after = {'(', '[', '{', '"', "'"}
        
        result = tokens[0]
        for token in tokens[1:]:
            if token in no_space_before or (result and result[-1] in no_space_after):
                result += token
            else:
                result += ' ' + token
        
        return result

    # Helper: Chunk a single section
    def chunk_section(section_text, section_heading):
        if not section_text or not section_text.strip():
            return []
        
        section_text = section_text.strip()
        tokens = section_text.split()
        
        if len(tokens) == 0:
            return []
        
        if len(tokens) < min_chunk_size:
            text = reconstruct_text(tokens)
            return [text]
        
        chunks = []
        chunk_idx = 0
        start_idx = 0
        
        full_section_text = reconstruct_text(tokens)
        if section_heading:
            section_context = section_heading + "\n" + full_section_text
        else:
            section_context = full_section_text
        
        while start_idx < len(tokens):
            theoretical_end_idx = min(start_idx + chunk_size, len(tokens))
            end_idx = find_chunk_boundary(tokens, theoretical_end_idx)
            
            if end_idx - start_idx < min_chunk_size:
                end_idx = min(start_idx + min_chunk_size, len(tokens))
            
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = reconstruct_text(chunk_tokens)
            
            if not chunk_text.strip():
                start_idx = end_idx
                continue
            
            final_chunk_text = chunk_text.strip()
            if chunk_idx == 0 and section_heading:
                final_chunk_text = section_heading + " " + final_chunk_text
            
            chunks.append(final_chunk_text)
            chunk_idx += 1
            
            if end_idx >= len(tokens):
                break
            
            next_start_theoretical = start_idx + (chunk_size - overlap)
            new_start_idx = find_chunk_boundary(tokens, next_start_theoretical)
            
            if new_start_idx <= start_idx:
                new_start_idx = min(start_idx + 1, len(tokens))
            
            start_idx = new_start_idx
        
        return chunks

    # Main processing workflow
    markdown = text
    markdown = remove_unwanted_sections(markdown)
    sections = split_by_headings(markdown)
    
    if not sections:
        return []
    
    all_chunks = []
    
    for section_heading, section_text in sections:
        section_chunks = chunk_section(section_text, section_heading)
        all_chunks.extend(section_chunks)
    
    logger_instance = get_logger()
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