"""
EXTRACT CORPUS FROM DATABASE AND CREATE SEMANTIC CHUNKS FROM MARKDOWN

PURPOSE: Extract articles from database using xhtml_md (markdown) field, 
         clean them, and create semantic chunks

Features:
- Extract 161K+ articles from database
- Work with markdown content (xhtml_md field)
- Split into semantic sections using markdown headings (# to ######)
- Create overlapping chunks with sentence boundary alignment
- Return chunks in memory

Usage:
    from create_chunks import create_chunks_from_database
    chunks = create_chunks_from_database()
"""

import os
os.environ['DATABASE_URL'] = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'lex_1.2.2.db')
import re
from typing import Tuple, List, Optional, Dict, Any

# ===============================================================
# CONFIGURATION
# ===============================================================

# Extraction parameters
MIN_TEXT_LENGTH = 10
MIN_TEXT_WORDS = 3

# Chunking parameters
SMALL_CHUNK_SIZE = 250
CHUNK_OVERLAP = 30
REWIND_LIMIT = 50
MIN_CHUNK_SIZE = 5

# Markdown metadata patterns to remove
METADATA_PATTERNS = [
    r'#{2,6}\s+Læs\s+mere\s+i\s+Lex.*?(?=#{1,6}\s|$)',
    r'#{2,6}\s+Se\s+også.*?(?=#{1,6}\s|$)',
    r'#{2,6}\s+Relateret.*?(?=#{1,6}\s|$)',
    r'#{2,6}\s+Eksterne\s+links?.*?(?=#{1,6}\s|$)',
    r'#{2,6}\s+External\s+links?.*?(?=#{1,6}\s|$)',
    r'#{2,6}\s+Det\s+sker.*?(?=#{1,6}\s|$)',
    r'Læs\s+mere\s+i\s+Lex\s*:\s*',
]

# Danish abbreviations
ABBREVIATIONS = {
    'bl.a.', 'dvs.', 'f.eks.', 'etc.', 'ca.', 'mht.', 'nr.', 'kap.', 'art.', 'stk.', 'al.', 'forts.',
    'dir.', 'vej.', 'skt.', 'sml.', 'udg.', 'red.', 'opr.', 'bearb.',
    'genv.', 'osv.', 'igennem', 'omkring',
    'u.s.', 'u.s.a.', 'u.k.', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
    'inc.', 'ltd.', 'co.', 'corp.', 'vs.', 'no.', 'pp.', 'eg.', 'ie.', 'etc',
}


# ===============================================================
# EXTRACTION HELPER FUNCTIONS
# ===============================================================

def clean_markdown(markdown_content: str) -> str:
    """Clean markdown by removing unwanted sections"""
    if not markdown_content:
        return ""
    
    cleaned = markdown_content
    
    for pattern in METADATA_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def build_title(headword: str, clarification: Optional[str]) -> str:
    """Build title from headword and clarification"""
    if not headword:
        return ""
    
    headword = headword.strip()
    clarification = (clarification or "").strip()
    
    if not clarification or clarification.upper() == "NULL":
        return headword
    
    return f"{headword} ({clarification})"


def build_url(permalink: Optional[str]) -> str:
    """Build URL from permalink"""
    if not permalink:
        return ""
    
    permalink = permalink.strip()
    if not permalink:
        return ""
    
    permalink = permalink.lstrip('/')
    return f"https://lex.dk/{permalink}"


def fetch_articles_from_database() -> Optional[List[Dict[str, Any]]]:
    """Fetch all published articles from database with markdown content"""
    try:
        from lex_db.database import get_db_connection
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, headword, clarification, xhtml_md as markdown, 
                       permalink, published_at, changed_at, language
                FROM articles
                WHERE state = 'published'
            """)
            
            articles = []
            for row in cursor.fetchall():
                articles.append({
                    'id': row[0],
                    'headword': row[1],
                    'clarification': row[2],
                    'markdown': row[3],
                    'permalink': row[4],
                    'published_at': row[5],
                    'changed_at': row[6],
                    'language': row[7]
                })

            return articles if articles else None
    
    except Exception as e:
        print(f"ERROR: Failed to fetch from database: {e}")
        return None


def extract_corpus(articles_raw: List[Dict]) -> List[Dict]:
    """
    Extract and clean corpus from raw articles
    
    Returns:
        List of cleaned article dictionaries
    """
    articles_data = []
    
    for idx, row in enumerate(articles_raw):
        try:
            article_id = row.get('id')
            headword = (row.get('headword') or "").strip()
            clarification = (row.get('clarification') or "").strip()
            markdown = (row.get('markdown') or "").strip()
            permalink = (row.get('permalink') or "").strip()
            published_at = row.get('published_at')
            changed_at = row.get('changed_at')
            language = row.get('language') or 'da'
            
            if not headword:
                continue
            
            title = build_title(headword, clarification)
            markdown_clean = clean_markdown(markdown)
            
            # Extract plain text for validation
            text = re.sub(r'[#\*\[\]`_~]', '', markdown_clean)
            text = re.sub(r'\s+', ' ', text).strip()
            
            text_words = len(text.split())
            if len(text) < MIN_TEXT_LENGTH or text_words < MIN_TEXT_WORDS or not markdown_clean.strip():
                continue
            
            url = build_url(permalink)
            
            article = {
                'id': article_id,
                'title': title,
                'url': url,
                'content': markdown_clean,  # Markdown content
                'language': language,
                'published_at': published_at,
                'changed_at': changed_at,
            }
            
            articles_data.append(article)
        
        except Exception as e:
            print(f"WARNING: Error processing article at index {idx}: {str(e)}")
            continue
    
    return articles_data


# ===============================================================
# CHUNKING HELPER FUNCTIONS
# ===============================================================

def tokenize(text: str) -> List[str]:
    """Split text into tokens"""
    if not text:
        return []
    return text.split()


def clean_text(text: str) -> str:
    """Clean and normalize whitespace from text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_unwanted_sections(markdown: str) -> str:
    """Remove unwanted sections from markdown"""
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


def split_by_headings(markdown: str) -> List[Tuple[str, str]]:
    """Split markdown by heading markers (# through ######)"""
    if not markdown:
        return []
    
    # Split by markdown headings (# through ######)
    parts = re.split(r'(#{1,6}\s+[^\n]*)', markdown, flags=re.IGNORECASE)
    sections = []
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Skip empty parts
        if not part or not part.strip():
            i += 1
            continue
        
        # Check if this is a heading (starts with 1-6 hashes)
        if re.match(r'#{1,6}\s+', part, re.IGNORECASE):
            current_heading = clean_text(part)
            
            # Get the next part as content
            content = ""
            if i + 1 < len(parts):
                content = clean_text(parts[i + 1])
                i += 2
            else:
                i += 1
            
            if content:
                sections.append((current_heading, content))
        else:
            # This is content without a heading
            content = clean_text(part)
            if content:
                sections.append(("", content))
            i += 1
    
    return sections


def reconstruct_text(tokens: List[str]) -> str:
    """Reconstruct text from tokens with proper punctuation spacing"""
    if not tokens:
        return ""
    
    NO_SPACE_BEFORE = {',', '.', '!', '?', ';', ':', ')', ']', '}', '"', "'"}
    NO_SPACE_AFTER = {'(', '[', '{', '"', "'"}
    
    result = tokens[0]
    
    for token in tokens[1:]:
        if token in NO_SPACE_BEFORE or (result and result[-1] in NO_SPACE_AFTER):
            result += token
        else:
            result += ' ' + token
    
    return result


def is_sentence_end(token: str, next_token: Optional[str] = None) -> bool:
    """Check if token ends a sentence"""
    if not token or not re.search(r'[.!?;]+$', token):
        return False
    
    if token.lower() in ABBREVIATIONS:
        return False
    
    if next_token and len(next_token) > 0:
        if next_token[0].isalpha():
            if next_token[0].islower():
                return False
        else:
            return False
    
    return True


def find_chunk_boundary(tokens: List[str], target_idx: int) -> int:
    """Find nearest sentence boundary for chunk alignment"""
    if target_idx >= len(tokens):
        return len(tokens)
    
    if target_idx < 0:
        return 0
    
    rewind_limit = max(0, target_idx - REWIND_LIMIT)
    
    for i in range(target_idx, rewind_limit - 1, -1):
        if i < 0 or i >= len(tokens):
            continue
        
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        
        if is_sentence_end(token, next_token):
            return i + 1
    
    minimum_overlap_start = target_idx - CHUNK_OVERLAP
    
    if minimum_overlap_start >= 0:
        return minimum_overlap_start
    else:
        return 0


def chunk_section(section_text: str, section_heading: str, section_id: int, article: Dict) -> List[Dict]:
    """Create chunks from a single semantic section"""
    
    if not section_text or not section_text.strip():
        return []
    
    section_text = section_text.strip()
    tokens = tokenize(section_text)
    
    if len(tokens) == 0:
        return []
    
    if len(tokens) < MIN_CHUNK_SIZE:
        text = reconstruct_text(tokens)
        return [{
            'article_id': article['id'],
            'chunk_id': f"{article['id']}.{section_id}.0",
            'title': article.get('title', ''),
            'section_heading': section_heading,
            'chunk_text': text,
            'context_text': text,
            'section_id': section_id,
            'chunk_start_pos': 0,
            'chunk_end_pos': len(tokens),
            'url': article.get('url', ''),
            'language': article.get('language', 'da'),
            'published_at': article.get('published_at', ''),
            'changed_at': article.get('changed_at', ''),
            'token_count': len(tokens),
        }]
    
    chunks = []
    chunk_idx = 0
    start_idx = 0
    
    full_section_text = reconstruct_text(tokens)
    if section_heading:
        section_context = section_heading + "\n" + full_section_text
    else:
        section_context = full_section_text
    
    while start_idx < len(tokens):
        theoretical_end_idx = min(start_idx + SMALL_CHUNK_SIZE, len(tokens))
        end_idx = find_chunk_boundary(tokens, theoretical_end_idx)
        
        if end_idx - start_idx < MIN_CHUNK_SIZE:
            end_idx = min(start_idx + MIN_CHUNK_SIZE, len(tokens))
        
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = reconstruct_text(chunk_tokens)
        
        if not chunk_text.strip():
            start_idx = end_idx
            continue
        
        final_chunk_text = chunk_text.strip()
        if chunk_idx == 0 and section_heading:
            final_chunk_text = section_heading + " " + final_chunk_text
        
        chunk = {
            'article_id': article['id'],
            'chunk_id': f"{article['id']}.{section_id}.{chunk_idx}",
            'title': article.get('title', ''),
            'section_heading': section_heading,
            'chunk_text': final_chunk_text,
            'context_text': section_context.strip(),
            'section_id': section_id,
            'chunk_start_pos': start_idx,
            'chunk_end_pos': end_idx,
            'url': article.get('url', ''),
            'language': article.get('language', 'da'),
            'published_at': article.get('published_at', ''),
            'changed_at': article.get('changed_at', ''),
            'token_count': len(chunk_tokens),
        }
        
        chunks.append(chunk)
        chunk_idx += 1
        
        if end_idx >= len(tokens):
            break
        
        next_start_theoretical = start_idx + (SMALL_CHUNK_SIZE - CHUNK_OVERLAP)
        new_start_idx = find_chunk_boundary(tokens, next_start_theoretical)
        
        if new_start_idx <= start_idx:
            new_start_idx = min(start_idx + 1, len(tokens))
        
        start_idx = new_start_idx
    
    return chunks


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def create_chunks_from_database() -> Optional[List[Dict]]:
    """
    Extract articles from database using markdown content and create chunks
    
    Returns:
        List of chunks as dictionaries, or None if failed
    
    Example:
        from create_chunks_markdown import create_chunks_from_database
        
        chunks = create_chunks_from_database()
    """
    
    # STEP 1: Fetch articles from database
    articles_raw = fetch_articles_from_database()
    
    if not articles_raw:
        print("ERROR: Could not fetch articles from database")
        return None
    
    # STEP 2: Extract and clean corpus
    
    articles_data = extract_corpus(articles_raw)
    
    if not articles_data:
        print("ERROR: No articles extracted")
        return None
    
    
    # STEP 3: Create chunks
    
    total_chunks = 0
    total_sections = 0
    skipped = 0
    chunks_list = []
    
    for article_idx, article in enumerate(articles_data):
        try:
            markdown = article.get('content', '')
            title = article.get('title', '')
            
            if not markdown or not markdown.strip():
                skipped += 1
                continue
            
            markdown = remove_unwanted_sections(markdown)
            sections = split_by_headings(markdown)
            
            if not sections:
                skipped += 1
                continue
            
            total_sections += len(sections)
            
            for section_id, (heading, section_text) in enumerate(sections):
                section_chunks = chunk_section(
                    section_text=section_text,
                    section_heading=heading,
                    section_id=section_id,
                    article=article
                )
                
                for chunk in section_chunks:
                    chunks_list.append(chunk)
                    total_chunks += 1
        
        except Exception as e:
            print(f"WARNING: Error processing article {article_idx}: {e}")
            skipped += 1
            continue
    
    return chunks_list if chunks_list else None


if __name__ == "__main__":
    chunks = create_chunks_from_database()
    if chunks:
        print(f"Successfully created {len(chunks):,} chunks")
    else:
        print("Failed to create chunks")
