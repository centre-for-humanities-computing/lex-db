"""
EXTRACT CORPUS AND CREATE CHUNKS

PURPOSE: Extract articles from database, clean them, and create semantic chunks

Features:
- Extract 161K+ articles from database
- Clean and validate articles
- Split into semantic sections
- Create overlapping chunks with sentence boundary alignment
- Save chunks to JSONL format
- Return chunks in memory

Usage:
    from create_chunks import create_chunks_from_database
    
    chunks = create_chunks_from_database()
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

# HTML metadata patterns to remove
METADATA_PATTERNS = [
    (r'Læs\s+mere\s+i\s+Lex\s*:\s*', 'læs_mere_text'),
    (r'<h[2-6][^>]*>\s*Læs\s+mere\s+i\s+Lex\s*</h[2-6]>.*?(?=<h[1-6]|$)', 'læs_mere_heading'),
    (r'<h[2-6][^>]*>\s*Se\s+også\s*</h[2-6]>.*?(?=<h[1-6]|$)', 'se_også_heading'),
    (r'<h[2-6][^>]*>\s*Relateret\s*</h[2-6]>.*?(?=<h[1-6]|$)', 'relateret_heading'),
    (r'<h[2-6][^>]*>\s*External\s+links?\s*</h[2-6]>.*?(?=<h[1-6]|$)', 'external_links_heading'),
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

def clean_html(html_content: str) -> str:
    """Clean HTML by removing metadata patterns"""
    if not html_content:
        return ""
    
    cleaned = html_content
    
    for pattern, name in METADATA_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def extract_text(html: str) -> str:
    """Extract plain text from HTML"""
    text = re.sub(r'<[^>]+>', '', html)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
    """Fetch all published articles from database"""
    try:
        from lex_db.database import get_db_connection
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, headword, clarification, xhtml as html, 
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
                    'xhtml': row[3],
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
            html = (row.get('xhtml') or row.get('html') or "").strip()
            permalink = (row.get('permalink') or "").strip()
            published_at = row.get('published_at')
            changed_at = row.get('changed_at')
            language = row.get('language') or 'da'
            
            if not headword:
                continue
            
            title = build_title(headword, clarification)
            html_clean = clean_html(html)
            text = extract_text(html_clean)
            
            text_words = len(text.split())
            if len(text) < MIN_TEXT_LENGTH or text_words < MIN_TEXT_WORDS or not html_clean.strip():
                continue
            
            url = build_url(permalink)
            
            article = {
                'id': article_id,
                'title': title,
                'url': url,
                'html': html_clean,
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
    """Remove HTML tags and normalize whitespace"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_unwanted_sections(html: str) -> str:
    """Remove unwanted sections from HTML"""
    if not html:
        return html
    
    unwanted_patterns = [
        r'<h[2-6][^>]*>\s*Læs\s+mere\s+i\s+Lex\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'<h[2-6][^>]*>\s*Se\s+også\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'<h[2-6][^>]*>\s*Relateret\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'<h[2-6][^>]*>\s*Eksterne\s+links?\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'<h[2-6][^>]*>\s*External\s+links?\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'<h[2-6][^>]*>\s*Det\s+sker\s*</h[2-6]>.*?(?=<h[1-6]|$)',
        r'Læs\s+mere\s+i\s+Lex\s*:\s*',
    ]
    
    for pattern in unwanted_patterns:
        html = re.sub(pattern, '', html, flags=re.IGNORECASE | re.DOTALL)
    
    return html


def recover_first_section(section_text: str, title: str) -> str:
    """Recover missing title parts at the start of the first section"""
    if not section_text or not title:
        return section_text
    
    section_text = section_text.strip()
    title = title.strip()
    
    section_words = section_text.split()
    if not section_words:
        return section_text
    
    first_section_word = section_words[0]
    first_section_word_lower = first_section_word.lower()
    
    if not first_section_word:
        return section_text
    
    connecting_words = {
        'er', 'var', 'blev', 'være', 'havde', 'har', 'ville', 'skulle', 'kan', 'må', 'vil',
        'og', 'men', 'eller', 'der', 'som', 'hvis', 'når', 'hvor', 'hvad', 'hvem',
        'i', 'på', 'til', 'fra', 'med', 'for', 'ved', 'uden', 'efter', 'før', 'over', 'under',
        'samt', 'fordi', 'da', 'eftersom', 'imens'
    }
    
    is_connecting_word = first_section_word[0].islower() and first_section_word_lower in connecting_words
    
    if is_connecting_word:
        title_words = title.split()
        for i, title_word in enumerate(title_words):
            if title_word.lower() == first_section_word_lower:
                if i > 0:
                    missing_parts = ' '.join(title_words[:i])
                    recovered_text = missing_parts + ' ' + section_text
                    return recovered_text
                return section_text
        
        recovered_text = title + ' ' + section_text
        return recovered_text
    
    if first_section_word[0].isupper():
        title_words = title.split()
        section_first_word_clean = first_section_word.rstrip('.,;:!?').lower()
        
        if title_words and title_words[0].rstrip('.,;:!?').lower() == section_first_word_clean:
            if len(title_words) > 1 and len(section_words) > 1:
                second_section_word = section_words[1].rstrip('.,;:!?').lower()
                if second_section_word in connecting_words:
                    recovered_text = title + ' ' + ' '.join(section_words[1:])
                    return recovered_text
    
    return section_text


def split_by_headings(html: str) -> List[Tuple[str, str]]:
    """Split HTML by heading tags (h2-h6)"""
    if not html:
        return []
    
    parts = re.split(r'(<h[2-6][^>]*>.*?</h[2-6]>)', html, flags=re.IGNORECASE | re.DOTALL)
    sections = []
    current_heading = ""
    
    for part in parts:
        if not part or not part.strip():
            continue
        
        if re.match(r'<h[2-6]', part, re.IGNORECASE):
            current_heading = clean_text(part)
        else:
            content = clean_text(part)
            if content:
                sections.append((current_heading, content))
                current_heading = ""
    
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
   Extract articles from database and create chunks
    
    Returns:
        List of chunks as dictionaries, or None if failed
    
    Example:
        from create_chunks import create_chunks_from_database
        
        chunks = create_chunks_from_database()
        if chunks:
            print(f"Created {len(chunks)} chunks")
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
    recovered = 0
    chunks_list = []
    
    for article_idx, article in enumerate(articles_data):
        try:
            html = article.get('html', '')
            title = article.get('title', '')
            
            if not html or not html.strip():
                skipped += 1
                continue
            
            html = remove_unwanted_sections(html)
            sections = split_by_headings(html)
            
            if not sections:
                skipped += 1
                continue
            
            total_sections += len(sections)
            
            for section_id, (heading, section_text) in enumerate(sections):
                if section_id == 0:
                    recovered_text = recover_first_section(section_text, article.get('title', ''))
                    if recovered_text != section_text:
                        recovered += 1
                    section_text = recovered_text
                
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
    