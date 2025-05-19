"""Utility functions for Lex DB."""

from typing import List

def split_document_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a document into chunks of text with specified size and overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: The maximum size of each chunk in characters
        overlap: The number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
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
            
        start_index += (chunk_size - overlap)
        if start_index >= text_len:  # Ensure we don't create an empty chunk if overlap is large
            break
            
    return chunks
