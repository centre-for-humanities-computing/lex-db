"""Utility functions for Lex DB."""

import logging
from typing import List

# Configure logger
logger = logging.getLogger("lex_db")

def get_logger():
    """Get the application logger."""
    return logger

def configure_logging(debug: bool = False):
    """Configure logging for the application.    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Set up console logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
        
    root_logger.addHandler(handler)
    
    # Configure lex_db logger
    lex_db_logger = logging.getLogger("lex_db")
    lex_db_logger.setLevel(level)

def split_document_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a document into chunks of text with specified size and overlap.    """
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
