"""Embedding generation utilities for vector search in Lex DB."""

import json
from enum import Enum
from typing import List

class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    LOCAL_SENTENCE_TRANSFORMER = "local_st"
    OPENAI_ADA_002 = "openai_ada_002"

def get_embedding_dimensions(model_choice: str) -> int:
    """Get the dimension of embeddings for a given model.
    
    Args:
        model_choice: The model to use for embedding generation
        
    Returns:
        The dimension of the embeddings
        
    Raises:
        ValueError: If the model is not supported
    """
    if model_choice == EmbeddingModel.LOCAL_SENTENCE_TRANSFORMER:
        return 384  # Example dimension for a common ST model
    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        return 1536  # Dimension for text-embedding-ada-002
    else:
        raise ValueError(f"Unknown model: {model_choice}")

def generate_embeddings(texts: List[str], model_choice: str) -> List[List[float]]:
    """Generate embeddings for a list of texts using the specified model.
    
    Args:
        texts: List of texts to generate embeddings for
        model_choice: The model to use for embedding generation
        
    Returns:
        List of embeddings, where each embedding is a list of floats
        
    Raises:
        ValueError: If the model is not supported
    """
    if model_choice == EmbeddingModel.LOCAL_SENTENCE_TRANSFORMER:
        # In production, replace with actual model loading and encoding:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        # embeddings = model.encode(texts)
        # return embeddings.tolist()
        
        # For now, use placeholder implementation
        print(f"Generating embeddings for {len(texts)} texts using local Sentence Transformer")
        dummy_dimension = get_embedding_dimensions(model_choice)
        return [[0.0] * dummy_dimension for _ in texts]  # Dummy embeddings
        
    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        # In production, replace with actual API call:
        # from openai import OpenAI
        # api_key = os.getenv("OPENAI_API_KEY")
        # client = OpenAI(api_key=api_key)
        # response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        # return [item.embedding for item in response.data]
        
        # For now, use placeholder implementation
        print(f"Generating embeddings for {len(texts)} texts using OpenAI API")
        dummy_dimension = get_embedding_dimensions(model_choice)
        return [[0.0] * dummy_dimension for _ in texts]  # Dummy embeddings
        
    else:
        raise ValueError(f"Unsupported embedding model: {model_choice}")
