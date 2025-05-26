"""Embedding generation utilities for vector search in Lex DB."""

from enum import Enum
import numpy as np
from src.lex_db.utils import get_logger

logger = get_logger()


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    LOCAL_E5_MULTILINGUAL = "local_e5"
    OPENAI_ADA_002 = "openai_ada_002"
    MOCK_MODEL = "mock_model"


def get_embedding_dimensions(model_choice: EmbeddingModel) -> int:
    """Get the dimension of embeddings for a given model."""
    if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        return 1024
    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        return 1536
    elif model_choice == EmbeddingModel.MOCK_MODEL:
        return 4  # A small, fixed dimension for testing
    else:
        raise ValueError(f"Unknown model: {model_choice}")


# Module-level cache for loaded models
_model_cache: dict[EmbeddingModel, object] = {}


def get_local_embedding_model(model_choice: EmbeddingModel) -> object:
    """Get a cached embedding model instance."""
    if model_choice not in _model_cache:
        if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_choice}")
            _model_cache[model_choice] = SentenceTransformer(
                "intfloat/multilingual-e5-large-instruct"
            )
        else:
            raise ValueError(f"Local model not supported: {model_choice}")

    return _model_cache[model_choice]


def generate_embeddings(
    texts: list[str], model_choice: EmbeddingModel
) -> list[list[float]]:
    """Generate embeddings for a list of texts using the specified model."""
    if model_choice == EmbeddingModel.MOCK_MODEL:  # Add this block
        logger.debug(f"Generating MOCK embeddings for {len(texts)} texts")
        # Return a list of random dummy embeddings for testing
        return [
            np.random.random_sample(
                get_embedding_dimensions(EmbeddingModel.MOCK_MODEL)
            ).tolist()
            for _ in texts
        ]

    elif model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        model = get_local_embedding_model(EmbeddingModel.LOCAL_E5_MULTILINGUAL)
        formatted_texts = [f"passage: {text}" for text in texts]

        logger.debug(
            f"Generating embeddings for {len(texts)} texts using E5 Multilingual model"
        )
        # Use getattr to safely access encode method
        encode_method = getattr(model, "encode", None)
        if encode_method is None:
            raise ValueError(f"No encode method for local model: {model_choice}")

        embeddings = encode_method(formatted_texts, normalize_embeddings=True)

        if hasattr(embeddings, "tolist"):
            embeddings_list = embeddings.tolist()
            return [list(map(float, emb)) for emb in embeddings_list]

        return [list(map(float, emb)) for emb in embeddings]

    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        try:
            from openai import OpenAI
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
                )

            client = OpenAI(api_key=api_key)
            logger.info(
                f"Generating embeddings for {len(texts)} texts using OpenAI API"
            )

            response = client.embeddings.create(
                input=texts, model="text-embedding-ada-002"
            )

            return [item.embedding for item in response.data]

        except ImportError:
            raise ImportError("OpenAI package not installed. Install with 'uv sync'.")
        except Exception as e:
            raise ValueError(f"Error generating OpenAI embeddings: {str(e)}")

    else:
        raise ValueError(f"Unsupported embedding model: {model_choice}")


def generate_query_embedding(
    query_text: str, model_choice: EmbeddingModel
) -> list[float]:
    """Generate embedding for a search query."""
    return generate_embeddings([query_text], model_choice)[0]
