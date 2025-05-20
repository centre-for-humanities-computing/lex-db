"""Embedding generation utilities for vector search in Lex DB."""

from enum import Enum

from lex_db.utils import get_logger

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


def get_embedding_model(model_choice: EmbeddingModel) -> object:
    """Get a cached embedding model instance."""
    if model_choice not in _model_cache:
        if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_choice}")
            _model_cache[model_choice] = SentenceTransformer(
                "intfloat/multilingual-e5-large-instruct"
            )

    return _model_cache[model_choice]


def generate_embeddings(
    texts: list[str], model_choice: EmbeddingModel
) -> list[list[float]]:
    """Generate embeddings for a list of texts using the specified model."""
    if model_choice == EmbeddingModel.MOCK_MODEL:  # Add this block
        logger.info(f"Generating MOCK embeddings for {len(texts)} texts")
        # Return a list of unique dummy embeddings for testing
        return [
            [float(i + 1) / 10.0] * get_embedding_dimensions(model_choice)
            for i, _ in enumerate(texts)
        ]

    elif model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        model = get_embedding_model(model_choice)
        formatted_texts = [f"passage: {text}" for text in texts]

        logger.info(
            f"Generating embeddings for {len(texts)} texts using E5 Multilingual model"
        )
        # Use getattr to safely access encode method
        encode_method = getattr(model, "encode", None)
        if encode_method is None:
            # Fallback mechanism if encode method doesn't exist
            return [[0.0] * get_embedding_dimensions(model_choice) for _ in texts]

        embeddings = encode_method(formatted_texts, normalize_embeddings=True)

        # Convert to list if not already a list
        if hasattr(embeddings, "tolist"):
            embeddings_list = embeddings.tolist()
            return [list(map(float, emb)) for emb in embeddings_list]
        # Ensure we're returning the correct type
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
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'."
            )
        except Exception as e:
            raise ValueError(f"Error generating OpenAI embeddings: {str(e)}")

    else:
        raise ValueError(f"Unsupported embedding model: {model_choice}")


def generate_query_embedding(
    query_text: str, model_choice: EmbeddingModel
) -> list[float]:
    """Generate embedding for a search query."""
    if model_choice == EmbeddingModel.MOCK_MODEL:
        logger.info(f"Generating MOCK query embedding for: {query_text}")
        # Return a fixed dummy query embedding
        return [0.1, 0.2, 0.3, 0.4][: get_embedding_dimensions(model_choice)]

    elif model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        model = get_embedding_model(model_choice)
        formatted_query = f"query: {query_text}"

        # Use getattr to safely access encode method
        encode_method = getattr(model, "encode", None)
        if encode_method is None:
            # Fallback mechanism if encode method doesn't exist
            return [0.0] * get_embedding_dimensions(model_choice)

        embedding = encode_method(formatted_query, normalize_embeddings=True)

        # Convert to list if not already a list
        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
            return list(map(float, embedding_list))
        # Ensure we're returning the correct type
        return list(map(float, embedding))
    else:
        return generate_embeddings([query_text], model_choice)[0]
