"""Embedding generation utilities for vector search in Lex DB."""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import time
from collections import deque
from threading import Lock
import tiktoken
from lex_db.utils import get_logger

logger = get_logger()


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    LOCAL_E5_MULTILINGUAL = "intfloat/multilingual-e5-large"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_SMALL_003 = "text-embedding-3-small"
    OPENAI_LARGE_003 = "text-embedding-3-large"
    MOCK_MODEL = "mock_model"


def get_embedding_dimensions(model_choice: EmbeddingModel) -> int:
    """Get the dimension of embeddings for a given model."""
    if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        return 1024
    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        return 1536
    elif model_choice == EmbeddingModel.OPENAI_SMALL_003:
        return 1536
    elif model_choice == EmbeddingModel.OPENAI_LARGE_003:
        return 3072
    elif model_choice == EmbeddingModel.MOCK_MODEL:
        return 4  # A small, fixed dimension for testing
    else:
        raise ValueError(f"Unknown model: {model_choice}")


# Rate limiting for OpenAI API
class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls."""

    def __init__(
        self, max_requests_per_minute: int = 9000, max_tokens_per_minute: int = 40000000
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_times: deque[float] = deque()
        self.token_usage: deque[tuple[float, int]] = deque()
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, texts: list[str]) -> int:
        """Count total tokens in a list of texts."""
        return sum(len(self.tokenizer.encode(text)) for text in texts)

    def wait_if_needed(self, texts: list[str]) -> None:
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            tokens_needed = self.count_tokens(texts)

            # Clean old entries (older than 1 minute)
            minute_ago = current_time - 60
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            while self.token_usage and self.token_usage[0][0] < minute_ago:
                self.token_usage.popleft()

            # Check request rate limit
            if len(self.request_times) >= self.max_requests_per_minute:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    logger.info(
                        f"Rate limit: sleeping {sleep_time:.2f}s for request limit"
                    )
                    time.sleep(sleep_time)
                    current_time = time.time()

            # Check token rate limit
            current_tokens = sum(usage[1] for usage in self.token_usage)
            if current_tokens + tokens_needed > self.max_tokens_per_minute:
                if self.token_usage:
                    sleep_time = 60 - (current_time - self.token_usage[0][0])
                    if sleep_time > 0:
                        logger.info(
                            f"Rate limit: sleeping {sleep_time:.2f}s for token limit"
                        )
                        time.sleep(sleep_time)
                        current_time = time.time()

            # Record this request
            self.request_times.append(current_time)
            self.token_usage.append((current_time, tokens_needed))

            logger.debug(
                f"Rate limiter: {len(self.request_times)} requests, "
                f"{sum(usage[1] for usage in self.token_usage)} tokens in last minute"
            )


# Global rate limiter instance
_openai_rate_limiter = OpenAIRateLimiter()

# Module-level cache for loaded models
_model_cache: dict[EmbeddingModel, object] = {}


def get_local_embedding_model(model_choice: EmbeddingModel) -> object:
    """Get a cached embedding model instance."""
    if model_choice not in _model_cache:
        if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_choice}")
            _model_cache[model_choice] = SentenceTransformer(
                EmbeddingModel.LOCAL_E5_MULTILINGUAL.value
            )
        else:
            raise ValueError(f"Local model not supported: {model_choice}")

    return _model_cache[model_choice]


def create_optimal_request_batches(
    texts: list[str], max_tokens_per_batch: int = 8000
) -> list[list[str]]:
    """Create batches that don't exceed the token limit."""

    tokenizer = tiktoken.get_encoding("cl100k_base")

    batches = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = len(tokenizer.encode(text))

        # If single text exceeds limit, truncate it
        if text_tokens > max_tokens_per_batch:
            logger.warning(f"Text with {text_tokens} tokens exceeds limit, truncating")
            tokens = tokenizer.encode(text)[:max_tokens_per_batch]
            text = tokenizer.decode(tokens)
            text_tokens = max_tokens_per_batch

        # If adding this text would exceed the limit, start a new batch
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens

    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)

    return batches


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

    elif model_choice in [
        EmbeddingModel.OPENAI_ADA_002,
        EmbeddingModel.OPENAI_SMALL_003,
        EmbeddingModel.OPENAI_LARGE_003,
    ]:
        try:
            from openai import OpenAI
            import os
            from concurrent.futures import ThreadPoolExecutor, as_completed

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")

            client = OpenAI(api_key=api_key)
            model_name = model_choice.value

            batches = create_optimal_request_batches(texts, max_tokens_per_batch=8000)
            logger.info(f"Split into {len(batches)} batches for parallel processing.")

            all_results: List[Optional[List[List[float]]]] = [None] * len(batches)

            def process_batch(
                batch_idx_batch: Tuple[int, List[str]],
            ) -> Tuple[int, Optional[List[List[float]]]]:
                batch_idx, batch = batch_idx_batch
                try:
                    _openai_rate_limiter.wait_if_needed(batch)
                    response = client.embeddings.create(input=batch, model=model_name)
                    embeddings = [item.embedding for item in response.data]
                    logger.debug(
                        f"Batch {batch_idx} completed with {len(embeddings)} embeddings."
                    )
                    return batch_idx, embeddings
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    return batch_idx, None

            max_workers = min(32, len(batches) + 4)  # Don't over-provision
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_batch, (i, batch))
                    for i, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    batch_idx, result = future.result()
                    if result is not None:
                        all_results[batch_idx] = result
            # Flatten results in correct order
            all_embeddings = []
            for result in all_results:
                if result is not None:
                    all_embeddings.extend(result)

            logger.info(
                f"Generated {len(all_embeddings)} embeddings from {len(batches)} batches."
            )
            return all_embeddings
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
    if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        model = get_local_embedding_model(EmbeddingModel.LOCAL_E5_MULTILINGUAL)
        # E5 queries MUST use "query: " prefix to match document space
        formatted_query = f"query: {query_text}"
        embedding = model.encode(formatted_query, normalize_embeddings=True)  # type: ignore[attr-defined]
        if hasattr(embedding, "tolist"):
            return embedding.tolist()  # type: ignore[no-any-return]
        return list(map(float, embedding))

    result = generate_embeddings([query_text], model_choice)[0]
    return list(map(float, result))


def generate_passage_embedding(
    passage_text: str, model_choice: EmbeddingModel
) -> list[float]:
    """Generate embedding for a passage/document (used for HyDE)."""
    if model_choice == EmbeddingModel.LOCAL_E5_MULTILINGUAL:
        model = get_local_embedding_model(EmbeddingModel.LOCAL_E5_MULTILINGUAL)
        # E5 passages MUST use "passage: " prefix
        formatted_passage = f"passage: {passage_text}"
        embedding = model.encode(formatted_passage, normalize_embeddings=True)  # type: ignore[attr-defined]
        if hasattr(embedding, "tolist"):
            return embedding.tolist()  # type: ignore[no-any-return]
        return list(map(float, embedding))

    result = generate_embeddings([passage_text], model_choice)[0]
    return list(map(float, result))
