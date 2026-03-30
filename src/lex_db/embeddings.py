"""Embedding generation utilities for vector search in Lex DB."""

from enum import Enum
import os
from typing import Protocol, runtime_checkable
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer  # type: ignore
from optimum.onnxruntime.configuration import AutoQuantizationConfig  # type: ignore
import onnxruntime as ort  # type: ignore
import torch
from torch.nn.functional import normalize
from transformers.modeling_utils import PreTrainedModel  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore
from transformers.models.auto.modeling_auto import AutoModel  # type: ignore
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore
import numpy as np
import time
from collections import deque
from threading import Lock
import tiktoken
from sentence_transformers import SentenceTransformer
from lex_db.utils import get_logger
from lex_db.config import get_settings
from pathlib import Path

logger = get_logger()


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    LOCAL_MULTILINGUAL_E5_SMALL = "intfloat/multilingual-e5-small"
    LOCAL_MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    JINA_V5_SMALL = "jinaai/jina-embeddings-v5-text-small"
    JINA_V5_NANO = "jinaai/jina-embeddings-v5-text-nano"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_SMALL_003 = "text-embedding-3-small"
    OPENAI_LARGE_003 = "text-embedding-3-large"
    MOCK_MODEL = "mock_model"


class TextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


def get_embedding_dimensions(model_choice: EmbeddingModel) -> int:
    """Get the dimension of embeddings for a given model."""
    dimensions = {
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL: 384,
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE: 1024,
        EmbeddingModel.OPENAI_ADA_002: 1536,
        EmbeddingModel.OPENAI_SMALL_003: 1536,
        EmbeddingModel.OPENAI_LARGE_003: 3072,
        EmbeddingModel.JINA_V5_SMALL: 1024,
        EmbeddingModel.JINA_V5_NANO: 768,
        EmbeddingModel.MOCK_MODEL: 4,
    }
    if model_choice not in dimensions:
        raise ValueError(f"Unknown model: {model_choice}")
    return dimensions[model_choice]


# =============================================================================
# Rate Limiting for OpenAI API
# =============================================================================


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


# =============================================================================
# Model Handler Pattern
# =============================================================================


@runtime_checkable
class EmbeddingModelHandler(Protocol):
    """Protocol defining the interface for model-specific embedding operations."""

    def load_model(self, use_gpu: bool) -> tuple:
        """Load model and tokenizer. Returns (model, tokenizer, use_gpu)."""
        ...

    def compute_embeddings(
        self,
        model: PreTrainedModel | ORTModelForFeatureExtraction | SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[tuple[str, TextType]],
    ) -> list[list[float]]:
        """Compute embeddings for texts."""
        ...


class E5ModelHandler:
    """Handler for multilingual E5 models (small and large).

    E5 models use:
    - "query: " prefix for queries
    - "passage: " prefix for passages
    - Mean pooling with attention mask
    - INT8 quantized ONNX for CPU inference
    """

    def __init__(self, model_choice: EmbeddingModel):
        self.model_choice = model_choice
        self.model_name = model_choice.value

    def _format_texts(self, texts: list[tuple[str, TextType]]) -> list[str]:
        """Format texts with E5-specific prefixes."""
        return [f"{text_type.value}: {text}" for text, text_type in texts]

    def load_model(self, use_gpu: bool) -> tuple:
        """Load E5 model for GPU or CPU inference."""
        if use_gpu:
            return self._load_for_gpu()
        else:
            return self._load_for_cpu()

    def _load_for_gpu(self) -> tuple:
        """Load transformers model for GPU inference."""
        device = "cuda:0"
        logger.info(f"Loading E5 model for GPU: {self.model_name}")

        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        logger.info(f"E5 model loaded on {device}")
        return model, tokenizer, True

    def _load_for_cpu(self) -> tuple:
        """Load ONNX model optimized for CPU inference with INT8 quantization."""
        settings = get_settings()
        num_threads = settings.CPU_NUM_THREADS

        sess_options = self._create_session_options(num_threads)
        cache_dir = get_onnx_cache_dir()
        model_cache_path = cache_dir / self.model_name.replace("/", "_")
        quantized_model_path = (
            cache_dir / f"{self.model_name.replace('/', '_')}_quantized"
        )

        # Check if quantized model exists
        if (
            quantized_model_path.exists()
            and (quantized_model_path / "model_quantized.onnx").exists()
        ):
            logger.info(
                f"Loading quantized E5 ONNX model from cache: {quantized_model_path}"
            )
            model = ORTModelForFeatureExtraction.from_pretrained(
                quantized_model_path,
                file_name="model_quantized.onnx",
                provider="CPUExecutionProvider",
                session_options=sess_options,
                provider_options={
                    "CPUExecutionProvider": {
                        "arena_extend_strategy": "kSameAsRequested"
                    }
                },
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            return model, tokenizer, False

        # Check if regular ONNX model exists
        if model_cache_path.exists() and (model_cache_path / "model.onnx").exists():
            logger.info(f"Loading E5 ONNX model from cache: {model_cache_path}")
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_cache_path,
                provider="CPUExecutionProvider",
                session_options=sess_options,
                provider_options={
                    "CPUExecutionProvider": {
                        "arena_extend_strategy": "kSameAsRequested"
                    }
                },
            )
            tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

            # Quantize and save
            model, tokenizer = self._quantize_model(
                model, tokenizer, quantized_model_path
            )
            return model, tokenizer, False

        # Export and quantize
        logger.info(f"Exporting E5 ONNX model to: {model_cache_path}")
        model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_name,
            export=True,
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model.save_pretrained(model_cache_path)
        tokenizer.save_pretrained(model_cache_path)
        logger.info(f"E5 ONNX model saved to cache: {model_cache_path}")

        # Quantize
        model, tokenizer = self._quantize_model(model, tokenizer, quantized_model_path)
        return model, tokenizer, False

    def _create_session_options(self, num_threads: int) -> ort.SessionOptions:
        """Create optimized ONNX Runtime session options."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        return sess_options

    def _quantize_model(
        self,
        model: ORTModelForFeatureExtraction,
        tokenizer: PreTrainedTokenizerBase,
        quantized_model_path: Path,
    ) -> tuple:
        """Quantize model to INT8 and save."""
        logger.info("Quantizing E5 model to INT8 for faster inference...")
        quantizer = ORTQuantizer.from_pretrained(model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        quantized_model_path.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=quantized_model_path,
            quantization_config=qconfig,
            file_suffix="quantized",
        )
        tokenizer.save_pretrained(quantized_model_path)

        # Reload quantized model
        sess_options = self._create_session_options(get_settings().CPU_NUM_THREADS)
        model = ORTModelForFeatureExtraction.from_pretrained(
            quantized_model_path,
            file_name="model_quantized.onnx",
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}
            },
        )
        logger.info(f"Quantized E5 model saved to: {quantized_model_path}")
        return model, tokenizer

    def compute_embeddings(
        self,
        model: PreTrainedModel | ORTModelForFeatureExtraction | SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[tuple[str, TextType]],
    ) -> list[list[float]]:
        """Compute embeddings using mean pooling."""

        formatted_texts = self._format_texts(texts)

        if isinstance(model, PreTrainedModel):
            return self._compute_embeddings_gpu(model, tokenizer, formatted_texts)
        elif isinstance(model, ORTModelForFeatureExtraction):
            return self._compute_embeddings_onnx(model, tokenizer, formatted_texts)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def _compute_embeddings_onnx(
        self,
        model: ORTModelForFeatureExtraction,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
    ) -> list[list[float]]:
        """Compute mean-pooled normalized embeddings using ONNX."""
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]

            # Mean pooling with attention mask
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # L2 normalize
            embeddings = normalize(embeddings, p=2, dim=1)

        return list(embeddings.numpy().tolist())

    def _compute_embeddings_gpu(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
        device: str = "cuda:0",
    ) -> list[list[float]]:
        """Compute mean-pooled normalized embeddings using transformers on GPU."""
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        # Move inputs to GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]

            # Mean pooling with attention mask
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # L2 normalize
            embeddings = normalize(embeddings, p=2, dim=1)

        return list(embeddings.cpu().numpy().tolist())


class JinaV5ModelHandler:
    """Handler for Jina v5 models (small and nano).

    Jina v5 models use:
    - "Query: " prefix for queries (retrieval task)
    - "Document: " prefix for passages (retrieval task)
    - Last-token pooling (critical for Jina v5)
    - Specialized ONNX models from 'onnx' subfolder for CPU
    - model.encode() API with task="retrieval" for GPU
    """

    def __init__(self, model_choice: EmbeddingModel):
        self.model_choice = model_choice
        self.model_name = model_choice.value
        # Jina provides specialized retrieval models for ONNX
        self.onnx_model_name = f"{model_choice.value}-retrieval"

    def _format_texts(self, texts: list[tuple[str, TextType]]) -> list[str]:
        """Format texts with Jina-specific prefixes for retrieval task."""
        formatted = []
        for text, text_type in texts:
            if text_type == TextType.QUERY:
                formatted.append(f"Query: {text}")
            else:
                formatted.append(f"Document: {text}")
        return formatted

    def load_model(self, use_gpu: bool) -> tuple:
        """Load Jina v5 model for GPU or CPU inference."""
        if use_gpu:
            return self._load_for_gpu()
        else:
            return self._load_for_cpu()

    def _load_for_gpu(self) -> tuple:
        """Load SentenceTransformer model for GPU inference."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Jina v5 model: {self.model_name}, on {device}")

        model_kwargs: dict = {"trust_remote_code": True}

        model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs=model_kwargs,
        )

        # SentenceTransformer doesn't need a separate tokenizer
        tokenizer = None

        logger.info(f"Jina v5 model loaded on {device}")
        return model, tokenizer, True

    def _load_for_cpu(self) -> tuple:
        """Load specialized Jina ONNX model for CPU inference."""
        settings = get_settings()
        num_threads = settings.CPU_NUM_THREADS

        sess_options = self._create_session_options(num_threads)
        cache_dir = get_onnx_cache_dir()
        # Use separate cache path for Jina retrieval models
        model_cache_path = cache_dir / self.onnx_model_name.replace("/", "_")

        # Jina provides pre-optimized ONNX models in the 'onnx' subfolder
        if model_cache_path.exists() and (model_cache_path / "model.onnx").exists():
            logger.info(f"Loading Jina v5 ONNX model from cache: {model_cache_path}")
        else:
            logger.info(f"Downloading Jina v5 ONNX model: {self.onnx_model_name}")

        model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_name,
            subfolder="onnx",
            file_name="model.onnx",
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}
            },
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.onnx_model_name,
            trust_remote_code=True,
        )

        # Cache the model for faster subsequent loads
        if not (model_cache_path / "model.onnx").exists():
            model_cache_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_cache_path)
            tokenizer.save_pretrained(model_cache_path)
            logger.info(f"Jina v5 ONNX model cached to: {model_cache_path}")

        return model, tokenizer, False

    def _create_session_options(self, num_threads: int) -> ort.SessionOptions:
        """Create optimized ONNX Runtime session options."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        return sess_options

    def compute_embeddings(
        self,
        model: PreTrainedModel | ORTModelForFeatureExtraction | SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[tuple[str, TextType]],
    ) -> list[list[float]]:
        """Compute embeddings using model-specific method."""
        if isinstance(model, SentenceTransformer):
            return self._compute_embeddings_gpu(model, texts)
        elif isinstance(model, ORTModelForFeatureExtraction):
            return self._compute_embeddings_onnx(
                model, tokenizer, self._format_texts(texts)
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def _compute_embeddings_onnx(
        self,
        model: ORTModelForFeatureExtraction,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
    ) -> list[list[float]]:
        """Compute embeddings using last-token pooling (critical for Jina v5)."""
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state

            # Last-token pooling: take the hidden state of the last non-padding token
            sequence_lengths = encoded["attention_mask"].sum(dim=1) - 1
            embeddings = last_hidden_state[
                torch.arange(last_hidden_state.size(0)), sequence_lengths
            ]

            # L2 normalize
            embeddings = normalize(embeddings, p=2, dim=1)

        return list(embeddings.numpy().tolist())

    def _compute_embeddings_gpu(
        self,
        model: SentenceTransformer,
        texts: list[tuple[str, TextType]],
    ) -> list[list[float]]:
        """Compute embeddings using SentenceTransformer's encode() API.

        SentenceTransformer handles pooling internally.
        We need to separate queries and documents to use the correct prompt_name.
        """

        queries = [text for text, text_type in texts if text_type == TextType.QUERY]
        documents = [text for text, text_type in texts if text_type == TextType.PASSAGE]

        all_embeddings: list[list[float]] = []
        # Encode queries
        if queries:
            query_embeddings = model.encode(  # type: ignore
                sentences=queries,
                task="retrieval",
                prompt_name="query",
            )
            all_embeddings.extend(query_embeddings)

        # Encode documents
        if documents:
            doc_embeddings = model.encode(  # type: ignore
                sentences=documents,
                task="retrieval",
                prompt_name="document",
            )
            all_embeddings.extend(doc_embeddings)

        return all_embeddings


# =============================================================================
# Handler Registry
# =============================================================================


def get_handler(model_choice: EmbeddingModel) -> EmbeddingModelHandler:
    """Get the appropriate handler for a model choice."""
    if model_choice in (
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL,
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE,
    ):
        return E5ModelHandler(model_choice)
    elif model_choice in (EmbeddingModel.JINA_V5_SMALL, EmbeddingModel.JINA_V5_NANO):
        return JinaV5ModelHandler(model_choice)
    else:
        raise ValueError(f"No handler for model: {model_choice}")


# =============================================================================
# Model Cache and Loading
# =============================================================================


def get_onnx_cache_dir() -> Path:
    """Get the directory for caching ONNX models."""
    cache_dir = Path.home() / ".cache" / "lex-db" / "onnx-models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Module-level cache for loaded models
_model_cache: dict[EmbeddingModel, dict] = {}


def get_local_embedding_model(model_choice: EmbeddingModel) -> dict:
    """Get a cached embedding model instance.

    Uses handlers to load models with appropriate settings for GPU or CPU inference.
    """
    if model_choice not in _model_cache:
        handler = get_handler(model_choice)

        # Automatically detect GPU availability
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU acceleration enabled. Using: {gpu_name}")
        else:
            logger.info("CUDA not available, using CPU for inference")

        model, tokenizer, actual_use_gpu = handler.load_model(use_gpu)

        _model_cache[model_choice] = {
            "model": model,
            "tokenizer": tokenizer,
            "use_gpu": actual_use_gpu,
            "handler": handler,
        }

        logger.info(f"Model loaded successfully (GPU: {actual_use_gpu})")

    return _model_cache[model_choice]


# =============================================================================
# Batching Utilities
# =============================================================================


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


# =============================================================================
# Main Embedding Generation
# =============================================================================


def generate_embeddings(
    texts: list[tuple[str, TextType]], model_choice: EmbeddingModel
) -> list[list[float]]:
    """Generate embeddings for a list of texts using the specified model."""
    # Handle empty input early
    if not texts:
        return []

    # Check for empty strings and raise error to maintain length alignment
    empty_indices = [i for i, t in enumerate(texts) if not t[0].strip()]
    if empty_indices:
        raise ValueError(
            f"Found {len(empty_indices)} empty/whitespace-only texts at indices {empty_indices[:5]}... "
            f"Please filter empty strings before calling generate_embeddings()"
        )

    # Handle mock model
    if model_choice == EmbeddingModel.MOCK_MODEL:
        logger.debug(f"Generating MOCK embeddings for {len(texts)} texts")
        return [
            np.random.random_sample(
                get_embedding_dimensions(EmbeddingModel.MOCK_MODEL)
            ).tolist()
            for _ in texts
        ]

    # Handle local models (E5 and Jina)
    if model_choice in (
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE,
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL,
        EmbeddingModel.JINA_V5_SMALL,
        EmbeddingModel.JINA_V5_NANO,
    ):
        model_data = get_local_embedding_model(model_choice)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        use_gpu = model_data["use_gpu"]
        handler = model_data["handler"]

        # Use larger batch size for GPU, smaller for CPU
        settings = get_settings()
        batch_size = settings.GPU_BATCH_SIZE if use_gpu else settings.CPU_BATCH_SIZE

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Compute embeddings using handler
            embeddings = handler.compute_embeddings(model, tokenizer, batch_texts)

            all_embeddings.extend(embeddings)

            if total_batches > 1:
                logger.debug(f"Processed batch {i // batch_size + 1}/{total_batches}")

        return all_embeddings

    # Handle OpenAI models
    elif model_choice in [
        EmbeddingModel.OPENAI_ADA_002,
        EmbeddingModel.OPENAI_SMALL_003,
        EmbeddingModel.OPENAI_LARGE_003,
    ]:
        try:
            from openai import OpenAI
            from concurrent.futures import ThreadPoolExecutor, as_completed

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")

            client = OpenAI(api_key=api_key)
            model_name = model_choice.value

            batches = create_optimal_request_batches(
                [text[0] for text in texts], max_tokens_per_batch=8000
            )
            logger.info(f"Split into {len(batches)} batches for parallel processing.")

            all_results: list[list[list[float]] | None] = [None] * len(batches)

            def process_batch(
                batch_idx_batch: tuple[int, list[str]],
            ) -> tuple[int, list[list[float]] | None]:
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

            max_workers = min(32, len(batches) + 4)
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
