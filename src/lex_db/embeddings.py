"""Embedding generation utilities for vector search in Lex DB."""

from enum import Enum
import os
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer  # type: ignore
from optimum.onnxruntime.configuration import AutoQuantizationConfig  # type: ignore
import onnxruntime as ort  # type: ignore
import torch
from torch.nn.functional import normalize
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import numpy as np
import time
from collections import deque
from threading import Lock
import tiktoken
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
    if model_choice == EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL:
        return 384
    elif model_choice == EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE:
        return 1024
    elif model_choice == EmbeddingModel.OPENAI_ADA_002:
        return 1536
    elif model_choice == EmbeddingModel.OPENAI_SMALL_003:
        return 1536
    elif model_choice == EmbeddingModel.OPENAI_LARGE_003:
        return 3072
    elif model_choice == EmbeddingModel.JINA_V5_SMALL:
        return 1024
    elif model_choice == EmbeddingModel.JINA_V5_NANO:
        return 768
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
_model_cache: dict[EmbeddingModel, dict] = {}


def get_onnx_cache_dir() -> Path:
    """Get the directory for caching ONNX models."""
    cache_dir = Path.home() / ".cache" / "lex-db" / "onnx-models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_local_embedding_model(model_choice: EmbeddingModel) -> dict:
    """Get a cached ONNX embedding model instance.

    Automatically uses GPU if available and USE_GPU=True in settings.
    Falls back to CPU-optimized quantized model otherwise.
    """
    if model_choice not in _model_cache:
        if model_choice not in (
            EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE,
            EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL,
            EmbeddingModel.JINA_V5_NANO,
            EmbeddingModel.JINA_V5_SMALL,
        ):
            raise ValueError(f"Local model not supported: {model_choice}")

        model_name = model_choice.value
        settings = get_settings()
        use_gpu = settings.USE_GPU

        # Define cache directory for this specific model
        # Use separate cache paths for GPU vs CPU to avoid loading wrong model variant
        cache_dir = get_onnx_cache_dir()
        model_cache_path = (
            cache_dir / f"{model_name.replace('/', '_')}{'_gpu' if use_gpu else ''}"
        )
        quantized_model_path = cache_dir / f"{model_name.replace('/', '_')}_quantized"

        # Check GPU availability
        if use_gpu:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(
                    f"GPU acceleration enabled. Found {gpu_count} GPU(s): {gpu_name}"
                )
            else:
                logger.warning(
                    "USE_GPU=True but CUDA is not available. Falling back to CPU."
                )
                use_gpu = False
                # Update cache path for CPU fallback
                model_cache_path = cache_dir / model_name.replace("/", "_")

        logger.info(f"Loading ONNX model: {model_name} (GPU: {use_gpu})")

        if use_gpu:
            # GPU path: Load model without quantization for better GPU performance
            device_ids = [int(d.strip()) for d in settings.GPU_DEVICE_IDS.split(",")]
            logger.info(f"Using GPU devices: {device_ids}")

            # Load one model instance per GPU for parallel processing
            models_per_gpu = []
            for device_id in device_ids:
                model, tokenizer = _load_model_for_gpu(
                    model_name, model_cache_path, device_id
                )
                models_per_gpu.append((model, tokenizer, device_id))

            # Use first model/tokenizer as primary (for single-GPU fallback)
            model, tokenizer, _ = models_per_gpu[0]
        else:
            # CPU path: Use quantized model for faster CPU inference
            model, tokenizer = _load_model_for_cpu(
                model_name, model_cache_path, quantized_model_path
            )
            models_per_gpu = None

        _model_cache[model_choice] = {
            "model": model,
            "tokenizer": tokenizer,
            "use_gpu": use_gpu,
            "models_per_gpu": models_per_gpu,
        }

        logger.info(f"ONNX model loaded successfully (GPU: {use_gpu})")

    return _model_cache[model_choice]


def _load_model_for_gpu(
    model_name: str, model_cache_path: Path, device_id: int = 0
) -> tuple:
    """Load ONNX model optimized for GPU inference.

    GPU inference works best with non-quantized FP32 models.

    Args:
        model_name: HuggingFace model name
        model_cache_path: Path to cache directory
        device_id: GPU device ID to use (default: 0)
    """
    settings = get_settings()
    gpu_mem_limit = (
        settings.GPU_MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    )  # Convert GB to bytes

    provider_options = {
        "device_id": device_id,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": gpu_mem_limit,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
    }

    # Check if model exists in cache
    if model_cache_path.exists() and (model_cache_path / "model.onnx").exists():
        logger.info(
            f"Loading ONNX model from cache: {model_cache_path} (GPU {device_id})"
        )
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_cache_path,
            provider="CUDAExecutionProvider",
            provider_options=provider_options,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cache_path)
    else:
        logger.info(
            f"Exporting and saving ONNX model to: {model_cache_path} (GPU {device_id})"
        )
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CUDAExecutionProvider",
            provider_options=provider_options,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save the model and tokenizer to disk
        model_cache_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_cache_path)
        tokenizer.save_pretrained(model_cache_path)
        logger.info(f"ONNX model saved to cache: {model_cache_path}")

    return model, tokenizer


def _load_model_for_cpu(
    model_name: str, model_cache_path: Path, quantized_model_path: Path
) -> tuple:
    """Load ONNX model optimized for CPU inference with INT8 quantization."""
    settings = get_settings()
    num_threads = settings.CPU_NUM_THREADS

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # Enable additional optimizations
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True

    # Check if quantized model exists
    if (
        quantized_model_path.exists()
        and (quantized_model_path / "model_quantized.onnx").exists()
    ):
        logger.info(f"Loading quantized ONNX model from cache: {quantized_model_path}")
        model = ORTModelForFeatureExtraction.from_pretrained(
            quantized_model_path,
            file_name="model_quantized.onnx",
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {
                    "arena_extend_strategy": "kSameAsRequested",
                }
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    # Check if regular ONNX model exists
    elif model_cache_path.exists() and (model_cache_path / "model.onnx").exists():
        logger.info(f"Loading ONNX model from cache: {model_cache_path}")
        logger.info("Quantizing model to INT8 for faster inference...")

        # Load the model first
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_cache_path,
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {
                    "arena_extend_strategy": "kSameAsRequested",
                }
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

        # Quantize the model
        quantizer = ORTQuantizer.from_pretrained(model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        # Save quantized model
        quantized_model_path.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=quantized_model_path,
            quantization_config=qconfig,
            file_suffix="quantized",
        )
        tokenizer.save_pretrained(quantized_model_path)

        # Reload the quantized model
        model = ORTModelForFeatureExtraction.from_pretrained(
            quantized_model_path,
            file_name="model_quantized.onnx",
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {
                    "arena_extend_strategy": "kSameAsRequested",
                }
            },
        )
        logger.info(f"Quantized model saved to: {quantized_model_path}")
    else:
        logger.info(f"Exporting and saving ONNX model to: {model_cache_path}")
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {
                    "arena_extend_strategy": "kSameAsRequested",
                }
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save the model and tokenizer to disk
        model.save_pretrained(model_cache_path)
        tokenizer.save_pretrained(model_cache_path)
        logger.info(f"ONNX model saved to cache: {model_cache_path}")

        # Quantize the newly exported model
        logger.info("Quantizing model to INT8 for faster inference...")
        quantizer = ORTQuantizer.from_pretrained(model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        quantized_model_path.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=quantized_model_path,
            quantization_config=qconfig,
            file_suffix="quantized",
        )
        tokenizer.save_pretrained(quantized_model_path)

        # Reload the quantized model
        model = ORTModelForFeatureExtraction.from_pretrained(
            quantized_model_path,
            file_name="model_quantized.onnx",
            provider="CPUExecutionProvider",
            session_options=sess_options,
            provider_options={
                "CPUExecutionProvider": {
                    "arena_extend_strategy": "kSameAsRequested",
                }
            },
        )
        logger.info(f"Quantized model saved to: {quantized_model_path}")

    return model, tokenizer


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


def create_text_batches(texts: list[str], batch_size: int = 32) -> list[list[str]]:
    """Create batches of texts for parallel processing."""
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def _compute_embeddings(
    model: ORTModelForFeatureExtraction,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device_id: int | None = None,
) -> list[list[float]]:
    """Compute mean-pooled normalized embeddings for a batch of texts."""

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    # Move tensors to GPU if specified
    if device_id is not None:
        encoded = {k: v.cuda(device_id) for k, v in encoded.items()}

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


def _generate_embeddings_single_device(
    texts: list[tuple[str, TextType]],
    model: ORTModelForFeatureExtraction,
    tokenizer: PreTrainedTokenizerBase,
    use_gpu: bool,
    batch_size: int,
) -> list[list[float]]:
    """Generate embeddings using a single device (CPU or single GPU)."""
    all_embeddings = []
    device_id = 0 if use_gpu else None
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        formatted_texts = [f"{text[1].value}: {text[0]}" for text in batch_texts]

        embeddings = _compute_embeddings(model, tokenizer, formatted_texts, device_id)
        all_embeddings.extend(embeddings)

        if total_batches > 1:
            logger.debug(f"Processed batch {i // batch_size + 1}/{total_batches}")

    return all_embeddings


def _generate_embeddings_multi_gpu(
    texts: list[tuple[str, TextType]],
    models_per_gpu: list[
        tuple[ORTModelForFeatureExtraction, PreTrainedTokenizerBase, int]
    ],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> list[list[float]]:
    """Generate embeddings using multiple GPUs in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_gpus = len(models_per_gpu)
    logger.info(f"Processing {len(texts)} texts across {num_gpus} GPUs")

    # Split texts into chunks for each GPU
    chunk_size = (len(texts) + num_gpus - 1) // num_gpus
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Pad with empty lists to match GPU count
    chunks.extend([[] for _ in range(num_gpus - len(chunks))])

    all_embeddings: list[list[list[float]]] = [[] for _ in range(len(chunks))]

    def process_chunk(
        chunk_idx: int, chunk: list, model_tokenizer_device: tuple
    ) -> tuple[int, list[list[float]]]:
        """Process a chunk of texts on a specific GPU."""
        model, _, device_id = model_tokenizer_device

        if not chunk:
            return chunk_idx, []

        chunk_embeddings = []
        total_batches = (len(chunk) + batch_size - 1) // batch_size

        for i in range(0, len(chunk), batch_size):
            batch_texts = chunk[i : i + batch_size]
            formatted_texts = [f"{text[1].value}: {text[0]}" for text in batch_texts]

            embeddings = _compute_embeddings(
                model, tokenizer, formatted_texts, device_id
            )
            chunk_embeddings.extend(embeddings)

            if total_batches > 1:
                logger.debug(
                    f"GPU {device_id}: Processed batch {i // batch_size + 1}/{total_batches}"
                )

        return chunk_idx, chunk_embeddings

    # Process chunks in parallel across GPUs
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(process_chunk, idx, chunk, mtd)
            for idx, (chunk, mtd) in enumerate(zip(chunks, models_per_gpu))
        ]
        for future in as_completed(futures):
            chunk_idx, chunk_embeddings = future.result()
            all_embeddings[chunk_idx] = chunk_embeddings

    # Flatten results in order
    result = [emb for chunk_embs in all_embeddings for emb in chunk_embs]
    logger.info(f"Multi-GPU processing complete: {len(result)} embeddings generated")
    return result


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

    if model_choice == EmbeddingModel.MOCK_MODEL:  # Add this block
        logger.debug(f"Generating MOCK embeddings for {len(texts)} texts")
        # Return a list of random dummy embeddings for testing
        return [
            np.random.random_sample(
                get_embedding_dimensions(EmbeddingModel.MOCK_MODEL)
            ).tolist()
            for t in texts
        ]

    elif model_choice in (
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE,
        EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL,
        EmbeddingModel.JINA_V5_SMALL,
        EmbeddingModel.JINA_V5_NANO,
    ):
        model_data = get_local_embedding_model(model_choice)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        use_gpu = model_data.get("use_gpu", False)
        models_per_gpu = model_data.get("models_per_gpu")

        # Use larger batch size for GPU, smaller for CPU
        settings = get_settings()
        batch_size = settings.GPU_BATCH_SIZE if use_gpu else settings.CPU_BATCH_SIZE

        # Check if we have multiple GPUs for parallel processing
        if use_gpu and models_per_gpu and len(models_per_gpu) > 1:
            # Multi-GPU parallel processing
            all_embeddings = _generate_embeddings_multi_gpu(
                texts, models_per_gpu, tokenizer, batch_size
            )
        else:
            # Single GPU or CPU processing
            all_embeddings = _generate_embeddings_single_device(
                texts, model, tokenizer, use_gpu, batch_size
            )

        return all_embeddings

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

            max_workers = min(32, len(batches) + 4)  # Don't over-provision
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_batch, (i, batch))
                    for i, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    batch_idx, result = future.result()  # type: ignore
                    if result is not None:
                        all_results[batch_idx] = result
            # Flatten results in correct order
            all_embeddings = []
            for result in all_results:  # type: ignore
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
