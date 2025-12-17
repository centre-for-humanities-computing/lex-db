"""Unit tests for embeddings module."""

import pytest
from lex_db.embeddings import (
    EmbeddingModel,
    get_embedding_dimensions,
    generate_embeddings,
    generate_query_embedding,
    create_text_batches,
)


class TestGetEmbeddingDimensions:
    """Tests for get_embedding_dimensions function."""

    def test_mock_model_dimensions(self) -> None:
        """Test MOCK_MODEL returns 4 dimensions."""
        assert get_embedding_dimensions(EmbeddingModel.MOCK_MODEL) == 4

    def test_e5_small_dimensions(self) -> None:
        """Test E5 small returns 384 dimensions."""
        assert (
            get_embedding_dimensions(EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL) == 384
        )

    def test_e5_large_dimensions(self) -> None:
        """Test E5 large returns 1024 dimensions."""
        assert (
            get_embedding_dimensions(EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE) == 1024
        )

    def test_invalid_model_raises_error(self) -> None:
        """Test invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_embedding_dimensions("invalid_model")  # type: ignore


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function."""

    def test_mock_model_generates_embeddings(
        self, sample_texts: list[str], mock_embedding_model: EmbeddingModel
    ) -> None:
        """Test MOCK_MODEL generates embeddings with correct dimensions."""
        embeddings = generate_embeddings(sample_texts, mock_embedding_model)

        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 4 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_empty_texts_list(self, mock_embedding_model: EmbeddingModel) -> None:
        """Test empty texts list returns empty embeddings."""
        embeddings = generate_embeddings([], mock_embedding_model)
        assert embeddings == []

    def test_single_text(self, mock_embedding_model: EmbeddingModel) -> None:
        """Test single text generates single embedding."""
        embeddings = generate_embeddings(["Test text"], mock_embedding_model)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 4

    def test_e5_small_model_generates_embeddings(self) -> None:
        """Test E5 small model generates embeddings with correct dimensions.

        This test uses the actual ONNX model to verify the model loading,
        caching, and embedding generation logic works correctly.
        Note: First run will be slow (model download/quantization),
        subsequent runs use cached model.
        """
        texts = ["Dette er en test.", "Her er mere tekst."]
        embeddings = generate_embeddings(
            texts, EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL, query=False
        )

        # Verify correct number of embeddings
        assert len(embeddings) == len(texts)

        # Verify correct dimensions (384 for E5 small)
        assert all(len(emb) == 384 for emb in embeddings)

        # Verify embeddings are normalized (L2 norm should be ~1.0)
        import numpy as np

        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01, f"Embedding not normalized: {norm}"

        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]

    def test_e5_small_query_vs_passage_prefix(self) -> None:
        """Test E5 small model uses different prefixes for query vs passage."""
        text = "Test text"

        # Generate with query=True
        query_emb = generate_embeddings(
            [text], EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL, query=True
        )[0]

        # Generate with query=False (passage)
        passage_emb = generate_embeddings(
            [text], EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL, query=False
        )[0]

        # Embeddings should be different due to different prefixes
        assert query_emb != passage_emb

        # Both should be normalized
        import numpy as np

        assert 0.99 < np.linalg.norm(query_emb) < 1.01
        assert 0.99 < np.linalg.norm(passage_emb) < 1.01

    def test_raises_error_on_empty_strings(
        self, mock_embedding_model: EmbeddingModel
    ) -> None:
        """Test that empty strings raise ValueError."""
        texts = ["Valid text", "", "  ", "\t", "\n", "Another valid text"]

        with pytest.raises(ValueError, match="empty/whitespace-only texts"):
            generate_embeddings(texts, mock_embedding_model)


class TestGenerateQueryEmbedding:
    """Tests for generate_query_embedding function."""

    def test_generates_single_embedding(
        self, mock_embedding_model: EmbeddingModel
    ) -> None:
        """Test generates single embedding for query text."""
        embedding = generate_query_embedding("Test query", mock_embedding_model)

        assert isinstance(embedding, list)
        assert len(embedding) == 4
        assert all(isinstance(val, float) for val in embedding)

    def test_empty_query_text(self, mock_embedding_model: EmbeddingModel) -> None:
        """Test empty query text raises error."""
        with pytest.raises(ValueError, match="empty/whitespace-only texts"):
            generate_query_embedding("", mock_embedding_model)


class TestCreateTextBatches:
    """Tests for create_text_batches function."""

    def test_single_batch(self) -> None:
        """Test texts fit in single batch."""
        texts = ["text1", "text2", "text3"]
        batches = create_text_batches(texts, batch_size=10)

        assert len(batches) == 1
        assert batches[0] == texts

    def test_multiple_batches(self) -> None:
        """Test texts split into multiple batches."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batches = create_text_batches(texts, batch_size=2)

        assert len(batches) == 3
        assert batches[0] == ["text1", "text2"]
        assert batches[1] == ["text3", "text4"]
        assert batches[2] == ["text5"]

    def test_empty_texts(self) -> None:
        """Test empty texts list returns empty batches."""
        batches = create_text_batches([], batch_size=10)
        assert batches == []
