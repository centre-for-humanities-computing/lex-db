"""Unit tests for vector_store module."""

import pytest
from unittest.mock import MagicMock, patch
from lex_db.vector_store import (
    create_vector_index,
    add_chunks_to_vector_index,
    add_precomputed_embeddings_to_vector_index,
    remove_article_from_vector_index,
    search_vector_index,
    insert_vector_index_metadata,
    update_vector_index_metadata,
    get_vector_index_metadata,
    get_all_vector_index_metadata,
    VectorSearchResults,
)
from lex_db.embeddings import EmbeddingModel


class TestCreateVectorIndex:
    """Tests for create_vector_index function."""

    @patch("lex_db.vector_store.insert_vector_index_metadata")
    def test_creates_table_without_force(
        self,
        mock_insert_metadata: MagicMock,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test creating vector index without force flag."""
        # Mock the existence check to return None (table doesn't exist)
        mock_db_connection.execute.return_value.fetchone.return_value = None

        create_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embedding_model_choice=mock_embedding_model,
            source_table="articles",
            source_column="xhtml_md",
            force=False,
        )

        # Verify table creation was called
        assert mock_db_connection.execute.called
        assert mock_db_connection.commit.called

        # Verify metadata was inserted
        assert mock_insert_metadata.called

    @patch("lex_db.vector_store.insert_vector_index_metadata")
    def test_drops_table_with_force(
        self,
        mock_insert_metadata: MagicMock,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test creating vector index with force=True drops existing table."""
        create_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embedding_model_choice=mock_embedding_model,
            source_table="articles",
            source_column="xhtml_md",
            force=True,
        )

        # Verify DROP TABLE was called
        execute_calls = [
            str(call) for call in mock_db_connection.execute.call_args_list
        ]
        assert any("DROP TABLE" in str(call) for call in execute_calls)

    def test_raises_error_if_table_exists_without_force(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test raises ValueError if table exists and force=False."""
        # Mock the existence check to return a row (table exists)
        mock_db_connection.execute.return_value.fetchone.return_value = {
            "tablename": "test_index"
        }

        with pytest.raises(ValueError, match="already exists"):
            create_vector_index(
                db_conn=mock_db_connection,
                vector_index_name="test_index",
                embedding_model_choice=mock_embedding_model,
                source_table="articles",
                source_column="xhtml_md",
                force=False,
            )


class TestAddChunksToVectorIndex:
    """Tests for add_chunks_to_vector_index function."""

    @patch("lex_db.vector_store.generate_embeddings")
    def test_adds_chunks_successfully(
        self,
        mock_generate: MagicMock,
        mock_db_connection: MagicMock,
        sample_chunks_data: list,
        mock_embeddings: list,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test adding chunks to vector index."""
        mock_generate.return_value = mock_embeddings

        add_chunks_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            chunks_data=sample_chunks_data,
            embedding_model_choice=mock_embedding_model,
        )

        # Verify embeddings were generated
        mock_generate.assert_called_once()

        # Verify cursor.executemany was called
        mock_db_connection.cursor.return_value.__enter__.return_value.executemany.assert_called_once()

        # Verify commit was called
        mock_db_connection.commit.assert_called_once()

    def test_empty_chunks_data(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test empty chunks data returns early."""
        add_chunks_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            chunks_data=[],
            embedding_model_choice=mock_embedding_model,
        )

        # Verify no database operations were performed
        assert not mock_db_connection.commit.called

    @patch("lex_db.vector_store.generate_embeddings")
    def test_handles_embedding_mismatch(
        self,
        mock_generate: MagicMock,
        mock_db_connection: MagicMock,
        sample_chunks_data: list,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test handles mismatch between chunks and embeddings."""
        # Return fewer embeddings than chunks
        mock_generate.return_value = [[0.1, 0.2, 0.3, 0.4]]

        add_chunks_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            chunks_data=sample_chunks_data,
            embedding_model_choice=mock_embedding_model,
        )

        # Should return early without committing
        assert not mock_db_connection.commit.called


class TestAddPrecomputedEmbeddingsToVectorIndex:
    """Tests for add_precomputed_embeddings_to_vector_index function."""

    def test_adds_embeddings_successfully(
        self,
        mock_db_connection: MagicMock,
        sample_embeddings_data: list,
    ) -> None:
        """Test adding pre-computed embeddings."""
        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embeddings_data=sample_embeddings_data,
            batch_size=1000,
        )

        # Verify stats
        assert stats["created"] == 3
        assert stats["errors"] == 0

        # Verify commit was called
        assert mock_db_connection.commit.called

    def test_empty_embeddings_data(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test empty embeddings data returns zero stats."""
        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embeddings_data=[],
        )

        assert stats["created"] == 0
        assert stats["errors"] == 0

    def test_skips_empty_chunk_text(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test skips entries with empty chunk text."""
        embeddings_data = [
            ("1", "0", "", [0.1, 0.2, 0.3, 0.4]),  # Empty text
            ("1", "1", "Valid text", [0.2, 0.3, 0.4, 0.5]),
        ]

        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embeddings_data=embeddings_data,
        )

        assert stats["created"] == 1
        assert stats["errors"] == 1

    def test_skips_empty_embedding(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test skips entries with empty embedding."""
        embeddings_data = [
            ("1", "0", "Text", []),  # Empty embedding
            ("1", "1", "Valid text", [0.2, 0.3, 0.4, 0.5]),
        ]

        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embeddings_data=embeddings_data,
        )

        assert stats["created"] == 1
        assert stats["errors"] == 1

    def test_batching(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test batching with small batch size."""
        embeddings_data = [
            ("1", "0", "Text 1", [0.1, 0.2, 0.3, 0.4]),
            ("1", "1", "Text 2", [0.2, 0.3, 0.4, 0.5]),
            ("1", "2", "Text 3", [0.3, 0.4, 0.5, 0.6]),
        ]

        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            embeddings_data=embeddings_data,
            batch_size=2,  # Small batch size
        )

        # Should process in 2 batches (2 + 1)
        assert stats["created"] == 3
        assert mock_db_connection.commit.call_count == 2


class TestRemoveArticleFromVectorIndex:
    """Tests for remove_article_from_vector_index function."""

    def test_removes_article_chunks(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test removing article chunks."""
        # Mock count query
        count_result = {"count": 5}
        mock_db_connection.execute.return_value.fetchone.return_value = count_result

        count = remove_article_from_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            article_id="123",
        )

        # Verify count returned
        assert count == 5

        # Verify DELETE was called
        assert mock_db_connection.execute.call_count == 2  # COUNT + DELETE
        assert mock_db_connection.commit.called

    def test_removes_nonexistent_article(
        self,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test removing non-existent article returns 0."""
        count_result = {"count": 0}
        mock_db_connection.execute.return_value.fetchone.return_value = count_result

        count = remove_article_from_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            article_id="999",
        )

        assert count == 0


class TestSearchVectorIndex:
    """Tests for search_vector_index function."""

    @patch("lex_db.vector_store.generate_query_embedding")
    def test_search_returns_results(
        self,
        mock_generate_query: MagicMock,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test vector search returns results."""
        mock_generate_query.return_value = [0.1, 0.2, 0.3, 0.4]

        # Mock search results
        search_rows = [
            {
                "id": 1,
                "source_article_id": 123,
                "chunk_sequence_id": 0,
                "chunk_text": "Test chunk",
                "distance": 0.15,
            }
        ]
        mock_db_connection.execute.return_value.fetchall.return_value = search_rows

        results = search_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            query_text="test query",
            embedding_model=mock_embedding_model,
            top_k=5,
        )

        # Verify results
        assert isinstance(results, VectorSearchResults)
        assert len(results.results) == 1
        assert results.results[0].id_in_index == 1
        assert results.results[0].source_article_id == "123"
        assert results.results[0].chunk_text == "Test chunk"
        assert results.results[0].distance == 0.15

    @patch("lex_db.vector_store.generate_query_embedding")
    def test_search_no_results(
        self,
        mock_generate_query: MagicMock,
        mock_db_connection: MagicMock,
        mock_embedding_model: EmbeddingModel,
    ) -> None:
        """Test search with no matching results."""
        mock_generate_query.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_db_connection.execute.return_value.fetchall.return_value = []

        results = search_vector_index(
            db_conn=mock_db_connection,
            vector_index_name="test_index",
            query_text="nonexistent",
            embedding_model=mock_embedding_model,
            top_k=5,
        )

        assert len(results.results) == 0


class TestVectorIndexMetadata:
    """Tests for vector index metadata functions."""

    @patch("lex_db.vector_store.create_vector_index_metadata_table")
    def test_insert_metadata(
        self,
        mock_create_table: MagicMock,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test inserting vector index metadata."""
        insert_vector_index_metadata(
            db_conn=mock_db_connection,
            index_name="test_index",
            source_table="articles",
            source_column="xhtml_md",
            embedding_model="mock_model",
            chunk_size=512,
            chunk_overlap=50,
            chunking_strategy="sections",
        )

        # Verify table creation was called
        mock_create_table.assert_called_once()

        # Verify insert was executed
        assert mock_db_connection.execute.called
        assert mock_db_connection.commit.called

    @patch("lex_db.vector_store.create_vector_index_metadata_table")
    def test_update_metadata(
        self,
        mock_create_table: MagicMock,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test updating vector index metadata."""
        update_vector_index_metadata(
            db_conn=mock_db_connection,
            index_name="test_index",
            chunk_size=1024,
            chunk_overlap=100,
        )

        # Verify update was executed
        assert mock_db_connection.execute.called
        assert mock_db_connection.commit.called

    @patch("lex_db.vector_store.create_vector_index_metadata_table")
    def test_get_metadata(
        self,
        mock_create_table: MagicMock,
        mock_db_connection: MagicMock,
        vector_index_metadata: dict,
    ) -> None:
        """Test getting vector index metadata."""
        mock_db_connection.execute.return_value.fetchone.return_value = (
            vector_index_metadata
        )

        metadata = get_vector_index_metadata(
            db_conn=mock_db_connection,
            index_name="test_index",
        )

        assert metadata == vector_index_metadata

    @patch("lex_db.vector_store.create_vector_index_metadata_table")
    def test_get_metadata_not_found(
        self,
        mock_create_table: MagicMock,
        mock_db_connection: MagicMock,
    ) -> None:
        """Test getting non-existent metadata returns None."""
        mock_db_connection.execute.return_value.fetchone.return_value = None

        metadata = get_vector_index_metadata(
            db_conn=mock_db_connection,
            index_name="nonexistent",
        )

        assert metadata is None

    @patch("lex_db.vector_store.create_vector_index_metadata_table")
    def test_get_all_metadata(
        self,
        mock_create_table: MagicMock,
        mock_db_connection: MagicMock,
        vector_index_metadata: dict,
    ) -> None:
        """Test getting all vector index metadata."""
        mock_db_connection.execute.return_value.fetchall.return_value = [
            vector_index_metadata,
            {**vector_index_metadata, "index_name": "another_index"},
        ]

        all_metadata = get_all_vector_index_metadata(db_conn=mock_db_connection)

        assert len(all_metadata) == 2
        assert all_metadata[0]["index_name"] == "test_index"
        assert all_metadata[1]["index_name"] == "another_index"
