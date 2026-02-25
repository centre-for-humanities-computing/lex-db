"""Vector store operations for Lex DB."""

from psycopg import Connection
from psycopg import sql
from typing import Any

from pydantic import BaseModel
from lex_db.utils import get_logger, split_document_into_chunks, ChunkingStrategy
from lex_db.embeddings import (
    EmbeddingModel,
    TextType,
    generate_embeddings,
    get_embedding_dimensions,
)

logger = get_logger()


def create_vector_index(
    db_conn: Connection[Any],
    vector_index_name: str,
    embedding_model_choice: EmbeddingModel,
    source_table: str,
    source_column: str,
    force: bool = False,
    updated_at_column: str = "changed_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.TOKENS,
) -> None:
    """Create a new vector index structure using pgvector and store its metadata."""
    embedding_dim = get_embedding_dimensions(embedding_model_choice)

    if force:
        # Drop existing table and all dependent objects (indexes, constraints, etc.)
        db_conn.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(vector_index_name)
            )
        )
        logger.info(f"Dropped existing table {vector_index_name}")
    else:
        # Check if the table already exists using PostgreSQL system catalog
        result = db_conn.execute(
            """
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' AND tablename = %s
            """,
            (vector_index_name,),
        )
        if result.fetchone():
            raise ValueError(
                f"Table {vector_index_name} already exists. Use --force-drop to drop it."
            )

    # Create table with pgvector extension
    db_conn.execute(
        sql.SQL("""
            CREATE TABLE {} (
                id BIGSERIAL PRIMARY KEY,
                embedding vector({}),
                source_article_id INTEGER NOT NULL,
                chunk_sequence_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT {} 
                    UNIQUE (source_article_id, chunk_sequence_id)
            )
        """).format(
            sql.Identifier(vector_index_name),
            sql.Literal(embedding_dim),
            sql.Identifier(f"{vector_index_name}_unique_chunk"),
        )
    )
    logger.info(f"Vector table {vector_index_name} created with pgvector")

    # Create HNSW index for fast similarity search using cosine distance
    # HNSW provides better accuracy than IVFFlat for most use cases
    db_conn.execute(
        sql.SQL("""
            CREATE INDEX {} 
            ON {} 
            USING hnsw (embedding vector_cosine_ops)
        """).format(
            sql.Identifier(f"{vector_index_name}_embedding_idx"),
            sql.Identifier(vector_index_name),
        )
    )
    logger.info(f"HNSW index created on {vector_index_name}.embedding")

    # Add foreign key constraint to articles table for referential integrity
    db_conn.execute(
        sql.SQL("""
            ALTER TABLE {}
            ADD CONSTRAINT {}
            FOREIGN KEY (source_article_id) REFERENCES articles(id)
            ON DELETE CASCADE
        """).format(
            sql.Identifier(vector_index_name),
            sql.Identifier(f"{vector_index_name}_fk_source_article"),
        )
    )
    logger.info(f"Foreign key constraint added to {vector_index_name}")

    # Add generated tsvector column for full-text search on chunk_text
    # This enables FTS queries without a separate FTS table
    db_conn.execute(
        sql.SQL("""
            ALTER TABLE {}
            ADD COLUMN chunk_text_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('danish', coalesce(chunk_text, ''))) STORED
        """).format(
            sql.Identifier(vector_index_name),
        )
    )
    logger.info(f"FTS tsvector column added to {vector_index_name}")

    # Create GIN index for fast full-text search queries
    db_conn.execute(
        sql.SQL("""
            CREATE INDEX {} ON {} USING GIN(chunk_text_tsv)
        """).format(
            sql.Identifier(f"{vector_index_name}_fts_idx"),
            sql.Identifier(vector_index_name),
        )
    )
    logger.info(f"GIN index created on {vector_index_name}.chunk_text_tsv")

    db_conn.commit()

    # Insert metadata if all required fields are provided
    if source_table and source_column:
        # Check if metadata entry already exists
        result = db_conn.execute(
            "SELECT index_name FROM vector_index_metadata WHERE index_name = %s",
            (vector_index_name,),
        )
        if result.fetchone():
            # Update existing metadata with new values
            update_vector_index_metadata(
                db_conn=db_conn,
                index_name=vector_index_name,
                source_table=source_table,
                source_column=source_column,
                updated_at_column=updated_at_column,
                embedding_model=str(embedding_model_choice.value),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy.value,
            )
            logger.info(f"Metadata for index '{vector_index_name}' updated.")
        else:
            # Insert new metadata
            insert_vector_index_metadata(
                db_conn=db_conn,
                index_name=vector_index_name,
                source_table=source_table,
                source_column=source_column,
                updated_at_column=updated_at_column,
                embedding_model=str(embedding_model_choice.value),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy.value,
            )
            logger.info(f"Metadata for index '{vector_index_name}' inserted.")
    else:
        logger.warning(
            f"Metadata not stored for index '{vector_index_name}': source_table or source_column missing."
        )


def add_chunks_to_vector_index(
    db_conn: Connection[Any],
    vector_index_name: str,
    chunks_data: list[
        tuple[str, str, str]
    ],  # List of (article_id, chunk_id, chunk_text)
    embedding_model_choice: EmbeddingModel,
) -> None:
    """Add multiple chunks to an existing vector index using PostgreSQL UPSERT."""
    if not chunks_data:
        logger.warning("No chunks provided to add_chunks_to_vector_index")
        return

    # Filter out chunks with empty text to avoid embedding generation errors
    filtered_chunks = [
        (aid, cid, text) for aid, cid, text in chunks_data if text.strip()
    ]
    if len(filtered_chunks) < len(chunks_data):
        logger.warning(
            f"Filtered out {len(chunks_data) - len(filtered_chunks)} chunks with empty text"
        )

    if not filtered_chunks:
        logger.warning("All chunks had empty text, nothing to add")
        return

    # Generate embeddings for all chunks in batch
    embeddings = generate_embeddings(
        [(chunk_text, TextType.PASSAGE) for _, _, chunk_text in chunks_data],
        model_choice=embedding_model_choice,
    )

    if not embeddings:
        logger.error("No embeddings generated for the provided chunks.")
        return

    if len(embeddings) != len(filtered_chunks):
        logger.error(
            f"Mismatch: {len(filtered_chunks)} chunks but {len(embeddings)} embeddings generated."
        )
        return

    # Prepare data for batch insert - pgvector accepts Python lists directly
    insert_data = [
        (
            embedding,  # pgvector handles list[float] -> vector conversion
            int(article_id),
            int(chunk_id),
            chunk_text,
        )
        for (article_id, chunk_id, chunk_text), embedding in zip(
            filtered_chunks, embeddings
        )
    ]

    # Use UPSERT for idempotency - if chunk exists, update it
    # PostgreSQL's ON CONFLICT handles the unique constraint on (source_article_id, chunk_sequence_id)
    with db_conn.cursor() as cursor:
        cursor.executemany(
            sql.SQL("""
                INSERT INTO {} 
                    (embedding, source_article_id, chunk_sequence_id, chunk_text, last_updated)
                VALUES (%s::vector, %s, %s, %s, NOW())
                ON CONFLICT (source_article_id, chunk_sequence_id)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    chunk_text = EXCLUDED.chunk_text,
                    last_updated = NOW()
            """).format(sql.Identifier(vector_index_name)),
            insert_data,
        )

    db_conn.commit()
    logger.debug(
        f"Added/updated {len(filtered_chunks)} chunks to index {vector_index_name}"
    )


def add_precomputed_embeddings_to_vector_index(
    db_conn: Connection[Any],
    vector_index_name: str,
    embeddings_data: list[
        tuple[str, str, str, list[float]]
    ],  # (article_id, chunk_idx, chunk_text, embedding)
    batch_size: int = 1000,
) -> dict[str, int]:
    """
    Add pre-computed embeddings to a vector index.

    This function is useful when you have embeddings from external sources
    (like OpenAI batch API) that you want to upload to the database.

    Args:
        db_conn: PostgreSQL database connection
        vector_index_name: Name of the vector index table
        embeddings_data: List of (article_id, chunk_idx, chunk_text, embedding) tuples
                        All fields must be present - no re-chunking is performed
        batch_size: Number of embeddings to insert per batch (default: 1000 for memory efficiency)

    Returns:
        Dictionary with stats: {"created": N, "errors": K}
    """
    stats = {"created": 0, "errors": 0}

    if not embeddings_data:
        logger.warning(
            "No embeddings data provided to add_precomputed_embeddings_to_vector_index"
        )
        return stats

    logger.info(
        f"Processing {len(embeddings_data)} pre-computed embeddings in batches of {batch_size}"
    )

    # Prepare batch data for insertion
    batch_data = []
    for article_id, chunk_idx, chunk_text, embedding in embeddings_data:
        try:
            if not chunk_text:
                logger.warning(
                    f"Empty chunk text for article {article_id}, chunk {chunk_idx} - skipping"
                )
                stats["errors"] += 1
                continue

            if not embedding:
                logger.warning(
                    f"Empty embedding for article {article_id}, chunk {chunk_idx} - skipping"
                )
                stats["errors"] += 1
                continue

            batch_data.append((embedding, int(article_id), int(chunk_idx), chunk_text))

        except Exception as e:
            logger.error(
                f"Error preparing embedding for article {article_id}, chunk {chunk_idx}: {e}"
            )
            stats["errors"] += 1

    # Insert embeddings in batches to avoid memory issues
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i : i + batch_size]

        try:
            with db_conn.cursor() as cursor:
                cursor.executemany(
                    sql.SQL("""
                        INSERT INTO {} 
                            (embedding, source_article_id, chunk_sequence_id, chunk_text, last_updated)
                        VALUES (%s::vector, %s, %s, %s, NOW())
                        ON CONFLICT (source_article_id, chunk_sequence_id)
                        DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            chunk_text = EXCLUDED.chunk_text,
                            last_updated = NOW()
                    """).format(sql.Identifier(vector_index_name)),
                    batch,
                )
            db_conn.commit()
            stats["created"] += len(batch)

            # Log progress every 10 batches
            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    f"Processed {stats['created']}/{len(batch_data)} embeddings..."
                )

        except Exception as e:
            logger.error(f"Error inserting batch starting at index {i}: {e}")
            stats["errors"] += len(batch)
            db_conn.rollback()

    logger.info(f"Pre-computed embeddings processing complete: {stats}")
    return stats


def remove_article_from_vector_index(
    db_conn: Connection[Any], vector_index_name: str, article_id: str
) -> int:
    """Remove all chunks for an article from a vector index."""
    # Count chunks before deletion for logging
    result = db_conn.execute(
        sql.SQL("SELECT COUNT(*) FROM {} WHERE source_article_id = %s").format(
            sql.Identifier(vector_index_name)
        ),
        (int(article_id),),
    )
    count = result.fetchone()["count"]  # type: ignore[index]

    # Delete all chunks for this article
    db_conn.execute(
        sql.SQL("DELETE FROM {} WHERE source_article_id = %s").format(
            sql.Identifier(vector_index_name)
        ),
        (int(article_id),),
    )

    db_conn.commit()
    logger.debug(
        f"Removed article {article_id} ({count} chunks) from index {vector_index_name}"
    )
    return int(count)


class VectorSearchResult(BaseModel):
    """Result of a vector search."""

    id_in_index: int
    source_article_id: str
    chunk_seq: int
    chunk_text: str
    distance: float


class VectorSearchResults(BaseModel):
    """Result of a vector search."""

    results: list[VectorSearchResult]


def search_vector_index(
    db_conn: Connection[Any],
    vector_index_name: str,
    queries: list[tuple[str, TextType]],
    embedding_model: EmbeddingModel,
    top_k: int = 5,
) -> list[VectorSearchResults]:
    """Search a vector index for similar content using pgvector cosine similarity."""
    # Accept both EmbeddingModel and string for embedding_model_choice
    if not isinstance(embedding_model, EmbeddingModel):
        try:
            embedding_model = EmbeddingModel(embedding_model)
        except Exception:
            raise ValueError(f"Unknown embedding model: {embedding_model}")

    embeddings = generate_embeddings(queries, model_choice=embedding_model)
    results = []
    for query_vector in embeddings:
        # pgvector query using cosine distance operator (<=>)
        # The query vector is passed twice because we use it in both SELECT and ORDER BY
        result = db_conn.execute(
            sql.SQL("""
                SELECT 
                    id,
                    source_article_id,
                    chunk_sequence_id,
                    chunk_text,
                    embedding <=> %s::vector AS distance
                FROM {}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """).format(sql.Identifier(vector_index_name)),
            (query_vector, query_vector, top_k),
        )

        results.append(result.fetchall())

    # Format results - note that results are dicts due to dict_row factory
    collected_results = [
        VectorSearchResults(
            results=[
                VectorSearchResult(
                    id_in_index=res["id"],  # type: ignore[call-overload]
                    source_article_id=str(res["source_article_id"]),  # type: ignore[call-overload]
                    chunk_seq=res["chunk_sequence_id"],  # type: ignore[call-overload]
                    chunk_text=res["chunk_text"],  # type: ignore[call-overload]
                    distance=res["distance"],  # type: ignore[call-overload]
                )
                for res in query_result
            ]
        )
        for query_result in results
    ]
    return collected_results


class RetrievalResult(BaseModel):
    """A single retrieval result from FTS or hybrid search."""

    id: int
    article_id: int
    chunk_sequence: int
    chunk_text: str
    score: float


def search_fts_chunks(
    db_conn: Connection[Any],
    vector_index_name: str,
    queries: list[str],
    top_k: int = 50,
) -> list[RetrievalResult]:
    """
    Search vector index chunks using PostgreSQL full-text search.

    Uses the generated tsvector column (chunk_text_tsv) with Danish language
    configuration for full-text search. Returns results ranked by ts_rank().

    Args:
        db_conn: PostgreSQL database connection
        vector_index_name: Name of the vector index table (must have chunk_text_tsv column)
        queries: List of query strings to search for
        top_k: Maximum number of results to return per query

    Returns:
        List of RetrievalResult objects with id, article_id, chunk_sequence,
        chunk_text, and score (ts_rank).
    """
    if not queries:
        return []

    results: list[RetrievalResult] = []

    for query in queries:
        if not query or not query.strip():
            continue

        # Use plainto_tsquery for natural language queries
        # It handles tokenization and Danish stemming automatically
        result = db_conn.execute(
            sql.SQL("""
                SELECT
                    id,
                    source_article_id,
                    chunk_sequence_id,
                    chunk_text,
                    ts_rank(chunk_text_tsv, plainto_tsquery('danish', %s)) AS score
                FROM {}
                WHERE chunk_text_tsv @@ plainto_tsquery('danish', %s)
                ORDER BY score DESC
                LIMIT %s
            """).format(sql.Identifier(vector_index_name)),
            (query, query, top_k),
        )

        for row in result.fetchall():
            results.append(
                RetrievalResult(
                    id=row["id"],  # type: ignore[call-overload]
                    article_id=row["source_article_id"],  # type: ignore[call-overload]
                    chunk_sequence=row["chunk_sequence_id"],  # type: ignore[call-overload]
                    chunk_text=row["chunk_text"],  # type: ignore[call-overload]
                    score=row["score"],  # type: ignore[call-overload]
                )
            )

    return results


def update_vector_index(
    db_conn: Connection[Any],
    vector_index_name: str,
    source_table: str,
    text_column: str,
    embedding_model_choice: EmbeddingModel,
    updated_at_column: str = "changed_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SECTIONS,
    batch_size: int = 64,
) -> dict[str, int]:
    """
    Update a vector index with new, modified, or deleted content.

    Compares timestamps between the vector index and source table to determine
    which articles need to be added, updated, or deleted.

    Args:
        db_conn: PostgreSQL database connection
        vector_index_name: Name of the vector index table
        source_table: Table containing source articles
        text_column: Column in source table with article text
        embedding_model_choice: Model to use for generating embeddings
        updated_at_column: Column tracking last update time (default: "changed_at")
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between consecutive chunks
        chunking_strategy: Strategy for splitting text into chunks
        batch_size: Number of chunks to process per batch

    Returns:
        Dictionary with stats: {"created": N, "deleted": M, "errors": K}
    """
    stats = {"created": 0, "deleted": 0, "errors": 0}

    # Get existing article timestamps from the vector index
    logger.info("Fetching existing article timestamps from vector index...")
    result = db_conn.execute(
        sql.SQL("SELECT source_article_id, last_updated FROM {}").format(
            sql.Identifier(vector_index_name)
        )
    )
    index_timestamps = {
        str(row["source_article_id"]): row["last_updated"] for row in result.fetchall()
    }  # type: ignore[call-overload]
    logger.info(
        f"Fetched {len(index_timestamps)} article timestamps from vector index."
    )

    # Get all articles from the source table
    logger.info("Fetching all articles from source table...")
    result = db_conn.execute(
        sql.SQL("SELECT id, headword, {}, {} FROM {}").format(
            sql.Identifier(updated_at_column),
            sql.Identifier(text_column),
            sql.Identifier(source_table),
        )
    )
    source_articles = {
        str(row["id"]): {  # type: ignore[call-overload]
            "headword": row["headword"],  # type: ignore[call-overload]
            "updated_at": row[updated_at_column],  # type: ignore[call-overload]
            "text": row[text_column],  # type: ignore[call-overload]
        }
        for row in result.fetchall()
    }
    logger.info(f"Fetched {len(source_articles)} articles from source table.")

    # Determine which articles need to be created, updated, or deleted
    index_ids = set(index_timestamps.keys())
    source_ids = set(source_articles.keys())
    ids_to_delete = index_ids - source_ids
    ids_to_create = source_ids - index_ids
    ids_to_update = {
        article_id
        for article_id in source_ids & index_ids
        if index_timestamps[article_id] < source_articles[article_id]["updated_at"]
    }
    ids_to_skip = index_ids & source_ids - ids_to_create - ids_to_update

    logger.info(f"IDs to create: {len(ids_to_create)}")
    logger.info(f"IDs to delete: {len(ids_to_delete)}")
    logger.info(f"IDs to update: {len(ids_to_update)}")
    logger.info(f"IDs to skip: {len(ids_to_skip)}")

    # Delete removed articles and articles that need updating
    last_message_index = 0
    for article_id in ids_to_delete | ids_to_update:
        try:
            chunks_deleted = remove_article_from_vector_index(
                db_conn, vector_index_name, article_id
            )
            stats["deleted"] += chunks_deleted
        except Exception as e:
            logger.error(f"Error removing article {article_id}: {str(e)}")
            stats["errors"] += 1

        # Give a status update on deletions every 100 chunks
        if (stats["deleted"] - last_message_index) > 100:
            logger.info(f"Chunks deleted: {stats['deleted']}")
            last_message_index = stats["deleted"]

    # Prepare chunks for new and updated articles
    chunks_to_create = []
    for article_id in ids_to_create | ids_to_update:
        article = source_articles[article_id]
        article_text = "# " + article["headword"] + "\n" + article["text"]
        if article_text:
            chunks = split_document_into_chunks(
                article_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
            )
            chunks_to_create += [
                (article_id, str(chunk_id), chunk_text)
                for chunk_id, chunk_text in enumerate(chunks)
            ]
        else:
            logger.warning(f"Article {article_id} has no text content.")
            stats["errors"] += 1

    # Add chunks in batches
    logger.info(f"Total chunks to create: {len(chunks_to_create)}")
    for i in range(0, len(chunks_to_create), batch_size):
        batch = chunks_to_create[i : i + batch_size]
        try:
            add_chunks_to_vector_index(
                db_conn=db_conn,
                vector_index_name=vector_index_name,
                chunks_data=batch,
                embedding_model_choice=embedding_model_choice,
            )
            stats["created"] += len(batch)
        except Exception as e:
            logger.error(f"Error adding batch of articles: {str(e)}")
            stats["errors"] += len(batch)

        logger.info(f"Chunks added: {stats['created']}/{len(chunks_to_create)}")

    # Update the metadata table's updated_at field for this index
    try:
        update_vector_index_metadata(db_conn, vector_index_name)
        logger.info(
            f"Updated metadata for index '{vector_index_name}' (updated_at field)."
        )
    except Exception as e:
        logger.warning(
            f"Could not update metadata for index '{vector_index_name}': {e}"
        )

    logger.info(f"Update summary: {stats}")
    return stats


def create_vector_index_metadata_table(db_conn: Connection[Any]) -> None:
    """Create the metadata table for vector indexes if it does not exist."""
    # Create the metadata table with PostgreSQL types
    db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_index_metadata (
            index_name TEXT PRIMARY KEY,
            source_table TEXT NOT NULL,
            source_column TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            chunking_strategy TEXT NOT NULL,
            updated_at_column TEXT NOT NULL DEFAULT 'changed_at',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
    )

    # Create trigger function for auto-updating updated_at
    db_conn.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
        """
    )

    # Drop existing trigger if it exists
    db_conn.execute(
        """
        DROP TRIGGER IF EXISTS update_vector_index_metadata_updated_at 
        ON vector_index_metadata
        """
    )

    # Create trigger for auto-updating updated_at
    db_conn.execute(
        """
        CREATE TRIGGER update_vector_index_metadata_updated_at
        BEFORE UPDATE ON vector_index_metadata
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column()
        """
    )

    db_conn.commit()


def insert_vector_index_metadata(
    db_conn: Connection[Any],
    index_name: str,
    source_table: str,
    source_column: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    updated_at_column: str = "changed_at",
) -> None:
    """Insert metadata for a new vector index."""
    create_vector_index_metadata_table(db_conn)

    # PostgreSQL uses %s placeholders and NOW() for timestamps
    db_conn.execute(
        """
        INSERT INTO vector_index_metadata (
            index_name, source_table, source_column, embedding_model, 
            chunk_size, chunk_overlap, chunking_strategy, updated_at_column
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            index_name,
            source_table,
            source_column,
            embedding_model,
            chunk_size,
            chunk_overlap,
            chunking_strategy,
            updated_at_column,
        ),
    )
    db_conn.commit()


def update_vector_index_metadata(
    db_conn: Connection[Any],
    index_name: str,
    **kwargs: Any,
) -> None:
    """Update metadata for an existing vector index. Only updates provided fields."""
    create_vector_index_metadata_table(db_conn)

    # Build SET clause with PostgreSQL placeholders
    fields = []
    values = []
    for key, value in kwargs.items():
        fields.append(f"{key} = %s")
        values.append(value)

    # No need to manually set updated_at - the trigger handles it
    values.append(index_name)
    sql = f"UPDATE vector_index_metadata SET {', '.join(fields)} WHERE index_name = %s"

    db_conn.execute(sql, values)
    db_conn.commit()


def get_all_vector_index_metadata(db_conn: Connection[Any]) -> list[dict]:
    """Retrieve metadata for all vector indexes."""
    create_vector_index_metadata_table(db_conn)
    rows = db_conn.execute("SELECT * FROM vector_index_metadata").fetchall()
    # rows are already dicts due to dict_row factory
    return [dict(row) for row in rows]  # type: ignore[call-overload]


def get_vector_index_metadata(db_conn: Connection[Any], index_name: str) -> dict | None:
    """Retrieve metadata for a specific vector index."""
    create_vector_index_metadata_table(db_conn)
    row = db_conn.execute(
        "SELECT * FROM vector_index_metadata WHERE index_name = %s", (index_name,)
    ).fetchone()
    if row:
        # row is already a dict due to dict_row factory
        return dict(row)  # type: ignore[call-overload]
    return None
