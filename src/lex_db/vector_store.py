"""Vector store operations for Lex DB."""

import json
import sqlite3
from datetime import datetime

from pydantic import BaseModel
from src.lex_db.utils import get_logger, split_document_into_chunks, ChunkingStrategy
from src.lex_db.embeddings import (
    EmbeddingModel,
    generate_embeddings,
    get_embedding_dimensions,
    generate_query_embedding,
)
from typing import Any

logger = get_logger()


def create_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    embedding_model_choice: EmbeddingModel,
    force: bool = False,
    source_table: str | None = None,
    source_column: str | None = None,
    updated_at_column: str = "changed_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.TOKENS,
) -> None:
    """Create a new vector index structure and store its metadata."""
    cursor = db_conn.cursor()
    embedding_dim = get_embedding_dimensions(embedding_model_choice)
    if force:
        drop_table_sql = f"DROP TABLE IF EXISTS {vector_index_name};"
        cursor.execute(drop_table_sql)
        logger.info(f"Dropped existing table {vector_index_name}")
    else:
        # Check if the table already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (vector_index_name,),
        )
        if cursor.fetchone():
            raise ValueError(
                f"Table {vector_index_name} already exists. Use --force-drop to drop it."
            )

    create_table_sql = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {vector_index_name} USING vec0(
        embedding FLOAT[{embedding_dim}],
        source_article_id TEXT,
        chunk_sequence_id INTEGER,
        chunk_text TEXT,
        last_updated TEXT)
    """
    cursor.execute(create_table_sql)
    logger.info(f"Virtual table {vector_index_name} created.")

    # Insert metadata if all required fields are provided
    if source_table and source_column:
        # Check if metadata entry already exists
        cursor.execute(
            "SELECT index_name FROM vector_index_metadata WHERE index_name = ?",
            (vector_index_name,),
        )
        if cursor.fetchone():
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


def add_single_article_to_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    article_rowid: str,
    article_text: str,
    embedding_model_choice: EmbeddingModel,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SECTIONS,
) -> None:
    """Add a single article to an existing vector index."""
    if not article_text:
        return

    cursor = db_conn.cursor()
    chunks = split_document_into_chunks(
        article_text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
    )

    if not chunks:
        return

    chunk_embeddings = generate_embeddings(chunks, model_choice=embedding_model_choice)

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
        insert_sql = f"""
        INSERT INTO {vector_index_name} (
            embedding, source_article_id, chunk_sequence_id, chunk_text, 
            last_updated
        )
        VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(
            insert_sql,
            (
                json.dumps(chunk_embedding),
                article_rowid,
                i,
                chunk_text,
                current_time_str,
            ),
        )

    db_conn.commit()
    logger.debug(f"Added article {article_rowid} to index {vector_index_name}.")


def add_chunks_to_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    chunks_data: list[
        tuple[str, str, str]
    ],  # List of (article_rowid, chunk_id, chunk_text)
    embedding_model_choice: EmbeddingModel,
) -> None:
    """Add multiple chunks to an existing vector index."""

    cursor = db_conn.cursor()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    embeddings = generate_embeddings(
        [chunk_text for _, _, chunk_text in chunks_data],
        model_choice=embedding_model_choice,
    )
    if not embeddings:
        logger.error("No embeddings generated for the provided chunks.")
        return
    if len(embeddings) != len(chunks_data):
        logger.error(
            "Mismatch between number of chunks and number of embeddings generated."
        )
        return

    for (article_rowid, chunk_id, chunk_text), embedding in zip(
        chunks_data, embeddings
    ):
        insert_sql = f"""
        INSERT INTO {vector_index_name} (
            embedding, source_article_id, chunk_sequence_id, chunk_text, 
            last_updated
        )
        VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(
            insert_sql,
            (
                json.dumps(embedding),
                article_rowid,
                int(chunk_id),  # chunk_sequence_id (not used in this context)
                chunk_text,
                current_time_str,
            ),
        )

    db_conn.commit()


def add_precomputed_embeddings_to_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    source_table: str,
    text_column: str,
    embeddings_data: list[
        tuple[str, str, str, list[float]]
    ],  # (article_id, chunk_idx, chunk_text, embedding)
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SECTIONS,
    batch_size: int = 16,
) -> dict[str, int]:
    """
    Add pre-computed embeddings to a vector index.

    This function is useful when you have embeddings from external sources
    (like OpenAI batch API) and need to reconstruct the chunk text from the source.
    """
    cursor = db_conn.cursor()
    stats = {"created": 0, "skipped": 0, "errors": 0}
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    logger.info(f"Processing {len(embeddings_data)} pre-computed embeddings")

    # Group embeddings by article_id to minimize database queries
    embeddings_by_article: dict[str, list[tuple[str, str, list[float]]]] = {}
    for article_id, chunk_idx, chunk_text, embedding in embeddings_data:
        if article_id not in embeddings_by_article:
            embeddings_by_article[article_id] = []
        embeddings_by_article[article_id].append((chunk_idx, chunk_text, embedding))

    # Process each article
    for article_id, article_embeddings in embeddings_by_article.items():
        try:
            # Get the original text if chunk_text is empty
            need_text = any(not chunk_text for _, chunk_text, _ in article_embeddings)

            original_chunks = []
            if need_text:
                # Fetch original text and recreate chunks
                cursor.execute(
                    f"SELECT {text_column} FROM {source_table} WHERE id = ?",
                    (article_id,),
                )
                row = cursor.fetchone()
                if not row or not row[0]:
                    logger.warning(f"No text found for article {article_id}, skipping")
                    stats["skipped"] += len(article_embeddings)
                    continue

                original_text = row[0]
                original_chunks = split_document_into_chunks(
                    original_text, chunk_size=chunk_size, overlap=chunk_overlap
                )

            # Process embeddings for this article in batches
            for i in range(0, len(article_embeddings), batch_size):
                batch = article_embeddings[i : i + batch_size]

                for chunk_idx, chunk_text, embedding in batch:
                    try:
                        # Use provided chunk_text or get from original_chunks
                        if not chunk_text and original_chunks:
                            if int(chunk_idx) < len(original_chunks):
                                chunk_text = original_chunks[int(chunk_idx)]
                            else:
                                logger.warning(
                                    f"Chunk index {chunk_idx} out of range for article {article_id}"
                                )
                                stats["errors"] += 1
                                continue
                        elif not chunk_text:
                            logger.warning(
                                f"No chunk text available for article {article_id}, chunk {chunk_idx}"
                            )
                            stats["errors"] += 1
                            continue

                        # Insert the embedding
                        insert_sql = f"""
                        INSERT OR REPLACE INTO {vector_index_name} (
                            embedding, source_article_id, chunk_sequence_id, chunk_text, 
                            last_updated
                        )
                        VALUES (?, ?, ?, ?, ?);
                        """
                        cursor.execute(
                            insert_sql,
                            (
                                json.dumps(embedding),
                                article_id,
                                chunk_idx,
                                chunk_text,
                                current_time_str,
                            ),
                        )
                        stats["created"] += 1

                    except Exception as e:
                        logger.error(
                            f"Error inserting embedding for article {article_id}, chunk {chunk_idx}: {e}"
                        )
                        stats["errors"] += 1

                # Commit after each batch
                db_conn.commit()

                if (i // batch_size + 1) % 10 == 0:  # Log progress every 10 batches
                    logger.info(f"Processed {stats['created']} embeddings so far...")

        except Exception as e:
            logger.error(f"Error processing article {article_id}: {e}")
            stats["errors"] += len(article_embeddings)

    logger.info(f"Pre-computed embeddings processing complete: {stats}")
    return stats


def remove_article_from_vector_index(
    db_conn: sqlite3.Connection, vector_index_name: str, article_rowid: str
) -> int:
    """Remove an article from a vector index."""
    cursor = db_conn.cursor()
    select_sql = (
        f"SELECT COUNT(*) FROM {vector_index_name} WHERE source_article_id = ?;"
    )
    cursor.execute(select_sql, (article_rowid,))
    count = cursor.fetchone()[0]
    delete_sql = f"DELETE FROM {vector_index_name} WHERE source_article_id = ?;"
    cursor.execute(delete_sql, (article_rowid,))
    db_conn.commit()
    logger.debug(
        f"Removed article {article_rowid} ({count} batches) from index {vector_index_name}."
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
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    query_text: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
) -> VectorSearchResults:
    """Search a vector index for similar content to the query text."""
    from src.lex_db.embeddings import EmbeddingModel

    cursor = db_conn.cursor()

    # Accept both EmbeddingModel and string for embedding_model_choice
    if not isinstance(embedding_model, EmbeddingModel):
        try:
            embedding_model = EmbeddingModel(embedding_model)
        except Exception:
            raise ValueError(f"Unknown embedding model: {embedding_model}")

    # Generate embedding for the query text
    query_vector = generate_query_embedding(query_text, model_choice=embedding_model)
    query_vector_json = json.dumps(query_vector)

    search_sql = f"""
    SELECT rowid, source_article_id, chunk_sequence_id, chunk_text, distance
    FROM {vector_index_name}
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT ?;
    """

    cursor.execute(search_sql, (query_vector_json, top_k))
    results = cursor.fetchall()

    # Format results
    formatted_results = [
        VectorSearchResult(
            id_in_index=res[0],
            source_article_id=res[1],
            chunk_seq=res[2],
            chunk_text=res[3],
            distance=res[4],
        )
        for res in results
    ]

    return VectorSearchResults(results=formatted_results)


def validate_tables_exist(
    db_conn: sqlite3.Connection,
    table_names: list[str],
) -> None:
    cursor = db_conn.cursor()
    for table in table_names:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if not cursor.fetchone():
            raise ValueError(f"Table '{table}' does not exist")


def update_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    source_table: str,
    text_column: str,
    embedding_model_choice: EmbeddingModel,
    updated_at_column: str = "updated_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SECTIONS,
    batch_size: int = 64,
) -> dict[str, int]:
    """Update a vector index with new, modified, or deleted content."""
    cursor = db_conn.cursor()
    stats = {"created": 0, "deleted": 0, "errors": 0}

    # Validate tables exist
    validate_tables_exist(
        db_conn,
        [vector_index_name, source_table],
    )

    # Get existing article timestamps from the vector index
    cursor.execute(f"SELECT source_article_id, last_updated FROM {vector_index_name}")
    logger.info("Fetching existing article timestamps from vector index...")
    index_timestamps = {str(row[0]): row[1] for row in cursor.fetchall()}
    logger.info("Fetched existing article timestamps from vector index.")

    # Get all articles from the source table
    cursor.execute(
        f"SELECT id, headword, {updated_at_column}, {text_column} FROM {source_table}"
    )
    logger.info("Fetching all articles from source table...")
    source_articles = {
        str(row[0]): {"headword": row[1], "updated_at": row[2], "text": row[3]}
        for row in cursor.fetchall()
    }
    logger.info("Fetched all articles from source table.")

    index_ids = set(index_timestamps.keys())
    source_ids = set(source_articles.keys())
    ids_to_delete = index_ids - source_ids
    ids_to_create = source_ids - index_ids
    ids_to_update = {
        article_id
        for article_id in source_ids & index_ids
        if datetime.strptime(index_timestamps[article_id], "%Y-%m-%d %H:%M:%S.%f")
        < datetime.strptime(
            source_articles[article_id]["updated_at"], "%Y-%m-%d %H:%M:%S.%f"
        )
    }
    ids_to_skip = index_ids & source_ids - ids_to_create - ids_to_update

    logger.info(f"IDs to create: {len(ids_to_create)}")
    logger.info(f"IDs to delete: {len(ids_to_delete)}")
    logger.info(f"IDs to update: {len(ids_to_update)}")
    logger.info(f"IDs to skip: {len(ids_to_skip)}")

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

        # Give a status update on deletions
        if (last_message_index - stats["deleted"]) > 100:
            logger.info(f"Chunks deleted: {stats['deleted']}.")
            last_message_index = stats["deleted"]

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

    db_conn.commit()

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


def create_vector_index_metadata_table(db_conn: sqlite3.Connection) -> None:
    """Create the metadata table for vector indexes if it does not exist."""
    cursor = db_conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_index_metadata (
            index_name TEXT PRIMARY KEY,
            source_table TEXT NOT NULL,
            source_column TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            chunking_strategy TEXT NOT NULL,
            updated_at_column TEXT NOT NULL DEFAULT 'updated_at', 
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    db_conn.commit()


def insert_vector_index_metadata(
    db_conn: sqlite3.Connection,
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
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    cursor = db_conn.cursor()
    cursor.execute(
        """
        INSERT INTO vector_index_metadata (
            index_name, source_table, source_column, embedding_model, chunk_size, chunk_overlap, chunking_strategy, updated_at_column, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            now,
            now,
        ),
    )
    db_conn.commit()


def update_vector_index_metadata(
    db_conn: sqlite3.Connection,
    index_name: str,
    **kwargs: Any,
) -> None:
    """Update metadata for an existing vector index. Only updates provided fields."""
    create_vector_index_metadata_table(db_conn)
    fields = []
    values = []
    for key, value in kwargs.items():
        fields.append(f"{key} = ?")
        values.append(value)
    fields.append("updated_at = ?")
    values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    values.append(index_name)
    sql = f"UPDATE vector_index_metadata SET {', '.join(fields)} WHERE index_name = ?"
    cursor = db_conn.cursor()
    cursor.execute(sql, values)
    db_conn.commit()


def get_all_vector_index_metadata(db_conn: sqlite3.Connection) -> list[dict]:
    """Retrieve metadata for all vector indexes."""
    create_vector_index_metadata_table(db_conn)
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM vector_index_metadata")
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_vector_index_metadata(
    db_conn: sqlite3.Connection, index_name: str
) -> dict | None:
    """Retrieve metadata for a specific vector index."""
    create_vector_index_metadata_table(db_conn)
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT * FROM vector_index_metadata WHERE index_name = ?", (index_name,)
    )
    row = cursor.fetchone()
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None
