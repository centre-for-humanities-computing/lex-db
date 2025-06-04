"""Vector store operations for Lex DB."""

import json
import sqlite3
from datetime import datetime

from pydantic import BaseModel
from src.lex_db.utils import get_logger, split_document_into_chunks
from src.lex_db.embeddings import (
    EmbeddingModel,
    generate_embeddings,
    get_embedding_dimensions,
    generate_query_embedding,
)

logger = get_logger()


def create_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    embedding_model_choice: EmbeddingModel,
    force: bool = False,
) -> None:
    """Create a new vector index structure."""
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
    logger.info("Index structure created. Use update_vector_indexes.py to populate it.")


def add_single_article_to_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    article_rowid: str,
    article_text: str,
    embedding_model_choice: EmbeddingModel,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """Add a single article to an existing vector index."""
    if not article_text:
        return

    cursor = db_conn.cursor()
    chunks = split_document_into_chunks(
        article_text, chunk_size=chunk_size, overlap=chunk_overlap
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


def remove_article_from_vector_index(
    db_conn: sqlite3.Connection, vector_index_name: str, article_rowid: str
) -> None:
    """Remove an article from a vector index."""
    cursor = db_conn.cursor()
    delete_sql = f"DELETE FROM {vector_index_name} WHERE source_article_id = ?;"
    cursor.execute(delete_sql, (article_rowid,))
    db_conn.commit()
    logger.info(f"Removed article {article_rowid} from index {vector_index_name}.")


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
    embedding_model_choice: EmbeddingModel,
    top_k: int = 5,
) -> VectorSearchResults:
    """Search a vector index for similar content to the query text."""
    cursor = db_conn.cursor()

    # Generate embedding for the query text
    query_vector = generate_query_embedding(
        query_text, model_choice=embedding_model_choice
    )
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


def update_vector_index(
    db_conn: sqlite3.Connection,
    vector_index_name: str,
    source_table: str,
    text_column: str,
    embedding_model_choice: str,
    updated_at_column: str = "updated_at",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = 100,
) -> dict[str, int]:
    """Update a vector index with new, modified, or deleted content."""
    cursor = db_conn.cursor()
    stats = {"new": 0, "updated": 0, "deleted": 0, "errors": 0, "unchanged": 0}

    # Validate tables exist
    for table in [vector_index_name, source_table]:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if not cursor.fetchone():
            raise ValueError(f"Table '{table}' does not exist")

    # Get existing article timestamps from the vector index
    cursor.execute(f"SELECT source_article_id, last_updated FROM {vector_index_name}")
    logger.info("Fetching existing article timestamps from vector index...")
    index_timestamps = {str(row[0]): row[1] for row in cursor.fetchall()}
    logger.info("Fetched existing article timestamps from vector index.")

    # Get all articles from the source table
    cursor.execute(f"SELECT id, {updated_at_column}, {text_column} FROM {source_table}")
    logger.info("Fetching all articles from source table...")
    source_articles = {
        str(row[0]): {"updated_at": row[1], "text": row[2]} for row in cursor.fetchall()
    }
    logger.info("Fetched all articles from source table.")
    index_ids = set(index_timestamps.keys())
    source_ids = set(source_articles.keys())

    ids_to_delete = index_ids - source_ids
    ids_to_create = source_ids - index_ids
    ids_to_check = source_ids & index_ids
    logger.info(f"IDs to create: {len(ids_to_create)}")
    logger.info(f"IDs to delete: {len(ids_to_delete)}")
    logger.info(f"IDs to check: {len(ids_to_check)}")

    for article_id in ids_to_create:
        article = source_articles[article_id]
        article_text = article["text"]
        article_updated_at = datetime.strptime(
            article["updated_at"], "%Y-%m-%d %H:%M:%S.%f"
        )

        if article_text:
            try:
                add_single_article_to_vector_index(
                    db_conn=db_conn,
                    vector_index_name=vector_index_name,
                    article_rowid=article_id,
                    article_text=article_text,
                    embedding_model_choice=EmbeddingModel(embedding_model_choice),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                stats["new"] += 1
            except Exception as e:
                logger.error(f"Error adding article {article_id}: {str(e)}")
                stats["errors"] += 1
        else:
            logger.warning(f"Article {article_id} has no text content.")
            stats["errors"] += 1

        # Commit every batch_size creations
        if (stats["new"] + 1) % batch_size == 0:
            logger.info(f"Committing after {stats['new'] + 1} creations.")
            db_conn.commit()

    for article_id in ids_to_delete:
        try:
            remove_article_from_vector_index(db_conn, vector_index_name, article_id)
            stats["deleted"] += 1
        except Exception as e:
            logger.error(f"Error removing article {article_id}: {str(e)}")
            stats["errors"] += 1

        # Commit every batch_size deletions
        if (stats["deleted"] + 1) % batch_size == 0:
            logger.info(f"Committing after {stats['deleted'] + 1} deletions.")
            db_conn.commit()

    # Commit after each batch
    db_conn.commit()

    for article_id in ids_to_check:
        article = source_articles[article_id]
        index_updated_at = datetime.strptime(
            index_timestamps[article_id], "%Y-%m-%d %H:%M:%S.%f"
        )
        article_updated_at = datetime.strptime(
            article["updated_at"], "%Y-%m-%d %H:%M:%S.%f"
        )
        if article_updated_at > index_updated_at:
            article_text = article["text"]
            if article_text:
                try:
                    remove_article_from_vector_index(
                        db_conn, vector_index_name, article_id
                    )
                    add_single_article_to_vector_index(
                        db_conn=db_conn,
                        vector_index_name=vector_index_name,
                        article_rowid=article_id,
                        article_text=article_text,
                        embedding_model_choice=EmbeddingModel(embedding_model_choice),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    stats["updated"] += 1
                except Exception as e:
                    logger.error(f"Error updating article {article_id}: {str(e)}")
                    stats["errors"] += 1
            else:
                logger.warning(f"Article {article_id} has no text content.")
                stats["errors"] += 1
        else:
            stats["unchanged"] += 1

        # Commit every batch_size updates
        if (stats["updated"] + 1) % batch_size == 0:
            logger.info(f"Committing after {stats['updated'] + 1} updates.")
            db_conn.commit()

        if (stats["unchanged"] + 1) % batch_size == 0:
            logger.info(f"Processed {stats['unchanged'] + 1} unchanged articles.")

    db_conn.commit()

    logger.info(f"Update summary: {stats}")
    return stats
