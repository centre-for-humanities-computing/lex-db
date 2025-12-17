"""Shared utilities for search implementations in Lex DB.

This module provides common functionality used across different search methods:
- FTS5 query sanitization
- Vector/semantic search
- FTS5 keyword search
- Article metadata retrieval
- RRF fusion
- Embedding utilities
"""

import sqlite3
from typing import List, Dict, Optional, Set
from sqlite3 import Connection
import numpy as np

from lex_db.embeddings import EmbeddingModel, generate_passage_embedding
from lex_db.vector_store import search_vector_index


# Default Danish stopwords for FTS5 queries
DEFAULT_STOPWORDS: Set[str] = {
    "og", "i", "på", "det", "en", "den", "at", "til", "der", "da", "af", "de",
    "han", "hun", "fra", "som", "et", "var", "for", "ikke", "kan", "hans", "er",
    "mellem", "havde", "ham", "hendes", "sig", "eller", "hvad", "hvilke",
    "hvordan", "hvorfor", "denne", "dette", "disse", "være", "bliver", "blev",
    "vil", "ville", "skal", "skulle", "har", "have", "med", "om", "så", "når",
}


def sanitize_fts_query(query: str, stopwords: Optional[Set[str]] = None) -> Optional[str]:
    """Clean query for FTS5 MATCH syntax.

    Args:
        query: Raw search query
        stopwords: Set of stopwords to filter out (uses DEFAULT_STOPWORDS if None)

    Returns:
        Sanitized FTS5 query string, or None if no valid words remain
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    cleaned = "".join(c if c.isalnum() or c in " æøåÆØÅ-" else " " for c in query)

    words = [
        w.strip()
        for w in cleaned.split()
        if w.strip() and w.strip().lower() not in stopwords
    ]

    if not words:
        return None

    return " OR ".join(f'"{w}"' for w in words)


def search_semantic(
    conn: Connection,
    vector_index: str,
    query: str,
    embedding_model: EmbeddingModel,
    top_k: int
) -> List[Dict]:
    """Get ranked candidates from semantic vector search.

    Args:
        conn: Database connection
        vector_index: Name of the vector index to search
        query: Search query text
        embedding_model: Model to use for query embedding
        top_k: Number of results to return

    Returns:
        List of search results with semantic_rank
    """
    results = search_vector_index(
        db_conn=conn,
        vector_index_name=vector_index,
        query_text=query,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    return [
        {
            "rowid": r.id_in_index,
            "article_id": r.source_article_id,
            "chunk_seq": r.chunk_seq,
            "chunk_text": r.chunk_text,
            "semantic_rank": rank,
        }
        for rank, r in enumerate(results.results, 1)
    ]


def search_fts5(
    conn: Connection,
    vector_index: str,
    fts_index: str,
    query: str,
    top_k: int,
    stopwords: Optional[Set[str]] = None
) -> List[Dict]:
    """Get ranked candidates from FTS5 keyword search.

    Args:
        conn: Database connection
        vector_index: Name of the vector index (for joining with FTS results)
        fts_index: Name of the FTS5 index
        query: Search query text
        top_k: Number of results to return
        stopwords: Set of stopwords to filter out

    Returns:
        List of search results with fts_rank

    Raises:
        RuntimeError: If FTS5 query fails
    """
    fts_query = sanitize_fts_query(query, stopwords)

    if fts_query is None:
        return []

    cursor = conn.cursor()

    try:
        cursor.execute(
            f"""
            SELECT
                fts.rowid,
                vec.source_article_id,
                vec.chunk_sequence_id,
                vec.chunk_text,
                bm25({fts_index}) as bm25_score
            FROM {fts_index} fts
            JOIN {vector_index} vec ON fts.rowid = vec.rowid
            WHERE {fts_index} MATCH ?
            ORDER BY bm25({fts_index})
            LIMIT ?
        """,
            (fts_query, top_k),
        )

        return [
            {
                "rowid": row[0],
                "article_id": row[1],
                "chunk_seq": row[2],
                "chunk_text": row[3],
                "fts_rank": rank,
            }
            for rank, row in enumerate(cursor.fetchall(), 1)
        ]
    except sqlite3.Error as e:
        raise RuntimeError(f"FTS5 search failed: {e}") from e


def get_article_headwords(conn: Connection, article_ids: List[int]) -> Dict[int, str]:
    """Fetch article headwords for display.

    Args:
        conn: Database connection
        article_ids: List of article IDs to fetch headwords for

    Returns:
        Dictionary mapping article_id to headword
    """
    if not article_ids:
        return {}

    cursor = conn.cursor()
    placeholders = ",".join("?" * len(article_ids))
    cursor.execute(
        f"SELECT id, headword FROM articles WHERE id IN ({placeholders})",
        article_ids,
    )
    return {row[0]: row[1] for row in cursor.fetchall()}


def calculate_rrf_score(
    rank1: Optional[int],
    rank2: Optional[int],
    rrf_k: int,
    weight1: float = 1.0,
    weight2: float = 1.0,
    normalize: bool = False
) -> float:
    """Calculate Reciprocal Rank Fusion (RRF) score.

    Args:
        rank1: Rank from first retrieval method (None if not present)
        rank2: Rank from second retrieval method (None if not present)
        rrf_k: RRF constant (typically 60)
        weight1: Weight for first method (default 1.0)
        weight2: Weight for second method (default 1.0)
        normalize: Whether to normalize by total weight (default False)

    Returns:
        RRF score

    Formula: score = weight1 * 1/(k + rank1) + weight2 * 1/(k + rank2)
    If normalize=True: score = score / (weight1 + weight2)
    """
    score = 0.0
    if rank1 is not None:
        score += weight1 * (1.0 / (rrf_k + rank1))
    if rank2 is not None:
        score += weight2 * (1.0 / (rrf_k + rank2))

    if normalize:
        total_weight = weight1 + weight2
        if total_weight > 0:
            score = score / total_weight

    return score


def fuse_results_rrf(
    results1: List[Dict],
    results2: List[Dict],
    rank1_key: str,
    rank2_key: str,
    source1_label: str,
    source2_label: str,
    rrf_k: int,
    weight1: float = 1.0,
    weight2: float = 1.0,
    normalize: bool = False
) -> List[Dict]:
    """Fuse two sets of search results using Reciprocal Rank Fusion.

    Args:
        results1: Results from first retrieval method
        results2: Results from second retrieval method
        rank1_key: Key name for rank in results1 (e.g., "semantic_rank", "hyde_rank")
        rank2_key: Key name for rank in results2 (e.g., "fts_rank")
        source1_label: Label for source1 (e.g., "SEMANTIC", "HYDE")
        source2_label: Label for source2 (e.g., "FTS5")
        rrf_k: RRF constant (typically 60)
        weight1: Weight for first method (default 1.0)
        weight2: Weight for second method (default 1.0)
        normalize: Whether to normalize scores by total weight (default False)

    Returns:
        List of fused results sorted by RRF score (descending)
    """
    combined = {}

    # Add results from first method
    for r in results1:
        rid = r["rowid"]
        combined[rid] = {
            "rowid": rid,
            "article_id": r["article_id"],
            "chunk_seq": r["chunk_seq"],
            "chunk_text": r["chunk_text"],
            rank1_key: r[rank1_key],
            rank2_key: None,
            "source": source1_label,
        }

    # Add/merge results from second method
    for r in results2:
        rid = r["rowid"]
        if rid in combined:
            combined[rid][rank2_key] = r[rank2_key]
            combined[rid]["source"] = "BOTH"
        else:
            combined[rid] = {
                "rowid": rid,
                "article_id": r["article_id"],
                "chunk_seq": r["chunk_seq"],
                "chunk_text": r["chunk_text"],
                rank1_key: None,
                rank2_key: r[rank2_key],
                "source": source2_label,
            }

    # Calculate RRF scores
    for data in combined.values():
        data["rrf_score"] = calculate_rrf_score(
            data[rank1_key], data[rank2_key], rrf_k, weight1, weight2, normalize
        )

    return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)


def embed_passage(passage: str, model: EmbeddingModel = EmbeddingModel.LOCAL_E5_MULTILINGUAL) -> np.ndarray:
    """Embed passage text using specified embedding model.

    Args:
        passage: Text to embed
        model: Embedding model to use (default: LOCAL_E5_MULTILINGUAL)

    Returns:
        Numpy array of embedding vector
    """
    embedding = generate_passage_embedding(passage, model)
    return np.array(embedding, dtype=np.float32)


def search_by_embedding(
    conn: Connection,
    vector_index: str,
    embedding: np.ndarray,
    top_k: int,
    rank_key: str = "rank",
    include_distance: bool = False
) -> List[Dict]:
    """Search vector index using an embedding vector.

    Args:
        conn: Database connection
        vector_index: Name of the vector index to search
        embedding: Embedding vector as numpy array
        top_k: Number of results to return
        rank_key: Key name for rank in results (default: "rank")
        include_distance: Whether to include distance in results (default: False)

    Returns:
        List of search results with specified rank key
    """
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT rowid, source_article_id, chunk_sequence_id, chunk_text, distance
        FROM {vector_index}
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """,
        (embedding.tobytes(), top_k),
    )

    results = []
    for rank, row in enumerate(cursor.fetchall(), 1):
        result = {
            "rowid": row[0],
            "article_id": row[1],
            "chunk_seq": row[2],
            "chunk_text": row[3],
            rank_key: rank,
        }
        if include_distance:
            result["distance"] = row[4]
        results.append(result)

    return results
