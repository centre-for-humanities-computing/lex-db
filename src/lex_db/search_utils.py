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
from pydantic import BaseModel


# Default Danish stopwords for FTS5 queries
DEFAULT_STOPWORDS: Set[str] = {
    "og",
    "i",
    "på",
    "det",
    "du",
    "mig",
    "en",
    "den",
    "at",
    "til",
    "der",
    "da",
    "af",
    "de",
    "han",
    "hun",
    "fra",
    "som",
    "et",
    "var",
    "for",
    "ikke",
    "kan",
    "hans",
    "er",
    "mellem",
    "havde",
    "ham",
    "hendes",
    "sig",
    "eller",
    "hvad",
    "hvilke",
    "hvordan",
    "hvorfor",
    "hvornår",
    "vores",
    "jeres",
    "deres",
    "min",
    "din",
    "sit",
    "mit",
    "dit",
    "denne",
    "dette",
    "disse",
    "være",
    "bliver",
    "blev",
    "vil",
    "ville",
    "skal",
    "skulle",
    "har",
    "have",
    "med",
    "om",
    "så",
    "når",
}


def sanitize_fts_query(
    query: str, stopwords: Set[str] = DEFAULT_STOPWORDS
) -> Optional[str]:
    """Clean query for FTS5 MATCH syntax."""

    cleaned = "".join(c if c.isalnum() or c in " æøåÆØÅ-" else " " for c in query)

    words = [
        w.strip()
        for w in cleaned.split()
        if w.strip() and w.strip().lower() not in stopwords
    ]

    if not words:
        return None

    return " OR ".join(f'"{w}"' for w in words)


class RetrievalResult(BaseModel):
    """A single retrieval result."""

    rowid: int
    article_id: int
    chunk_sequence: int
    chunk_text: str
    score: float


def search_fts5(
    conn: Connection,
    vector_index: str,
    fts_index: str,
    queries: list[str],
    top_k: int,
    stopwords: Set[str] = DEFAULT_STOPWORDS,
) -> list[RetrievalResult]:
    """Get ranked candidates from FTS5 keyword search."""

    fts_queries = [sanitize_fts_query(query, stopwords) for query in queries]

    if not fts_queries or all(q is None for q in fts_queries):
        return []

    cursor = conn.cursor()

    results: list[RetrievalResult] = []
    try:
        for fts_query in fts_queries:
            if fts_query is None:
                continue

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

            results.extend(
                [
                    RetrievalResult(
                        rowid=row[0],
                        article_id=row[1],
                        chunk_sequence=row[2],
                        chunk_text=row[3],
                        score=row[4],
                    )
                    for row in cursor.fetchall()
                ]
            )

        return results
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
    ranks: list[int],
    rrf_k: int,
    weights: Optional[list[float]] = None,
    normalize: bool = False,
) -> float:
    """Calculate Reciprocal Rank Fusion (RRF) score for a set of ranks."""
    if weights is None:
        weights = [1.0] * len(ranks)
    if len(weights) != len(ranks):
        raise ValueError("Length of weights must match length of ranks")

    score = 0.0
    for rank, weight in zip(ranks, weights):
        score += weight * (1.0 / (rrf_k + rank))

    if normalize:
        total_weight = sum(weights)
        if total_weight > 0:
            score = score / total_weight

    return score


def fuse_results_rrf(
    results: list[list[RetrievalResult]],
    rrf_k: int,
    weights: Optional[list[float]] = None,
    normalize: bool = False,
) -> List[RetrievalResult]:
    """Fuse two sets of search results using Reciprocal Rank Fusion."""
    combined: dict[int, tuple[list[RetrievalResult], list[float]]] = {}
    if weights is None:
        weights = [1.0 for _ in results]

    if len(weights) != len(results):
        raise ValueError("Length of weights must match number of result sets")

    for method_weight, method_results in zip(weights, results):
        for r in method_results:
            rid = r.rowid
            if rid not in combined:
                combined[rid] = ([], [])
            combined[rid][0].append(r)
            combined[rid][1].append(method_weight)
    fused_results: List[RetrievalResult] = []
    for rid, (res_list, weight_list) in combined.items():
        ranks = list(range(1, len(res_list) + 1))
        score = calculate_rrf_score(ranks, rrf_k, weight_list, normalize)
        fused_results.append(
            RetrievalResult(
                rowid=rid,
                article_id=res_list[0].article_id,
                chunk_sequence=res_list[0].chunk_sequence,
                chunk_text=res_list[0].chunk_text,
                score=score,
            )
        )

    return sorted(fused_results, key=lambda x: x.score, reverse=True)
