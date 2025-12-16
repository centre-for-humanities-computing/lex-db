"""
Hybrid Search - Reciprocal Rank Fusion (RRF)

Two-stage retrieval:
1. Get candidates from BOTH semantic search AND keyword search (FTS5)
2. Combine using RRF scoring (no neural reranker needed)

RRF Formula: score(doc) = Σ 1 / (k + rank_in_system)

This ensures entity-specific queries include chunks that
contain the entity name, not just semantically similar chunks.

USAGE:
    python hybrid_search.py
"""

import sqlite3
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pydantic import BaseModel
from lex_db.vector_store import search_vector_index
from lex_db.embeddings import EmbeddingModel


# ===============================================================
# CONFIGURATION
# ===============================================================

VECTOR_INDEX = "article_embeddings_e5"
FTS_INDEX = "fts_article_embeddings_e5"

TOP_K_SEMANTIC = 50
TOP_K_FTS = 50
TOP_K_FINAL = 10
RRF_K = 60

DEFAULT_STOPWORDS: Set[str] = {
"og", "i", "på", "det", "en", "den", "at", "til", "der", "da", "af", "de",
"han", "hun", "fra", "som", "et", "var", "for", "ikke", "kan", "hans", "er",
"mellem", "havde", "ham", "hendes", "sig", "eller", "hvad", "hvilke",
"hvordan", "hvorfor", "denne", "dette", "disse", "være", "bliver", "blev",
"vil", "ville", "skal", "skulle", "har", "have", "med", "om", "så", "når",
}


# ===============================================================
# DATA CLASSES
# ===============================================================


@dataclass
class SearchResult:
    """A single search result."""

    rank: int
    article_id: int
    article_headword: str
    chunk_sequence: int
    chunk_text: str
    rrf_score: float
    semantic_rank: Optional[int]
    fts_rank: Optional[int]
    source: str  # 'SEMANTIC', 'FTS5', or 'BOTH'


class HybridSearchResults(BaseModel):
    """Results of a hybrid search."""

    results: list[SearchResult]


# ===============================================================
# HYBRID SEARCH WITH RRF
# ===============================================================


class HybridSearch:
    """Two-stage retrieval: semantic + keyword candidates → RRF fusion."""

    def __init__(
        self,
        vector_index: str = VECTOR_INDEX,
        fts_index: str = FTS_INDEX,
        rrf_k: int = RRF_K,
        stopwords: Optional[Set[str]] = None,
    ):
        self.vector_index = vector_index
        self.fts_index = fts_index
        self.rrf_k = rrf_k
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.conn = None

    def sanitize_fts_query(self, query: str) -> Optional[str]:
        """Clean query for FTS5 MATCH syntax."""
        cleaned = "".join(c if c.isalnum() or c in " æøåÆØÅ-" else " " for c in query)

        words = [
            w.strip()
            for w in cleaned.split()
            if w.strip() and w.strip().lower() not in self.stopwords
        ]

        if not words:
            return None

        return " OR ".join(f'"{w}"' for w in words)

    def search_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Get ranked candidates from semantic search."""
        if self.conn is None:
            raise ValueError(
                "Database connection not initialized. Assign connection to searcher.conn before calling search methods."
            )
        results = search_vector_index(
            db_conn=self.conn,
            vector_index_name=self.vector_index,
            query_text=query,
            embedding_model=EmbeddingModel.LOCAL_E5_MULTILINGUAL,
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

    def search_fts5(self, query: str, top_k: int) -> List[Dict]:
        """Get ranked candidates from FTS5 keyword search."""
        fts_query = self.sanitize_fts_query(query)

        if fts_query is None:
            return []

        if self.conn is None:
            raise ValueError(
                "Database connection not initialized. Assign connection to searcher.conn before calling search methods."
            )

        cursor = self.conn.cursor()

        try:
            cursor.execute(
                f"""
                SELECT 
                    fts.rowid,
                    vec.source_article_id,
                    vec.chunk_sequence_id,
                    vec.chunk_text,
                    bm25({self.fts_index}) as bm25_score
                FROM {self.fts_index} fts
                JOIN {self.vector_index} vec ON fts.rowid = vec.rowid
                WHERE {self.fts_index} MATCH ?
                ORDER BY bm25({self.fts_index})
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

    def calculate_rrf_score(
        self, semantic_rank: Optional[int], fts_rank: Optional[int]
    ) -> float:
        """Calculate RRF score: RRF(d) = Σ 1 / (k + rank)"""
        score = 0.0
        if semantic_rank is not None:
            score += 1.0 / (self.rrf_k + semantic_rank)
        if fts_rank is not None:
            score += 1.0 / (self.rrf_k + fts_rank)
        return score

    def fuse_results(
        self, semantic_results: List[Dict], fts_results: List[Dict]
    ) -> List[Dict]:
        """Fuse semantic and FTS results using RRF."""
        combined = {}

        for r in semantic_results:
            rid = r["rowid"]
            combined[rid] = {
                "rowid": rid,
                "article_id": r["article_id"],
                "chunk_seq": r["chunk_seq"],
                "chunk_text": r["chunk_text"],
                "semantic_rank": r["semantic_rank"],
                "fts_rank": None,
                "source": "SEMANTIC",
            }

        for r in fts_results:
            rid = r["rowid"]
            if rid in combined:
                combined[rid]["fts_rank"] = r["fts_rank"]
                combined[rid]["source"] = "BOTH"
            else:
                combined[rid] = {
                    "rowid": rid,
                    "article_id": r["article_id"],
                    "chunk_seq": r["chunk_seq"],
                    "chunk_text": r["chunk_text"],
                    "semantic_rank": None,
                    "fts_rank": r["fts_rank"],
                    "source": "FTS5",
                }

        for data in combined.values():
            data["rrf_score"] = self.calculate_rrf_score(
                data["semantic_rank"], data["fts_rank"]
            )

        return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

    def get_article_headwords(self, article_ids: List[int]) -> Dict[int, str]:
        """Fetch article headwords for display."""
        if not article_ids:
            return {}

        if self.conn is None:
            raise ValueError(
                "Database connection not initialized. Assign connection to searcher.conn before calling search methods."
            )

        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(article_ids))
        cursor.execute(
            f"SELECT id, headword FROM articles WHERE id IN ({placeholders})",
            article_ids,
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        top_k_semantic: int = TOP_K_SEMANTIC,
        top_k_fts: int = TOP_K_FTS,
    ) -> HybridSearchResults:
        """Hybrid search with RRF fusion."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        # Stage 1a: Semantic search
        semantic_results = self.search_semantic(query, top_k_semantic)

        # Stage 1b: FTS5 keyword search
        fts_results = self.search_fts5(query, top_k_fts)

        # Stage 2: RRF Fusion
        fused = self.fuse_results(semantic_results, fts_results)

        # Build final results
        top_results = fused[:top_k]
        article_ids = list(set(int(r["article_id"]) for r in top_results))
        headwords = self.get_article_headwords(article_ids)

        return HybridSearchResults(
            results=[
                SearchResult(
                    rank=rank,
                    article_id=int(r["article_id"]),
                    article_headword=headwords.get(int(r["article_id"]), "Unknown"),
                    chunk_sequence=r["chunk_seq"],
                    chunk_text=r["chunk_text"],
                    rrf_score=round(r["rrf_score"], 6),
                    semantic_rank=r["semantic_rank"],
                    fts_rank=r["fts_rank"],
                    source=r["source"],
                )
                for rank, r in enumerate(top_results, 1)
            ]
        )
