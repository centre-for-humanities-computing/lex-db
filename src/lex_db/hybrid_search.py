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

from typing import Optional, Set
from dataclasses import dataclass
from pydantic import BaseModel
from lex_db.embeddings import EmbeddingModel
from lex_db import search_utils


# ===============================================================
# CONFIGURATION
# ===============================================================

VECTOR_INDEX = "article_embeddings_e5"
FTS_INDEX = "fts_article_embeddings_e5"

TOP_K_SEMANTIC = 50
TOP_K_FTS = 50
TOP_K_FINAL = 10
RRF_K = 60


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
        self.stopwords = stopwords if stopwords is not None else search_utils.DEFAULT_STOPWORDS
        self.conn = None

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

        if self.conn is None:
            raise ValueError(
                "Database connection not initialized. Assign connection to searcher.conn before calling search methods."
            )

        query = query.strip()

        # Stage 1a: Semantic search
        semantic_results = search_utils.search_semantic(
            conn=self.conn,
            vector_index=self.vector_index,
            query=query,
            embedding_model=EmbeddingModel.LOCAL_E5_MULTILINGUAL,
            top_k=top_k_semantic,
        )

        # Stage 1b: FTS5 keyword search
        fts_results = search_utils.search_fts5(
            conn=self.conn,
            vector_index=self.vector_index,
            fts_index=self.fts_index,
            query=query,
            top_k=top_k_fts,
            stopwords=self.stopwords,
        )

        # Stage 2: RRF Fusion
        fused = search_utils.fuse_results_rrf(
            results1=semantic_results,
            results2=fts_results,
            rank1_key="semantic_rank",
            rank2_key="fts_rank",
            source1_label="SEMANTIC",
            source2_label="FTS5",
            rrf_k=self.rrf_k,
            weight1=1.0,
            weight2=1.0,
            normalize=False
        )

        # Build final results
        top_results = fused[:top_k]
        article_ids = list(set(int(r["article_id"]) for r in top_results))
        headwords = search_utils.get_article_headwords(self.conn, article_ids)

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
