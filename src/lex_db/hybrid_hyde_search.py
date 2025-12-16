"""
Hybrid Search: HyDE + FTS with Adaptive RRF Weighting

USAGE:
    export OPENROUTER_API_KEY="your-key"
    python hybrid_hyde_search.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import requests # type: ignore[import-untyped]

from lex_db.database import create_connection
from lex_db.embeddings import EmbeddingModel, generate_passage_embedding

# ===============================================================
# CONFIGURATION
# ===============================================================

VECTOR_INDEX = "article_embeddings_e5"
FTS_INDEX = "fts_article_embeddings_e5"
TOP_K_HYDE = 50
TOP_K_FTS = 50
TOP_K_FINAL = 10
RRF_K = 60

OPENROUTER_MODEL = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HYDE_PROMPT = """Du er en dansk leksikonskribent.

Analyser først spørgsmålet og klassificer det som ÉN af:
- ENTITY: Spørgsmål om en specifik person, sted eller ting (f.eks. "Christian IV", "Hvem var Kierkegaard?")
- CONCEPTUAL: Spørgsmål om ideer, processer, forklaringer eller generelle emner (f.eks. "Hvad er eksistentialisme?", "Hvordan opstod renæssancen?")

Skriv derefter et kort encyklopædisk afsnit (3-4 sætninger) der besvarer spørgsmålet.

Spørgsmål: {query}

Svar i dette format:
QUERYTYPE: [ENTITY eller CONCEPTUAL]
PASSAGE: [dit encyklopædiske afsnit her - plain text uden formatering]

Regler for afsnittet:
- Skriv i plain text uden formatering (ingen **, ##, eller punktopstillinger)
- Nævn relevante navne, begreber og årstal
- Skriv i encyklopædisk stil, ikke som en AI-assistent
- Hvis du er usikker på detaljer, hold dig til generelle fakta"""

DEFAULT_STOPWORDS: Set[str] = {
"og", "i", "på", "det", "en", "den", "at", "til", "der", "da", "af", "de",
"han", "hun", "fra", "som", "et", "var", "for", "ikke", "kan", "hans", "er",
"mellem", "havde", "ham", "hendes", "sig", "eller", "hvad", "hvilke",
"hvordan", "hvorfor", "denne", "dette", "disse", "være", "bliver", "blev",
"vil", "ville", "skal", "skulle", "har", "have", "med", "om", "så", "når",
}

# ===============================================================
# QUERY TYPE CLASSIFICATION
# ===============================================================


class QueryType(Enum):
    """Query types for adaptive weighting (LLM-based classification)."""

    ENTITY = "entity"
    CONCEPTUAL = "conceptual"


@dataclass
class QueryWeights:
    hyde_weight: float
    fts_weight: float
    description: str


QUERY_WEIGHTS = {
    QueryType.ENTITY: QueryWeights(
        0.8, 1.2, "Entity-focused: FTS boosted for exact name matching"
    ),
    QueryType.CONCEPTUAL: QueryWeights(
        1.2, 0.8, "Conceptual: HyDE boosted for semantic understanding"
    ),
}


# ===============================================================
# DATA CLASSES
# ===============================================================


@dataclass
class HybridHyDESearchResults:
    rank: int
    article_id: int
    article_headword: str
    chunk_sequence: int
    chunk_text: str
    rrf_score: float
    hyde_rank: Optional[int]
    fts_rank: Optional[int]
    source: str


# ===============================================================
# LLM PASSAGE GENERATION
# ===============================================================


def generate_hypothetical_passage(query: str) -> tuple[str, QueryType]:
    """Generate hypothetical passage and classify query using LLM via OpenRouter.

    Returns:
        tuple: (passage_text, query_type)

    Raises:
        ValueError: If API key is not set or response is malformed
        requests.exceptions.RequestException: For network/API errors
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "user", "content": HYDE_PROMPT.format(query=query)}
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()

        # Parse and validate response
        try:
            response_data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from OpenRouter: {e}") from e

        # Extract content with validation
        try:
            content = response_data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("Empty response from LLM")

            content = content.strip()

            # Parse the formatted response
            query_type = QueryType.CONCEPTUAL  # Default fallback
            passage = content

            # Try to parse QUERYTYPE and PASSAGE
            if "QUERYTYPE:" in content and "PASSAGE:" in content:
                lines = content.split("\n")
                querytype_line = ""
                passage_lines = []
                found_passage = False

                for line in lines:
                    if line.startswith("QUERYTYPE:"):
                        querytype_line = line
                    elif line.startswith("PASSAGE:"):
                        found_passage = True
                        passage_lines.append(line.replace("PASSAGE:", "").strip())
                    elif found_passage:
                        passage_lines.append(line)

                # Extract query type
                if "ENTITY" in querytype_line.upper():
                    query_type = QueryType.ENTITY
                else:
                    query_type = QueryType.CONCEPTUAL  # Default

                # Extract passage
                if passage_lines:
                    passage = " ".join(passage_lines).strip()

            # Validation: ensure we have a passage
            if not passage:
                raise ValueError("Could not extract passage from LLM response")

            return passage, query_type

        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Malformed response structure from OpenRouter: {e}"
            ) from e

    except requests.exceptions.Timeout:
        raise requests.exceptions.RequestException(
            f"OpenRouter API request timed out after 30 seconds for query: {query[:50]}..."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            raise requests.exceptions.RequestException(
                "OpenRouter API rate limit exceeded. Please try again later."
            ) from e
        elif e.response.status_code == 401:
            raise ValueError("Invalid OpenRouter API key") from e
        elif e.response.status_code >= 500:
            raise requests.exceptions.RequestException(
                f"OpenRouter API server error (status {e.response.status_code})"
            ) from e
        else:
            raise requests.exceptions.RequestException(
                f"OpenRouter API error: {e.response.status_code} - {e.response.text}"
            ) from e
    except requests.exceptions.ConnectionError as e:
        raise requests.exceptions.RequestException(
            "Failed to connect to OpenRouter API. Check your internet connection."
        ) from e
    except requests.exceptions.RequestException as e:
        # Catch-all for other request errors
        raise requests.exceptions.RequestException(
            f"OpenRouter API request failed: {str(e)}"
        ) from e


# ===============================================================
# HYBRID SEARCH WITH ADAPTIVE WEIGHTING
# ===============================================================


class HybridHyDESearch:
    """Hybrid retrieval with adaptive RRF weighting based on query type."""

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

    def embed_passage(self, passage: str) -> np.ndarray:
        """Embed passage text using local E5 model."""
        embedding = generate_passage_embedding(
            passage, EmbeddingModel.LOCAL_E5_MULTILINGUAL
        )
        return np.array(embedding, dtype=np.float32)

    def search_by_embedding(self, embedding: np.ndarray, top_k: int) -> list[dict]:
        """Search vector index using embedding."""
        conn = create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT rowid, source_article_id, chunk_sequence_id, chunk_text, distance
                FROM {self.vector_index}
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """,
                (embedding.tobytes(), top_k),
            )

            return [
                {
                    "rowid": row[0],
                    "article_id": row[1],
                    "chunk_seq": row[2],
                    "chunk_text": row[3],
                    "hyde_rank": rank,
                }
                for rank, row in enumerate(cursor.fetchall(), 1)
            ]
        finally:
            conn.close()

    def search_hyde(self, query: str, top_k: int) -> tuple[list[dict], str, QueryType]:
        """Perform HyDE search and classify query.

        Returns:
            tuple: (results, hypothetical_passage, query_type)
        """
        hypothetical, query_type = generate_hypothetical_passage(query)

        embedding = self.embed_passage(hypothetical)

        results = self.search_by_embedding(embedding, top_k)

        return results, hypothetical, query_type

    def sanitize_fts_query(self, query: str) -> Optional[str]:
        """Clean query for FTS5 MATCH syntax."""
        cleaned = "".join(c if c.isalnum() or c in " æøåÆØÅ-" else " " for c in query)

        # Filter stopwords efficiently (single strip per word)
        words = []
        for w in cleaned.split():
            w = w.strip()
            if w and w.lower() not in self.stopwords:
                words.append(w)

        if not words:
            return None

        phrase = f'"{" ".join(words)}"'
        individual_terms = " OR ".join(f'"{w}"' for w in words)

        return f"{phrase} OR ({individual_terms})"

    def search_fts5(self, query: str, top_k: int) -> list[dict]:
        """Get ranked candidates from FTS5 keyword search."""
        fts_query = self.sanitize_fts_query(query)

        if fts_query is None:
            return []

        conn = create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT fts.rowid, vec.source_article_id, vec.chunk_sequence_id, vec.chunk_text, bm25({self.fts_index}) as bm25_score
                FROM {self.fts_index} fts
                JOIN {self.vector_index} vec ON fts.rowid = vec.rowid
                WHERE {self.fts_index} MATCH ?
                ORDER BY bm25({self.fts_index})
                LIMIT ?
            """,
                (fts_query, top_k),
            )

            results = [
                {
                    "rowid": row[0],
                    "article_id": row[1],
                    "chunk_seq": row[2],
                    "chunk_text": row[3],
                    "fts_rank": rank,
                }
                for rank, row in enumerate(cursor.fetchall(), 1)
            ]

            return results
        finally:
            conn.close()

    def calculate_rrf_score(
        self, hyde_rank: Optional[int], fts_rank: Optional[int], weights: QueryWeights
    ) -> float:
        """Calculate weighted RRF score, normalized by total weight.

        Normalization ensures scores are comparable across different query types
        with different weight distributions.
        """
        score = 0.0
        if hyde_rank is not None:
            score += weights.hyde_weight * (1.0 / (self.rrf_k + hyde_rank))
        if fts_rank is not None:
            score += weights.fts_weight * (1.0 / (self.rrf_k + fts_rank))

        # Normalize by total weight to make scores comparable across query types
        total_weight = weights.hyde_weight + weights.fts_weight
        if total_weight > 0:
            score = score / total_weight

        return score

    def fuse_results(
        self, hyde_results: list[dict], fts_results: list[dict], weights: QueryWeights
    ) -> list[dict]:
        """Fuse HyDE and FTS results using weighted RRF."""
        combined = {}

        for r in hyde_results:
            rid = r["rowid"]
            combined[rid] = {
                "rowid": rid,
                "article_id": r["article_id"],
                "chunk_seq": r["chunk_seq"],
                "chunk_text": r["chunk_text"],
                "hyde_rank": r["hyde_rank"],
                "fts_rank": None,
                "source": "HYDE",
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
                    "hyde_rank": None,
                    "fts_rank": r["fts_rank"],
                    "source": "FTS5",
                }

        for data in combined.values():
            data["rrf_score"] = self.calculate_rrf_score(
                data["hyde_rank"], data["fts_rank"], weights
            )

        return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

    def get_article_headwords(self, article_ids: list[int]) -> dict[int, str]:
        """Fetch article headwords for display."""
        if not article_ids:
            return {}
        conn = create_connection()
        try:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(article_ids))
            cursor.execute(
                f"SELECT id, headword FROM articles WHERE id IN ({placeholders})",
                article_ids,
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        top_k_hyde: int = TOP_K_HYDE,
        top_k_fts: int = TOP_K_FTS,
        override_query_type: Optional[QueryType] = None,
    ) -> list[HybridHyDESearchResults]:
        """Hybrid search with adaptive weighting based on query type.

        Gracefully degrades to single-method search if one method fails.

        Raises:
            ValueError: If query is empty or both search methods fail
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        hyde_results = []
        fts_results = []
        hyde_error = None
        fts_error = None
        query_type = None

        # Execute both searches in parallel with error handling
        with ThreadPoolExecutor(max_workers=2) as executor:
            hyde_future = executor.submit(self.search_hyde, query, top_k_hyde)
            fts_future = executor.submit(self.search_fts5, query, top_k_fts)

            for future in as_completed([hyde_future, fts_future]):  # type: ignore[type-arg,var-annotated,arg-type]
                try:
                    if future == hyde_future:
                        hyde_results, _, query_type = future.result()
                    else:
                        fts_results = future.result()
                except Exception as e:
                    # Graceful degradation
                    if future == hyde_future:
                        hyde_error = e
                    else:
                        fts_error = e

        # Determine query type and weights
        if override_query_type:
            query_type = override_query_type
        elif query_type is None:
            # If HyDE failed and we don't have classification, default to CONCEPTUAL
            query_type = QueryType.CONCEPTUAL

        weights = QUERY_WEIGHTS[query_type]

        # Check if both methods failed
        if not hyde_results and not fts_results:
            error_msg = "Both HyDE and FTS search failed. "
            if hyde_error:
                error_msg += f"HyDE error: {str(hyde_error)}. "
            if fts_error:
                error_msg += f"FTS error: {str(fts_error)}."
            raise ValueError(error_msg)

        # Fuse results (works even if one is empty)
        fused = self.fuse_results(hyde_results, fts_results, weights)

        top_results = fused[:top_k]
        article_ids = list(set(int(r["article_id"]) for r in top_results))
        headwords = self.get_article_headwords(article_ids)

        return [
            HybridHyDESearchResults(
                rank=rank,
                article_id=int(r["article_id"]),
                article_headword=headwords.get(int(r["article_id"]), "Unknown"),
                chunk_sequence=r["chunk_seq"],
                chunk_text=r["chunk_text"],
                rrf_score=round(r["rrf_score"], 6),
                hyde_rank=r["hyde_rank"],
                fts_rank=r["fts_rank"],
                source=r["source"],
            )
            for rank, r in enumerate(top_results, 1)
        ]
