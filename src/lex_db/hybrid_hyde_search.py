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
import requests # type: ignore[import-untyped]

from lex_db.database import create_connection
from lex_db import search_utils

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
        self.stopwords = stopwords if stopwords is not None else search_utils.DEFAULT_STOPWORDS

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

        # Helper function for HyDE search
        def search_hyde_internal(q: str, tk: int) -> tuple[list[dict], str, QueryType]:
            """Perform HyDE search and classify query."""
            hypothetical, qtype = generate_hypothetical_passage(q)
            embedding = search_utils.embed_passage(hypothetical)

            conn = create_connection()
            try:
                results = search_utils.search_by_embedding(
                    conn=conn,
                    vector_index=self.vector_index,
                    embedding=embedding,
                    top_k=tk,
                    rank_key="hyde_rank"
                )
                return results, hypothetical, qtype
            finally:
                conn.close()

        # Helper function for FTS5 search
        def search_fts5_internal(q: str, tk: int) -> list[dict]:
            """Get ranked candidates from FTS5 keyword search."""
            conn = create_connection()
            try:
                return search_utils.search_fts5(
                    conn=conn,
                    vector_index=self.vector_index,
                    fts_index=self.fts_index,
                    query=q,
                    top_k=tk,
                    stopwords=self.stopwords,
                )
            finally:
                conn.close()

        hyde_results = []
        fts_results = []
        hyde_error = None
        fts_error = None
        query_type = None

        # Execute both searches in parallel with error handling
        with ThreadPoolExecutor(max_workers=2) as executor:
            hyde_future = executor.submit(search_hyde_internal, query, top_k_hyde)
            fts_future = executor.submit(search_fts5_internal, query, top_k_fts)

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

        # Fuse results using weighted RRF
        fused = search_utils.fuse_results_rrf(
            results1=hyde_results,
            results2=fts_results,
            rank1_key="hyde_rank",
            rank2_key="fts_rank",
            source1_label="HYDE",
            source2_label="FTS5",
            rrf_k=self.rrf_k,
            weight1=weights.hyde_weight,
            weight2=weights.fts_weight,
            normalize=True  # Normalize for adaptive weighting
        )

        top_results = fused[:top_k]
        article_ids = list(set(int(r["article_id"]) for r in top_results))

        # Fetch article headwords
        conn = create_connection()
        try:
            headwords = search_utils.get_article_headwords(conn, article_ids)
        finally:
            conn.close()

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
