from enum import Enum
import os

import requests
from pydantic import BaseModel

from lex_db.search_utils import (
    DEFAULT_STOPWORDS,
    RetrievalResult,
    fuse_results_rrf,
    search_fts5,
)
import lex_db.database as db
import lex_db.vector_store as vs
from lex_db.embeddings import EmbeddingModel, TextType


TOP_K_SEMANTIC = 50
TOP_K_FTS = 50
TOP_K_FINAL = 10
RRF_K = 60

OPENROUTER_MODEL = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HYDE_PROMPT = """Du er en dansk leksikonskribent.

Analyser først spørgsmålet og klassificer det som ÉN af:
- ENTITY: Spørgsmål om en specifik person, sted eller ting (f.eks. "Christian IV", "Hvem var Kierkegaard?")
- CONCEPTUAL: Spørgsmål om ideer, processer, forklaringer eller generelle emner (f.eks. "Hvad er eksistentialisme?", "Hvordan opstod renæssancen?")
- UNCLEAR: Spørgsmål der ikke klart falder i én af de to ovenstående kategorier.

Skriv derefter et kort encyklopædisk afsnit (3-4 sætninger) der besvarer spørgsmålet.

Spørgsmål: {query}

Svar i dette format:
QUERYTYPE: [ENTITY, CONCEPTUAL, eller MIXED]
PASSAGE: [dit encyklopædiske afsnit her - plain text uden formatering]

Regler for afsnittet:
- Skriv i plain text uden formatering (ingen **, ##, eller punktopstillinger)
- Nævn relevante navne, begreber og årstal
- Skriv i encyklopædisk stil, ikke som en AI-assistent
- Hvis du er usikker på detaljer, hold dig til generelle fakta"""


class QueryType(Enum):
    """Query types for adaptive weighting (LLM-based classification)."""

    ENTITY = "entity"
    CONCEPTUAL = "conceptual"
    UNCLEAR = "unclear"


class SearchMethod(str, Enum):
    SEMANTIC = "SEMANTIC"
    FULLTEXT = "FULLTEXT"
    HYDE = "HYDE"


QUERY_WEIGHTS = {
    QueryType.ENTITY: {
        SearchMethod.SEMANTIC: 0.8,
        SearchMethod.FULLTEXT: 1.2,
        SearchMethod.HYDE: 0.8,
    },
    QueryType.CONCEPTUAL: {
        SearchMethod.SEMANTIC: 1.2,
        SearchMethod.FULLTEXT: 0.8,
        SearchMethod.HYDE: 1.2,
    },
    QueryType.UNCLEAR: {
        SearchMethod.SEMANTIC: 1.0,
        SearchMethod.FULLTEXT: 1.0,
        SearchMethod.HYDE: 1.0,
    },
}


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
                elif "CONCEPTUAL" in querytype_line.upper():
                    query_type = QueryType.CONCEPTUAL
                else:
                    query_type = QueryType.UNCLEAR  # Default

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


class HybridSearchResults(BaseModel):
    """Results of a hybrid search."""

    results: list[RetrievalResult]


def hybrid_search(
    semantic_queries: list[tuple[str, TextType]],
    keyword_queries: list[str],
    query_type: QueryType = QueryType.UNCLEAR,
    vector_index: str = "article_embeddings_e5",
    fts_index: str = "fts_article_embeddings_e5",
    embedding_model: EmbeddingModel = EmbeddingModel.LOCAL_MULTILINGUAL_E5_LARGE,
    top_k: int = 10,
    top_k_semantic: int = 50,
    top_k_fts: int = 50,
    rrf_k: int = 60,
    stopwords: set[str] = DEFAULT_STOPWORDS,
) -> HybridSearchResults:
    """Hybrid search with RRF fusion."""

    if not semantic_queries and not keyword_queries:
        raise ValueError(
            "At least one of semantic_queries or keyword_queries must be provided"
        )

    with db.get_db_connection() as conn:
        # Stage 1a: Semantic search
        semantic_results = vs.search_vector_index(
            db_conn=conn,
            vector_index_name=vector_index,
            queries=semantic_queries,
            embedding_model=embedding_model,
            top_k=top_k_semantic,
        )

        # Stage 1b: FTS5 keyword search (only if keyword queries provided)
        fts_results = []
        if keyword_queries:
            fts_results = search_fts5(
                conn=conn,
                vector_index=vector_index,
                fts_index=fts_index,
                queries=keyword_queries,
                top_k=top_k_fts,
                stopwords=stopwords,
            )

        combined_results: list[list[RetrievalResult]] = [
            [
                RetrievalResult(
                    rowid=r.id_in_index,
                    article_id=int(r.source_article_id),
                    chunk_sequence=r.chunk_seq,
                    chunk_text=r.chunk_text,
                    score=r.distance,
                )
                for r in res.results
            ]
            for res in semantic_results
        ]

        # Only append FTS results if they exist
        if fts_results:
            combined_results.append(fts_results)

        # Stage 2: Fusion (only needed if we have multiple result sets)
        if len(combined_results) == 1:
            # Single search method - no fusion needed, just take top-k
            top_results = combined_results[0][:top_k]
        else:
            # Multiple search methods - use RRF fusion
            query_weights = QUERY_WEIGHTS[query_type]
            weights = [query_weights[SearchMethod.SEMANTIC]] * len(semantic_queries)
            if keyword_queries:
                weights.extend([query_weights[SearchMethod.FULLTEXT]] * len(keyword_queries))

            fused = fuse_results_rrf(
                results=combined_results, rrf_k=rrf_k, weights=weights, normalize=False
            )
            top_results = fused[:top_k]

        return HybridSearchResults(results=top_results)