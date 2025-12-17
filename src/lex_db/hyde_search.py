"""
HyDE Search (Hypothetical Document Embedding)

Uses LLM to generate a hypothetical encyclopedia passage, embeds it,
and searches for similar chunks in the vector database.

USAGE:
    export OPENROUTER_API_KEY="your-key"
    python hyde_search.py
"""

import os
import requests # type: ignore[import-untyped]

from lex_db import search_utils

# ===============================================================
# CONFIGURATION
# ===============================================================

VECTOR_INDEX = "article_embeddings_e5"
TOP_K_CANDIDATES = 25

OPENROUTER_MODEL = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HYDE_PROMPT = """Du er en dansk leksikonskribent. Skriv et kort afsnit (3-4 sætninger) der besvarer dette spørgsmål:

{query}

Regler:
- Skriv i plain text uden formatering (ingen **, ##, eller punktopstillinger)
- Nævn relevante navne, begreber og årstal
- Skriv i encyklopædisk stil, ikke som en AI-assistent
- Hvis du er usikker på detaljer, hold dig til generelle fakta"""

# ===============================================================
# LLM PASSAGE GENERATION
# ===============================================================


def generate_hypothetical_passage(query: str) -> str:
    """Generate hypothetical passage using LLM via OpenRouter.

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
            content: str = response_data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("Empty response from LLM")
            return content.strip()
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
# HYDE SEARCH CLASS
# ===============================================================


class HyDESearch:
    """HyDE search: LLM generates hypothetical passage → embed → search."""

    def __init__(self, vector_index: str = VECTOR_INDEX):
        self.vector_index = vector_index
        self.conn = None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search using HyDE approach.
        Args:
            query: Search query text
            top_k: Number of results to return
        Returns:
            List of search results with headword, similarity score, and chunk text
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.conn is None:
            raise ValueError(
                "Database connection not initialized. Assign connection to searcher.conn before calling search methods."
            )

        query = query.strip()
        # Step 1: Generate hypothetical passage
        hypothetical = generate_hypothetical_passage(query)

        # Step 2: Embed passage
        embedding = search_utils.embed_passage(hypothetical)

        # Step 3: Search vector index using embedding
        results = search_utils.search_by_embedding(
            conn=self.conn,
            vector_index=self.vector_index,
            embedding=embedding,
            top_k=TOP_K_CANDIDATES,
            rank_key="rank",
            include_distance=True
        )

        # Convert distance to similarity score
        for r in results:
            r["similarity"] = 1 - (r["distance"] ** 2 / 2)  # Convert L2 distance to cosine similarity
            del r["distance"]  # Remove distance, keep only similarity

        # Get top results
        top_results = results[:top_k]

        # Fetch headwords and format output
        article_ids = list({int(r["article_id"]) for r in top_results})
        headwords = search_utils.get_article_headwords(self.conn, article_ids)

        return [
            {
                "rank": i + 1,
                "article_id": int(r["article_id"]),
                "headword": headwords.get(int(r["article_id"]), "Unknown"),
                "chunk_seq": r["chunk_seq"],
                "similarity": round(r["similarity"], 4),
                "text": r["chunk_text"],
            }
            for i, r in enumerate(top_results)
        ]
