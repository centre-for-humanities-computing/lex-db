# Vector Store Implementation Plan for lex-db

This document outlines the plan for implementing a vector store using SQLite and `sqlite-vec` for the lex-db project.

## Phase 1: Core Vector Store Functionality

### 1. Ability to Create a New Vector Index

**Goal:** Create a new vector index for a given SQLite table, using either local or remote embedding models. Initial implementation will use a naive document splitter. Index creation will be a script-based or manual process.

**Components & Pseudocode:**

*   **Embedding Generation Module (e.g., `lex_db/embeddings.py`)**
    *   Purpose: Abstract the process of generating embeddings from text using different models.
    *   Pseudocode:
        ```python
        # lex_db/embeddings.py
        # (Conceptual - actual library usage will differ)

        # Supported models (enum or constants)
        MODEL_LOCAL_SENTENCE_TRANSFORMER = "local_st"
        MODEL_OPENAI_ADA_002 = "openai_ada_002"

        function get_embedding_dimensions(model_choice: str) -> int:
            if model_choice == MODEL_LOCAL_SENTENCE_TRANSFORMER:
                return 384 # Example dimension for a common ST model
            else if model_choice == MODEL_OPENAI_ADA_002:
                return 1536 # Dimension for text-embedding-ada-002
            else:
                raise ValueError(f"Unknown model: {model_choice}")

        function generate_embeddings(texts: list[str], model_choice: str) -> list[list[float]]:
            embeddings_list = []
            if model_choice == MODEL_LOCAL_SENTENCE_TRANSFORMER:
                # model = load_sentence_transformer_model_from_registry("all-MiniLM-L6-v2")
                # embeddings = model.encode(texts)
                # return embeddings.tolist() # Convert numpy arrays to lists of floats
                print(f"Placeholder: Generating embeddings for {len(texts)} texts using local Sentence Transformer")
                # Replace with actual model loading and encoding
                dummy_dimension = get_embedding_dimensions(model_choice)
                for _ in texts:
                    embeddings_list.append([0.0] * dummy_dimension) # Dummy embedding
                return embeddings_list

            else if model_choice == MODEL_OPENAI_ADA_002:
                # api_key = load_openai_api_key_from_config()
                # client = OpenAI(api_key=api_key)
                # response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
                # embeddings = [item.embedding for item in response.data]
                # return embeddings
                print(f"Placeholder: Generating embeddings for {len(texts)} texts using OpenAI API")
                # Replace with actual API call
                dummy_dimension = get_embedding_dimensions(model_choice)
                for _ in texts:
                    embeddings_list.append([0.0] * dummy_dimension) # Dummy embedding
                return embeddings_list
            else:
                raise ValueError(f"Unsupported embedding model: {model_choice}")
        ```

*   **Document Splitting Utility (e.g., in `lex_db/utils.py` or within `embeddings.py`)**
    *   Purpose: Split large documents into smaller, manageable chunks for embedding.
    *   Pseudocode:
        ```python
        # lex_db/utils.py (or similar)
        function split_document_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
            chunks = []
            start_index = 0
            text_len = len(text)
            while start_index < text_len:
                end_index = min(start_index + chunk_size, text_len)
                chunks.append(text[start_index:end_index])
                if end_index == text_len:
                    break
                start_index += (chunk_size - overlap)
                if start_index >= text_len: # Ensure we don\'t create an empty chunk if overlap is large
                    break
            return chunks
        ```

*   **Index Creation and Population Function (in `lex_db/database.py`)**
    *   Purpose: Execute SQL to create the `sqlite-vec` virtual table and populate it with embeddings. This function would be called from a script.
    *   Pseudocode:
        ```python
        # lex_db/database.py
        # from .embeddings import generate_embeddings, get_embedding_dimensions
        # from .utils import split_document_into_chunks

        function create_vector_index(db_conn, source_table_name: str, source_text_column: str,
                                     vector_index_name: str, embedding_model_choice: str,
                                     chunk_size: int = 512, chunk_overlap: int = 50):
            cursor = db_conn.cursor()
            embedding_dim = get_embedding_dimensions(embedding_model_choice)

            # 1. Create the sqlite-vec virtual table
            #    The embedding column name here is 'embedding', dimension is embedding_dim
            #    Factory can be omitted for default (IVF geralmente) or specified, e.g., 'IVF4096,Flat'
            #    See sqlite-vec docs for CREATE VIRTUAL TABLE syntax details.
            create_table_sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {vector_index_name} USING vec0(
                embedding {embedding_dim}D, -- Defines the vector column and its dimensions
                source_article_id TEXT,    -- To link back to the original article
                chunk_sequence_id INTEGER  -- To identify the chunk order within an article
            );
            """
            cursor.execute(create_table_sql)
            print(f"Virtual table {vector_index_name} created or already exists.")

            # 2. Fetch articles from the source table
            cursor.execute(f"SELECT rowid, {source_text_column} FROM {source_table_name}")
            articles = cursor.fetchall()

            # 3. Process each article: chunk, embed, and insert
            for article_rowid, text_content in articles:
                if not text_content: continue # Skip if no text

                chunks = split_document_into_chunks(text_content, chunk_size=chunk_size, overlap=chunk_overlap)
                if not chunks: continue

                chunk_embeddings = generate_embeddings(chunks, model_choice=embedding_model_choice)

                for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    # The rowid for vec0 is typically managed by sqlite-vec itself upon insert.
                    # We store source_article_id and chunk_sequence_id as separate columns.
                    # The vector itself is passed as a Python list of floats, sqlite-vec handles conversion.
                    # Note: sqlite-vec expects the vector as a blob.
                    # The Python sqlite3 driver can convert a bytes object to a blob.
                    # How you get this bytes object depends on the embedding library and sqlite-vec's Python API.
                    # For now, assuming generate_embeddings returns list[float] and sqlite-vec handles it via its Python bindings
                    # or you'd convert it to JSON string / raw bytes as per sqlite-vec requirements.
                    # For sqlite-vec, you typically pass the list of floats directly if using its Python API,
                    # or a JSON array string if using raw SQL with json_encode.
                    # Let's assume direct list[float] for now if sqlite-vec Python wrapper is used.
                    # If not, it would be something like: json.dumps(chunk_embedding)
                    insert_sql = f"""
                    INSERT INTO {vector_index_name} (embedding, source_article_id, chunk_sequence_id)
                    VALUES (?, ?, ?);
                    """
                    cursor.execute(insert_sql, (chunk_embedding, str(article_rowid), i))

            db_conn.commit()
            print(f"Finished populating {vector_index_name} from {source_table_name}.")
        ```

### 2. Ability to Move a Vector Index

**Goal:** Allow the SQLite database file (containing the vector index) to be moved to a different location.

**Components & Pseudocode:**

*   **File System Operation (Manual or Scripted)**
    *   This is primarily a manual or external script process. The application needs to be reconfigured to point to the new database location.
    *   Pseudocode (Conceptual):
        ```bash
        # 1. Stop the application using the database
        #    (e.g., make stop_server)

        # 2. Move the SQLite database file
        #    mv /current/path/to/lex.db /new/shared_drive/lex.db

        # 3. Update DATABASE_URL in .env file
        #    DATABASE_URL=/new/shared_drive/lex.db

        # 4. Restart the application
        #    (e.g., make run)
        ```
*   **Configuration Update:**
    *   Ensure `lex_db/config.py` correctly reads the `DATABASE_URL` from the environment. No code change needed if already implemented this way.

### 3. Ability to Maintain the Vector Index

**Goal:** Add new items to and remove items from the vector index. This will be handled by a batch update script (e.g., run via a CRON job).

**Components & Pseudocode:**

*   **Adding Items to Index (in `lex_db/database.py`)**
    *   Purpose: Add embeddings of new articles/chunks to an existing index.
    *   Pseudocode:
        ```python
        # lex_db/database.py
        function add_single_article_to_vector_index(db_conn, vector_index_name: str,
                                                   article_rowid: any, article_text: str,
                                                   embedding_model_choice: str,
                                                   chunk_size: int = 512, chunk_overlap: int = 50):
            if not article_text: return

            cursor = db_conn.cursor()
            chunks = split_document_into_chunks(article_text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks: return

            chunk_embeddings = generate_embeddings(chunks, model_choice=embedding_model_choice)

            for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                insert_sql = f"""
                INSERT INTO {vector_index_name} (embedding, source_article_id, chunk_sequence_id)
                VALUES (?, ?, ?);
                """
                cursor.execute(insert_sql, (chunk_embedding, str(article_rowid), i))
            db_conn.commit()
            print(f"Added article {article_rowid} to index {vector_index_name}.")
        ```

*   **Removing Items from Index (in `lex_db/database.py`)**
    *   Purpose: Remove embeddings related to a specific article ID from the index.
    *   Pseudocode:
        ```python
        # lex_db/database.py
        function remove_article_from_vector_index(db_conn, vector_index_name: str, article_rowid: any):
            cursor = db_conn.cursor()
            # Delete all chunks associated with the given source_article_id
            delete_sql = f"DELETE FROM {vector_index_name} WHERE source_article_id = ?;"
            cursor.execute(delete_sql, (str(article_rowid),))
            db_conn.commit()
            print(f"Removed article {article_rowid} from index {vector_index_name}.")
        ```

*   **Batch Update Script (e.g., `scripts/update_vector_indexes.py`)**
    *   Purpose: A script that runs periodically (e.g., via CRON), checks for new/modified/deleted articles in the source table, and updates the vector index using the functions above.
    *   Pseudocode (Conceptual for a batch script):
        ```python
        # scripts/update_vector_indexes.py (Example)
        # import time
        # from lex_db.database import (get_db_connection, add_single_article_to_vector_index,
        #                            remove_article_from_vector_index, get_embedding_dimensions)
        # from lex_db.config import settings # Assuming settings has DB_PATH, etc.

        function get_last_processed_timestamp(index_name: str) -> float:
            # Read from a state file or a dedicated table
            return 0.0 # Placeholder

        function set_last_processed_timestamp(index_name: str, timestamp: float):
            # Write to a state file or a dedicated table
            pass

        function main_update_cycle(vector_index_name: str, source_table: str, text_column: str, model_choice: str):
            db = get_db_connection(settings.DATABASE_URL) # Or however you get it
            last_ts = get_last_processed_timestamp(vector_index_name)
            current_ts = time.time()

            # Fetch new/updated articles (assuming an \'updated_at\' column in source_table)
            # cursor = db.cursor()
            # cursor.execute(f\"SELECT rowid, {text_column} FROM {source_table} WHERE updated_at > ?\", (last_ts,))
            # new_articles = cursor.fetchall()
            # for article_id, text in new_articles:
            #     # Might need to remove old chunks first if it\'s an update
            #     remove_article_from_vector_index(db, vector_index_name, article_id)
            #     add_single_article_to_vector_index(db, vector_index_name, article_id, text, model_choice)

            # Fetch deleted articles (this is harder without soft deletes or a log table)
            # One approach: get all current IDs in source, compare with IDs in vector index, delete orphans.
            # cursor.execute(f\"SELECT DISTINCT source_article_id FROM {vector_index_name}\")
            # indexed_ids = {row[0] for row in cursor.fetchall()}
            # cursor.execute(f\"SELECT rowid FROM {source_table}\")
            # source_ids = {str(row[0]) for row in cursor.fetchall()}
            # ids_to_delete = indexed_ids - source_ids
            # for article_id_to_delete in ids_to_delete:
            #     remove_article_from_vector_index(db, vector_index_name, article_id_to_delete)

            set_last_processed_timestamp(vector_index_name, current_ts)
            db.close()
        ```

### 4. API Endpoint for Similarity Search

**Goal:** Provide an API endpoint that takes a query text and returns top-k (or top-p) similar items from a chosen vector index.

**Components & Pseudocode:**

*   **Similarity Search Function (in `lex_db/database.py`)**
    *   Purpose: Execute the vector search query against `sqlite-vec`.
    *   Pseudocode:
        ```python
        # lex_db/database.py
        # from .embeddings import generate_embeddings

        function search_vector_index(db_conn, vector_index_name: str, query_text: str,
                                     embedding_model_choice: str, top_k: int = 5) -> list[dict]:
            cursor = db_conn.cursor()

            # 1. Generate embedding for the query text
            #    Ensure the same model and preprocessing as used for indexing
            query_vector = generate_embeddings([query_text], model_choice=embedding_model_choice)[0]

            # 2. Perform the search using sqlite-vec\'s MATCH operator.
            #    The query vector needs to be passed in a format sqlite-vec understands (e.g., blob via JSON).
            #    If `query_vector` is `list[float]`, `json.dumps(query_vector)` is common.
            import json # Make sure to import json
            query_vector_json = json.dumps(query_vector)

            search_sql = f\"\"\"
            SELECT rowid, source_article_id, chunk_sequence_id, distance
            FROM {vector_index_name}
            WHERE embedding MATCH ? 
            ORDER BY distance
            LIMIT ?;
            \"\"\"
            # The \'?\' for MATCH expects the vector, often as a JSON string.
            # Consult sqlite-vec documentation for the precise format if issues arise.
            cursor.execute(search_sql, (query_vector_json, top_k))
            results = cursor.fetchall()

            # Format results
            # (rowid, source_article_id, chunk_sequence_id, distance)
            formatted_results = [
                {\"id_in_index\": res[0], \"source_article_id\": res[1], \"chunk_seq\": res[2], \"distance\": res[3]}
                for res in results
            ]
            return formatted_results
        ```

*   **API Endpoint (in `lex_db/api/routes.py`)**
    *   Purpose: Expose similarity search via API.
    *   Pseudocode:
        ```python
        # lex_db/api/routes.py
        # class SearchRequest(BaseModel):
        #     vector_index_name: str
        #     query_text: str
        #     embedding_model_choice: str # Important: Must match the index\'s model
        #     top_k: int = 5

        # @router.post(\"/vector_indexes/search\")
        # async def api_search_vector_index(request: SearchRequest, db: sqlite3.Connection = Depends(get_db_connection)):\n            # ... (rest of the function as previously defined) ...
        ```

## Phase 2: Enhancements and Refinements

*   **Configuration for Indexes:** Store metadata about created vector indexes (e.g., source table, embedding model used, chunking strategy) perhaps in a separate SQLite table or a JSON config file. This helps in managing multiple indexes and selecting the correct embedding model for queries.
*   **Advanced Chunking:** Explore more sophisticated chunking strategies (e.g., semantic chunking, sentence splitting).
*   **Top-P Sampling:** Implement logic for top-p (nucleus) sampling in the search results if required. This would typically involve fetching more than `top_k` results, sorting by similarity, and then applying the cumulative probability threshold.
*   **Error Handling and Logging:** Robust error handling and comprehensive logging throughout the new modules.
*   **Testing:** Unit and integration tests for all new functionalities.
*   **Asynchronous Operations:** For long-running operations like full index creation, consider using background tasks (e.g., with FastAPI's `BackgroundTasks` or a task queue like Celery).

## Key `sqlite-vec` Documentation Points to Reference:

*   **Creating Virtual Tables:** Syntax for `CREATE VIRTUAL TABLE ... USING vec0(...)`. Pay attention to specifying the embedding column name, dimensions, and any factory parameters for the underlying index (e.g., IVF, HNSW).
*   **Inserting Data:** How to `INSERT` vectors. Vectors are typically stored as BLOBs. Python lists of floats might be automatically converted by `sqlite-vec`'s Python bindings, or you might need to serialize them (e.g., to a JSON string or raw bytes).
*   **Querying/Searching:** Syntax for similarity search (e.g., `MATCH` operator or a specific search function like `vec_search` or `vss_search`). How to pass the query vector and use `ORDER BY distance` and `LIMIT`.
*   **Python API:** Any specific Python utility functions provided by `sqlite-vec` for easier interaction (e.g., for vector serialization/deserialization if needed).
*   **`sqlite-rembed` / `sqlite-lembed`:** If considering using these for in-database embedding generation, their specific functions and how they integrate with `sqlite-vec` tables will be key. The current plan assumes embeddings are generated outside SQLite and then inserted.

This plan provides a starting point. Specific SQL syntax and Python API usage for `sqlite-vec` will need to be confirmed against its official documentation during implementation.
