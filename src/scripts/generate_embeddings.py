"""CLI tool to generate embeddings from a JSONL articles dump.

This script reads articles from a JSONL file (produced by dump_articles.py),
chunks the text, generates embeddings, and outputs a JSONL file that can be
imported using import_vector_embeddings.py.

This script is designed to run on a remote server without database access.
"""

import argparse
import json
from datetime import datetime, timezone

from lex_db.utils import (
    get_logger,
    configure_logging,
    ChunkingStrategy,
    split_document_into_chunks,
)
from lex_db.embeddings import (
    EmbeddingModel,
    TextType,
    generate_embeddings,
    get_embedding_dimensions,
)

logger = get_logger()


def generate_embeddings_from_jsonl(
    input_file: str,
    output_file: str,
    embedding_model_choice: EmbeddingModel,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SECTIONS,
    batch_size: int = 2048,
) -> dict[str, int]:
    """
    Generate embeddings from a JSONL articles dump.

    Args:
        input_file: Path to input JSONL file with articles
        output_file: Path to output JSONL file with embeddings
        embedding_model_choice: Model to use for generating embeddings
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between consecutive chunks
        chunking_strategy: Strategy for splitting text into chunks
        batch_size: Number of chunks to process per batch

    Returns:
        Dictionary with stats: {"articles_processed": N, "chunks_created": M, "errors": K}
    """
    stats = {"articles_processed": 0, "chunks_created": 0, "errors": 0}

    # Read metadata and articles from input file
    logger.info(f"Reading articles from {input_file}")
    articles = []
    metadata = None

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # First line should be metadata
                if line_num == 1 and "_metadata" in data:
                    metadata = data["_metadata"]
                    logger.info(f"Input metadata: {metadata}")
                    continue

                # Regular article line
                articles.append(data)

                # Log progress every 10000 articles
                if len(articles) % 10000 == 0:
                    logger.info(f"Loaded {len(articles)} articles...")

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                stats["errors"] += 1

    logger.info(f"Loaded {len(articles)} articles from {input_file}")

    # Prepare chunks for all articles
    logger.info("Chunking articles...")
    all_chunks = []  # List of (article_id, chunk_idx, chunk_text)

    for article in articles:
        try:
            article_id = article["id"]
            headword = article.get("headword", "")
            text = article.get("xhtml_md", "")

            if not text:
                logger.warning(f"Article {article_id} has no text content, skipping")
                stats["errors"] += 1
                continue

            # Prepend headword as title (same as update_vector_index does)
            article_text = f"# {headword}\n{text}"

            # Split into chunks
            chunks = split_document_into_chunks(
                article_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
            )

            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append((article_id, chunk_idx, chunk_text))

            stats["articles_processed"] += 1

            # Log progress every 1000 articles
            if stats["articles_processed"] % 1000 == 0:
                logger.info(
                    f"Chunked {stats['articles_processed']} articles, "
                    f"{len(all_chunks)} total chunks"
                )

        except Exception as e:
            logger.error(f"Error chunking article {article.get('id', 'unknown')}: {e}")
            stats["errors"] += 1

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks from "
        f"{stats['articles_processed']} articles"
    )

    # Generate embeddings in batches and write to output
    logger.info(f"Generating embeddings using {embedding_model_choice.value}...")
    embedding_dim = get_embedding_dimensions(embedding_model_choice)
    logger.info(f"Embedding dimension: {embedding_dim}")

    with open(output_file, "w", encoding="utf-8") as f:
        # Write metadata header
        output_metadata = {
            "_metadata": {
                "index_name": None,  # Will be set during import
                "total_chunks": len(all_chunks),
                "export_date": datetime.now(timezone.utc).isoformat(),
                "embedding_model": embedding_model_choice.value,
                "embedding_dimensions": embedding_dim,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_strategy": chunking_strategy.value,
                "source_dump": metadata,
            }
        }
        f.write(json.dumps(output_metadata, ensure_ascii=False) + "\n")

        # Process chunks in batches
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            try:
                # Generate embeddings for this batch
                texts_to_embed = [
                    (chunk_text, TextType.PASSAGE) for _, _, chunk_text in batch
                ]
                embeddings = generate_embeddings(texts_to_embed, embedding_model_choice)

                # Write each chunk with its embedding
                for (article_id, chunk_idx, chunk_text), embedding in zip(
                    batch, embeddings
                ):
                    chunk_data = {
                        "article_id": article_id,
                        "chunk_idx": chunk_idx,
                        "chunk_text": chunk_text,
                        "embedding": embedding,
                    }
                    f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                    stats["chunks_created"] += 1

                # Log progress
                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.info(
                        f"Processed batch {batch_num}/{total_batches}: "
                        f"{stats['chunks_created']}/{len(all_chunks)} chunks"
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                stats["errors"] += len(batch)

    logger.info(
        f"Embedding generation complete: {stats['chunks_created']} chunks written to {output_file}"
    )
    return stats


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from JSONL articles dump for remote processing"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input JSONL file with articles"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output JSONL file for embeddings"
    )
    parser.add_argument(
        "--embedding-model",
        "-e",
        choices=[m.value for m in EmbeddingModel],
        default=EmbeddingModel.LOCAL_MULTILINGUAL_E5_SMALL.value,
        help="Embedding model to use (default: intfloat/multilingual-e5-small)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks for embedding (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks (default: 50)",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=ChunkingStrategy.SECTIONS.value,
        choices=[s.value for s in ChunkingStrategy],
        help="Strategy for splitting text into chunks (default: sections)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of chunks to process per batch (default: 2048)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    try:
        stats = generate_embeddings_from_jsonl(
            input_file=args.input,
            output_file=args.output,
            embedding_model_choice=EmbeddingModel(args.embedding_model),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunking_strategy=ChunkingStrategy(args.chunking_strategy),
            batch_size=args.batch_size,
        )

        logger.info(
            f"Successfully processed {stats['articles_processed']} articles, "
            f"created {stats['chunks_created']} chunks"
        )
        if stats["errors"] > 0:
            logger.warning(f"Encountered {stats['errors']} errors during processing")
            exit(1)

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
