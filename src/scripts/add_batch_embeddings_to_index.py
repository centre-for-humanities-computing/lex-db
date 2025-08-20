"""Process OpenAI batch embedding results and add them to a vector index."""

import argparse
import json
import sys
from pathlib import Path

from lex_db.database import create_connection
from lex_db.config import get_settings
from lex_db.utils import get_logger, configure_logging
from lex_db.vector_store import (
    validate_tables_exist,
    add_precomputed_embeddings_to_vector_index,
)

logger = get_logger()


def load_input_texts(input_file: Path) -> dict[str, str]:
    """Load custom_id -> input text mapping from batch input file."""
    input_texts = {}

    try:
        with open(input_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                request = json.loads(line)
                custom_id = request.get("custom_id")
                input_text = request.get("body", {}).get("input", "")

                if custom_id and input_text:
                    input_texts[custom_id] = input_text

    except Exception as e:
        logger.error(f"Error loading input texts from {input_file}: {e}")

    return input_texts


def load_batch_results(results_dir: Path) -> list[tuple[str, str, str, list[float]]]:
    """Load embedding results with reconstructed chunk text from input files."""
    embeddings: list[tuple[str, str, str, list[float]]] = []
    result_files = sorted(results_dir.glob("batch_*_results.jsonl"))

    if not result_files:
        logger.warning(f"No batch result files found in {results_dir}")
        return embeddings

    logger.info(f"Processing {len(result_files)} batch result files")

    for result_file in result_files:
        # Find corresponding input file
        batch_name = result_file.stem.replace("_results", "")
        input_file = results_dir / f"{batch_name}.jsonl"

        if not input_file.exists():
            logger.warning(f"Input file not found for {result_file}: {input_file}")
            continue

        # Load input texts for this batch
        input_texts = load_input_texts(input_file)
        logger.info(f"Loaded {len(input_texts)} input texts from {input_file}")

        # Process results
        try:
            with open(result_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        result = json.loads(line)
                        custom_id = result.get("custom_id")

                        if not custom_id:
                            logger.warning(
                                f"Missing custom_id in {result_file}:{line_num}"
                            )
                            continue

                        if result.get("response", {}).get("status_code") != 200:
                            logger.warning(f"Failed request {custom_id}")
                            continue

                        response_body = result.get("response", {}).get("body", {})
                        data = response_body.get("data", [])

                        if not data or not data[0].get("embedding"):
                            logger.warning(f"No embedding data for {custom_id}")
                            continue

                        embedding = data[0]["embedding"]

                        # Parse custom_id: {article_id}_{chunk_idx}
                        if "_" not in custom_id:
                            logger.warning(f"Invalid custom_id format: {custom_id}")
                            continue

                        parts = custom_id.rsplit("_", 1)
                        if len(parts) != 2:
                            logger.warning(f"Invalid custom_id format: {custom_id}")
                            continue

                        article_id, chunk_idx_str = parts

                        # Get chunk text from input file
                        chunk_text = input_texts.get(custom_id, "")
                        if not chunk_text:
                            logger.warning(f"No input text found for {custom_id}")
                            continue

                        embeddings.append(
                            (article_id, chunk_idx_str, chunk_text, embedding)
                        )

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decode error in {result_file}:{line_num}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
            continue

    logger.info(f"Loaded {len(embeddings)} embeddings with chunk text")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process OpenAI batch embedding results and add them to a vector index"
    )

    parser.add_argument(
        "--results-dir",
        "-r",
        required=True,
        help="Directory containing batch files and result files",
    )

    parser.add_argument(
        "--index-name", "-i", required=True, help="Name of the vector index table"
    )

    parser.add_argument(
        "--source-table", "-s", required=True, help="Source table containing articles"
    )

    parser.add_argument(
        "--text-column", "-c", required=True, help="Column containing article text"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size used during embedding generation",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap used during embedding generation",
    )

    parser.add_argument(
        "--batch-size", type=int, default=100, help="Database batch size for insertions"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate results without inserting into database",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        logger.error(f"Invalid results directory: {results_dir}")
        sys.exit(1)

    try:
        logger.info(f"Loading batch results from {results_dir}")
        embeddings = load_batch_results(results_dir)

        if not embeddings:
            logger.error("No embeddings found in batch results")
            sys.exit(1)

        if args.dry_run:
            logger.info(f"Dry run: Found {len(embeddings)} embeddings")
            for i, (article_id, chunk_idx, chunk_text, _) in enumerate(embeddings[:3]):
                logger.info(f"  {article_id}_{chunk_idx}: {chunk_text[:50]}...")
            if len(embeddings) > 3:
                logger.info(f"  ... and {len(embeddings) - 3} more")
            return

        # Process embeddings
        settings = get_settings()
        logger.info("Connecting to database")
        db_conn = create_connection()

        validate_tables_exist(db_conn, [args.index_name, args.source_table])

        stats = add_precomputed_embeddings_to_vector_index(
            db_conn=db_conn,
            vector_index_name=args.index_name,
            source_table=args.source_table,
            text_column=args.text_column,
            embeddings_data=embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )

        logger.info(f"Processing complete: {stats}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if "db_conn" in locals():
            db_conn.close()  # type: ignore


if __name__ == "__main__":
    main()
