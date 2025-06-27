"""Create and submit OpenAI batch jobs for embedding generation."""

import argparse
from dataclasses import dataclass
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Optional
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.lex_db.database import create_connection
from src.lex_db.config import get_settings
from src.lex_db.utils import (
    get_logger,
    configure_logging,
    split_document_into_chunks,
    count_tokens,
)

logger = get_logger()


@dataclass
class BatchStatus:
    """Data class to hold batch status information."""

    status: str
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    request_counts: Optional[dict] = None
    error: Optional[str] = None


def create_batch_requests(
    articles: list[tuple[str, str]], chunk_size: int = 512, chunk_overlap: int = 50
) -> list[dict]:
    """Create batch API requests from articles."""
    requests = []

    for article_id, article_text in articles:
        if not article_text:
            continue

        chunks = split_document_into_chunks(
            article_text, chunk_size=chunk_size, overlap=chunk_overlap
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            custom_id = f"{article_id}_{chunk_idx}"
            token_count = count_tokens(chunk_text)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "text-embedding-ada-002", "input": chunk_text},
                "token_count": token_count,  # Add token count for partitioning
            }
            requests.append(request)

    return requests


def partition_requests(
    requests: list[dict],
    max_requests: int = 50000,
    max_tokens: int = 2800000,  # Leave some buffer under 3M limit
) -> list[list[dict]]:
    """Partition requests into batches respecting both request count and token limits."""
    batches = []
    current_batch: list[dict] = []
    current_request_count = 0
    current_token_count = 0

    for request in requests:
        request_tokens = request.get("token_count", 0)

        # Check if adding this request would exceed either limit
        if current_batch and (
            current_request_count + 1 > max_requests
            or current_token_count + request_tokens > max_tokens
        ):
            # Start new batch
            batches.append(current_batch)
            current_batch = []
            current_request_count = 0
            current_token_count = 0

        # Remove token_count from request before adding to batch (not needed in API call)
        clean_request = {k: v for k, v in request.items() if k != "token_count"}
        current_batch.append(clean_request)
        current_request_count += 1
        current_token_count += request_tokens

    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)

    # Log batch statistics
    for i, batch in enumerate(batches):
        batch_tokens = sum(count_tokens(req["body"]["input"]) for req in batch)
        logger.info(f"Batch {i}: {len(batch)} requests, ~{batch_tokens:,} tokens")

    return batches


def save_batch_file(
    batch_requests: list[dict], output_dir: Path, batch_idx: int
) -> Path:
    """Save batch requests to JSONL file."""
    batch_file = output_dir / f"batch_{batch_idx:03d}.jsonl"

    with open(batch_file, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    logger.info(f"Saved {len(batch_requests)} requests to {batch_file}")
    return batch_file


def save_progress(
    output_dir: Path,
    batch_idx: int,
    status: str,
    job_id: Optional[str] = None,
    **kwargs: (str | BatchStatus),
) -> None:
    """Save progress to a file, updating existing entries for the same batch."""
    progress_file = output_dir / "batch_progress.jsonl"

    # Load existing progress
    existing_progress = {}
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        existing_progress[entry["batch_idx"]] = entry
        except Exception as e:
            logger.warning(f"Could not load existing progress: {e}")

    # Update the entry for this batch
    existing_progress[batch_idx] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "batch_idx": batch_idx,
        "status": status,
        "job_id": job_id,
        **kwargs,
    }

    # Write all progress back to file (sorted by batch_idx)
    with open(progress_file, "w") as f:
        for idx in sorted(existing_progress.keys()):
            f.write(json.dumps(existing_progress[idx]) + "\n")


def load_progress(output_dir: Path) -> dict:
    """Load progress state from file."""
    progress_file = output_dir / "batch_progress.jsonl"

    if not progress_file.exists():
        return {}

    progress_state = {}
    try:
        with open(progress_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    batch_idx = entry["batch_idx"]
                    # Only keep the most recent entry for each batch_idx
                    progress_state[batch_idx] = entry
    except Exception as e:
        logger.warning(f"Could not load progress: {e}")

    return progress_state


def submit_batch_job(batch_file: Path) -> str:
    """Submit a batch file to OpenAI and return job ID."""
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Upload file
        with open(batch_file, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")

        # Create batch job
        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

        logger.info(f"Submitted batch job: {batch_response.id} for file {batch_file}")
        return batch_response.id

    except Exception as e:
        raise RuntimeError(f"Error submitting batch job: {str(e)}")


def check_batch_status(job_id: str) -> BatchStatus:
    """Check the status of a batch job."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        batch = client.batches.retrieve(job_id)
        return BatchStatus(
            status=batch.status,
            completed_at=batch.completed_at,
            failed_at=batch.failed_at,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            request_counts=batch.request_counts.__dict__
            if batch.request_counts
            else {},
        )
    except Exception as e:
        logger.error(f"Error checking batch status for {job_id}: {e}")
        return BatchStatus(status="unknown", error=str(e))


def download_batch_results(
    job_id: str, output_file_id: str, output_dir: Path, batch_idx: int
) -> Path:
    """Download batch results to a file."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Download the results
        file_response = client.files.content(output_file_id)

        # Save to file
        results_file = output_dir / f"batch_{batch_idx:03d}_results.jsonl"
        with open(results_file, "wb") as f:
            f.write(file_response.content)

        logger.info(f"Downloaded results for batch {batch_idx} to {results_file}")
        return results_file

    except Exception as e:
        logger.error(f"Error downloading results for job {job_id}: {e}")
        raise


def wait_for_batch_completion(
    job_id: str, check_interval: int = 60, timeout_hours: int = 25
) -> BatchStatus:
    """Wait for a batch job to complete, checking status periodically."""
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    logger.info(
        f"Waiting for batch {job_id} to complete (checking every {check_interval}s)"
    )

    while time.time() - start_time < timeout_seconds:
        status_info = check_batch_status(job_id)
        status = status_info.status if status_info else "unknown"

        if status in ["completed", "failed", "expired", "cancelled"]:
            logger.info(f"Batch {job_id} finished with status: {status}")
            return status_info
        elif status in ["validating", "in_progress", "finalizing"]:
            logger.info(f"Batch {job_id} status: {status}")
        else:
            logger.warning(f"Batch {job_id} unknown status: {status}")

        time.sleep(check_interval)

    logger.error(f"Batch {job_id} timed out after {timeout_hours} hours")
    return BatchStatus(status="timeout")


def process_batches_sequentially(
    output_dir: Path, total_batches: int, check_interval: int = 60, start_from: int = 0
) -> None:
    """Process batch files sequentially, waiting for each to complete."""
    progress_state = load_progress(output_dir)

    for batch_idx in range(start_from, total_batches):
        batch_file = output_dir / f"batch_{batch_idx:03d}.jsonl"

        if not batch_file.exists():
            logger.error(f"Batch file {batch_file} not found, skipping")
            continue

        # Check if this batch was already processed
        if batch_idx in progress_state:
            last_status = progress_state[batch_idx].get("status")
            if last_status == "downloaded":
                logger.info(
                    f"Batch {batch_idx} already completed and downloaded, skipping"
                )
                continue
            elif last_status in ["submitted", "completed"]:
                job_id = progress_state[batch_idx].get("job_id")
                if job_id:
                    logger.info(f"Resuming batch {batch_idx} with job ID {job_id}")
                    # Check current status and potentially download results
                    status_info = check_batch_status(job_id)
                    if status_info.status == "completed":
                        output_file_id = status_info.output_file_id
                        if output_file_id:
                            try:
                                download_batch_results(
                                    job_id, output_file_id, output_dir, batch_idx
                                )
                                save_progress(
                                    output_dir,
                                    batch_idx,
                                    "downloaded",
                                    job_id,
                                    output_file_id=output_file_id,
                                )
                                continue
                            except Exception as e:
                                logger.error(
                                    f"Failed to download results for batch {batch_idx}: {e}"
                                )

                    # If not completed, wait for completion
                    final_status = wait_for_batch_completion(job_id, check_interval)
                    if final_status.status == "completed":
                        output_file_id = final_status.output_file_id
                        if output_file_id:
                            download_batch_results(
                                job_id, output_file_id, output_dir, batch_idx
                            )
                            save_progress(
                                output_dir,
                                batch_idx,
                                "downloaded",
                                job_id,
                                output_file_id=output_file_id,
                            )
                    else:
                        save_progress(
                            output_dir, batch_idx, "failed", job_id, error=final_status
                        )
                    continue

        # Submit new batch
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: {batch_file}")

        try:
            # Submit the batch
            job_id = submit_batch_job(batch_file)
            save_progress(output_dir, batch_idx, "submitted", job_id)

            # Wait for completion
            final_status = wait_for_batch_completion(job_id, check_interval)

            if final_status.status == "completed":
                save_progress(output_dir, batch_idx, "completed", job_id)

                # Download results
                output_file_id = final_status.output_file_id
                if output_file_id:
                    download_batch_results(
                        job_id, output_file_id, output_dir, batch_idx
                    )
                    save_progress(
                        output_dir,
                        batch_idx,
                        "downloaded",
                        job_id,
                        output_file_id=output_file_id,
                    )
                else:
                    logger.error(f"No output file ID for completed batch {batch_idx}")
                    save_progress(
                        output_dir,
                        batch_idx,
                        "failed",
                        job_id,
                        error="No output file ID",
                    )
            else:
                logger.error(f"Batch {batch_idx} failed: {final_status}")
                save_progress(
                    output_dir, batch_idx, "failed", job_id, error=final_status
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            save_progress(output_dir, batch_idx, "failed", None, error=str(e))

        # Small delay between batches to be respectful
        time.sleep(5)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create and submit OpenAI batch jobs for embedding generation"
    )

    parser.add_argument(
        "--source-table", "-s", required=True, help="Source table containing articles"
    )

    parser.add_argument(
        "--text-column", "-c", required=True, help="Column containing article text"
    )

    parser.add_argument(
        "--output-dir", "-o", required=True, help="Directory to save batch files"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks for embedding (in tokens)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks (in tokens)",
    )

    parser.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=50000,
        help="Maximum requests per batch (OpenAI limit)",
    )

    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=2800000,
        help="Maximum tokens per batch (buffer under 3M OpenAI limit)",
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between status checks when waiting for batch completion",
    )

    parser.add_argument(
        "--start-from-batch",
        type=int,
        default=0,
        help="Start processing from this batch index (for resuming)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create batch files but don't submit to OpenAI",
    )

    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip batch creation and only process existing batch files",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    configure_logging(args.debug)

    # Get settings (loads from .env)
    settings = get_settings()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_batches = 0

        if not args.process_only:
            # Connect to database and fetch articles
            logger.info(f"Connecting to database at {settings.DATABASE_URL}")
            db_conn = create_connection()

            cursor = db_conn.cursor()
            cursor.execute(f"SELECT id, {args.text_column} FROM {args.source_table}")
            articles = cursor.fetchall()

            logger.info(f"Fetched {len(articles)} articles from {args.source_table}")

            # Create batch requests
            logger.info("Creating batch requests...")
            requests = create_batch_requests(
                articles, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
            )

            total_tokens = sum(req.get("token_count", 0) for req in requests)
            logger.info(
                f"Created {len(requests)} embedding requests (~{total_tokens:,} total tokens)"
            )

            # Partition into batches
            batches = partition_requests(
                requests, args.max_requests_per_batch, args.max_tokens_per_batch
            )
            total_batches = len(batches)
            logger.info(f"Partitioned into {total_batches} batches")

            # Save all batch files first
            logger.info("Saving all batch files...")
            for batch_idx, batch_requests in enumerate(batches):
                save_batch_file(batch_requests, output_dir, batch_idx)
                save_progress(output_dir, batch_idx, "pending", None)

            if "db_conn" in locals():
                db_conn.close()
        else:
            # Count existing batch files
            batch_files = list(output_dir.glob("batch_*.jsonl"))
            total_batches = len(
                [
                    f
                    for f in batch_files
                    if f.name.startswith("batch_")
                    and not f.name.endswith("_results.jsonl")
                ]
            )
            logger.info(f"Found {total_batches} existing batch files")

        if args.dry_run:
            logger.info("Dry run completed. Batch files created but not submitted.")
        else:
            logger.info(f"Starting sequential processing of {total_batches} batches...")
            process_batches_sequentially(
                output_dir, total_batches, args.check_interval, args.start_from_batch
            )
            logger.info("All batches processed!")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if "db_conn" in locals():
            db_conn.close()  # type: ignore


if __name__ == "__main__":
    main()
