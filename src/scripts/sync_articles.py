"""CLI tool to synchronize articles from lex.dk sitemaps to PostgreSQL."""

import argparse
import asyncio
from datetime import datetime

from lex_db.config import get_settings
from lex_db.database import get_db_connection, get_url_base, upsert_article
from lex_db.sitemap import fetch_all_sitemaps, fetch_article_json, SitemapEntry
from lex_db.utils import configure_logging, get_logger

logger = get_logger()


def parse_encyclopedia_ids(ids_str: str | None) -> set[int] | None:
    """Parse comma-separated encyclopedia IDs into a set."""
    if not ids_str:
        return None
    
    try:
        ids = {int(id_str.strip()) for id_str in ids_str.split(",")}
        valid_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20}
        invalid = ids - valid_ids
        if invalid:
            raise ValueError(f"Invalid encyclopedia IDs: {invalid}")
        return ids
    except ValueError as e:
        raise ValueError(f"Failed to parse encyclopedia IDs: {e}")


def fetch_article_metadata() -> dict[int, tuple[str, int, datetime | None]]:
    """
    Fetch article metadata from database.
    
    Returns:
        Dict mapping article_id to (permalink, encyclopedia_id, changed_at)
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, permalink, encyclopedia_id, changed_at FROM articles"
        ).fetchall()
        
        return {
            row["id"]: (row["permalink"], row["encyclopedia_id"], row["changed_at"])
            for row in rows
        }


def categorize_articles(
    sitemap_entries: list[SitemapEntry],
    db_metadata: dict[int, tuple[str, int, datetime | None]],
    encyclopedia_ids: set[int] | None,
) -> tuple[list[str], list[str], list[int], int]:
    """
    Categorize articles as new, modified, deleted, or unchanged.
    
    Args:
        sitemap_entries: List of entries from sitemaps
        db_metadata: Dict mapping article_id to (permalink, encyclopedia_id, changed_at)
        encyclopedia_ids: Set of encyclopedia IDs being synced (for deletion filtering)
    
    Returns:
        Tuple of (new_urls, modified_urls, deleted_ids, unchanged_count)
    """
    # Build lookup: (encyclopedia_id, permalink) -> (article_id, changed_at)
    db_lookup = {
        (enc_id, permalink): (article_id, changed_at)
        for article_id, (permalink, enc_id, changed_at) in db_metadata.items()
    }
    
    # Build lookup: (encyclopedia_id, permalink) -> sitemap_entry
    sitemap_lookup = {
        (entry.encyclopedia_id, entry.permalink): entry
        for entry in sitemap_entries
    }
    
    new_urls = []
    modified_urls = []
    unchanged_count = 0
    
    # Find new, modified, and unchanged articles
    for (enc_id, permalink), entry in sitemap_lookup.items():
        key = (enc_id, permalink)
        
        if key not in db_lookup:
            new_urls.append(entry.url)
        else:
            article_id, db_changed_at = db_lookup[key]
            
            # Compare timestamps using database timezone
            if db_changed_at is None or entry.lastmod > db_changed_at:
                modified_urls.append(entry.url)
            else:
                unchanged_count += 1
    
    # Find deleted articles (in DB but not in sitemap)
    # Only consider articles from encyclopedias we're syncing
    deleted_ids = []
    for (enc_id, permalink), (article_id, _) in db_lookup.items():
        if (enc_id, permalink) not in sitemap_lookup:
            # Only delete if we're syncing this encyclopedia
            if encyclopedia_ids is None or enc_id in encyclopedia_ids:
                deleted_ids.append(article_id)
    
    return new_urls, modified_urls, deleted_ids, unchanged_count


async def fetch_articles_batch(urls: list[str], max_concurrent: int = 10) -> list[dict]:
    """
    Fetch article JSON data for a list of URLs with rate limiting.
    
    Args:
        urls: List of article URLs to fetch
        max_concurrent: Maximum number of concurrent requests
    
    Returns:
        List of successfully fetched article data dictionaries
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_one_with_retry(url: str) -> dict | None:
        max_retries = 5
        base_delay = 1.0
        
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await fetch_article_json(url)
                    # Small delay between successful requests to avoid rate limits
                    await asyncio.sleep(0.1)
                    return result
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if it's a 429 rate limit error
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        if attempt < max_retries - 1:
                            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Rate limit hit for {url}, retrying in {delay}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    # For other errors or final retry, log and return None
                    if attempt == max_retries - 1:
                        logger.warning(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    tasks = [fetch_one_with_retry(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out None values (failed fetches)
    return [article for article in results if article is not None]


async def sync_articles_async(
    dry_run: bool,
    batch_size: int,
    encyclopedia_ids: set[int] | None,
) -> int:
    """
    Main async workflow for article synchronization.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    settings = get_settings()
    
    logger.info("Starting article synchronization")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encyclopedia IDs: {encyclopedia_ids or 'all'}")
    
    # Step 1: Fetch sitemaps
    logger.info("Fetching sitemaps...")
    try:
        if encyclopedia_ids:
            sitemap_entries = await fetch_all_sitemaps(encyclopedia_ids)
        else:
            # Default to all encyclopedias
            sitemap_entries = await fetch_all_sitemaps({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20})
    except Exception as e:
        logger.error(f"Failed to fetch sitemaps: {e}", exc_info=True)
        return 1
    
    logger.info(f"Fetched {len(sitemap_entries)} articles from sitemaps")
    
    # Step 2: Compare with database
    logger.info("Fetching article metadata from database...")
    try:
        db_metadata = fetch_article_metadata()
    except Exception as e:
        logger.error(f"Failed to fetch database metadata: {e}", exc_info=True)
        return 1
    
    logger.info(f"Found {len(db_metadata)} articles in database")
    
    # Categorize articles
    new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
        sitemap_entries, db_metadata, encyclopedia_ids
    )
    
    logger.info(f"Categorized articles:")
    logger.info(f"  New: {len(new_urls)}")
    logger.info(f"  Modified: {len(modified_urls)}")
    logger.info(f"  Unchanged: {unchanged_count}")
    logger.info(f"  Deleted: {len(deleted_ids)}")
    
    # Step 3: Fetch article JSON
    urls_to_fetch = new_urls + modified_urls
    fetched_articles = []
    
    if urls_to_fetch:
        logger.info(f"Fetching {len(urls_to_fetch)} articles...")
        logger.info(f"Rate limit: {settings.SITEMAP_RATE_LIMIT} concurrent requests")
        
        for i in range(0, len(urls_to_fetch), batch_size):
            batch = urls_to_fetch[i:i + batch_size]
            logger.info(f"Fetching batch {i // batch_size + 1} ({len(batch)} articles)...")
            
            batch_articles = await fetch_articles_batch(batch, max_concurrent=settings.SITEMAP_RATE_LIMIT)
            fetched_articles.extend(batch_articles)
            
            logger.info(f"Successfully fetched {len(batch_articles)}/{len(batch)} articles in batch")
        
        logger.info(f"Total fetched: {len(fetched_articles)}/{len(urls_to_fetch)} articles")
    
    # Step 4: Batch upsert articles
    upsert_success_count = 0
    upsert_failure_count = 0
    
    if fetched_articles:
        if dry_run:
            logger.info(f"DRY RUN: Would upsert {len(fetched_articles)} articles")
            upsert_success_count = len(fetched_articles)
        else:
            logger.info(f"Upserting {len(fetched_articles)} articles...")
            
            for article in fetched_articles:
                if upsert_article(article):
                    upsert_success_count += 1
                else:
                    upsert_failure_count += 1
            
            logger.info(f"Upserted {upsert_success_count} articles successfully")
            if upsert_failure_count > 0:
                logger.warning(f"Failed to upsert {upsert_failure_count} articles")
    
    # TODO: Step 5: Delete missing articles
    # TODO: Step 6: Update vector indexes
    # TODO: Step 7: Log summary statistics
    
    logger.info("Article synchronization completed")
    return 0


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Synchronize articles from lex.dk sitemaps to PostgreSQL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without modifying database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of articles to process per batch (default: 1000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--encyclopedia-ids",
        type=str,
        help="Comma-separated encyclopedia IDs to sync (e.g., '14,15,18')",
    )
    
    args = parser.parse_args()
    configure_logging(args.debug)
    
    try:
        encyclopedia_ids = parse_encyclopedia_ids(args.encyclopedia_ids)
        exit_code = asyncio.run(
            sync_articles_async(
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                encyclopedia_ids=encyclopedia_ids,
            )
        )
        exit(exit_code)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Critical failure: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
