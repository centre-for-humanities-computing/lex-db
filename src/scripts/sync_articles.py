"""CLI tool to synchronize articles from lex.dk sitemaps to PostgreSQL."""

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from lex_db.config import get_settings, VALID_ENCYCLOPEDIA_IDS
from lex_db.database import (
    get_db_connection,
    upsert_article,
    delete_articles,
)
from lex_db.sitemap import fetch_all_sitemaps, fetch_article_json, SitemapEntry
from lex_db.utils import configure_logging, get_logger
from lex_db.vector_store import get_all_vector_index_metadata, update_vector_index
from lex_db.embeddings import EmbeddingModel

logger = get_logger()


@dataclass
class ArticleMetadata:
    """Metadata for a single article from the database."""

    permalink: str
    encyclopedia_id: int
    changed_at: datetime | None


@dataclass
class VectorIndexStats:
    """Statistics for a single vector index update."""

    created: int = 0
    deleted: int = 0
    errors: int = 0
    error_message: str | None = None

    def is_error(self) -> bool:
        """Check if this represents an error state."""
        return self.error_message is not None


@dataclass
class CategorizedArticles:
    """Results of fetching and categorizing articles from sitemaps and database."""

    # Raw data
    sitemap_entries: list[SitemapEntry]
    db_metadata: dict[int, ArticleMetadata]
    new_urls: list[str]
    modified_urls: list[str]
    deleted_ids: list[int]

    # Metadata
    successful_sitemaps: set[int]
    total_sitemaps: int
    unchanged_article_count: int
    skip_deletions: bool = False


@dataclass
class SyncStats:
    """Statistics for article synchronization."""

    # Sitemap stats
    sitemap_entries_count: int = 0
    successful_sitemaps: int = 0
    total_sitemaps: int = 0

    # Article categorization
    new_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    deleted_count: int = 0

    # Fetch stats
    fetch_attempted: int = 0
    fetch_successful: int = 0

    # Database operations
    upsert_success: int = 0
    upsert_failure: int = 0
    articles_deleted: int = 0

    # Vector index stats
    vector_stats: dict[str, VectorIndexStats] = field(default_factory=dict)

    # Metadata
    db_articles_before: int = 0
    deletions_skipped: bool = False


def parse_encyclopedia_ids(ids_str: str | None) -> set[int] | None:
    """Parse comma-separated encyclopedia IDs into a set."""
    if not ids_str:
        return None

    try:
        ids = {int(id_str.strip()) for id_str in ids_str.split(",")}
        invalid = ids - VALID_ENCYCLOPEDIA_IDS
        if invalid:
            raise ValueError(f"Invalid encyclopedia IDs: {invalid}")
        return ids
    except ValueError as e:
        raise ValueError(f"Failed to parse encyclopedia IDs: {e}")


def fetch_article_metadata() -> dict[int, ArticleMetadata]:
    """
    Fetch article metadata from database.

    Returns:
        Dict mapping article_id to ArticleMetadata
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, permalink, encyclopedia_id, changed_at FROM articles"
        ).fetchall()

        return {
            row["id"]:  # type: ignore[call-overload]
            ArticleMetadata(
                permalink=row["permalink"],  # type: ignore[call-overload]
                encyclopedia_id=row["encyclopedia_id"],  # type: ignore[call-overload]
                changed_at=row["changed_at"],  # type: ignore[call-overload]
            )
            for row in rows
        }


def categorize_articles(
    sitemap_entries: list[SitemapEntry],
    db_metadata: dict[int, ArticleMetadata],
    encyclopedia_ids: set[int] | None,
) -> tuple[list[str], list[str], list[int], int]:
    """
    Categorize articles as new, modified, deleted, or unchanged.

    Args:
        sitemap_entries: List of entries from sitemaps
        db_metadata: Dict mapping article_id to ArticleMetadata
        encyclopedia_ids: Set of encyclopedia IDs being synced (for deletion filtering)

    Returns:
        Tuple of (new_urls, modified_urls, deleted_ids, unchanged_count)
    """
    # Build lookup: (encyclopedia_id, permalink) -> (article_id, changed_at)
    db_lookup = {
        (meta.encyclopedia_id, meta.permalink): (article_id, meta.changed_at)
        for article_id, meta in db_metadata.items()
    }

    # Build lookup: (encyclopedia_id, permalink) -> sitemap_entry
    sitemap_lookup = {
        (entry.encyclopedia_id, entry.permalink): entry for entry in sitemap_entries
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
        max_retries = 7
        base_delay = 5.0

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
                            # Exponential backoff: 5s, 10s, 20s, 40s, 80s, 160s, 320s
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                f"Rate limit hit for {url}, retrying in {delay}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue

                    # For other errors or final retry, log and return None
                    if attempt == max_retries - 1:
                        logger.warning(
                            f"Failed to fetch {url} after {max_retries} attempts: {e}"
                        )
                    return None

        return None

    tasks = [fetch_one_with_retry(url) for url in urls]
    results = await asyncio.gather(*tasks)

    # Filter out None values (failed fetches)
    return [article for article in results if article is not None]


async def fetch_and_categorize_articles(
    encyclopedia_ids: set[int] | None,
) -> CategorizedArticles | None:
    """
    Fetch sitemaps and database metadata, then categorize articles.

    Returns:
        CategorizedArticles with all categorization results,
        or None if a critical error occurred.
    """
    settings = get_settings()

    # Fetch sitemaps
    logger.info("Fetching sitemaps...")
    try:
        if encyclopedia_ids:
            sitemap_entries, successful_sitemaps = await fetch_all_sitemaps(
                encyclopedia_ids
            )
        else:
            sitemap_entries, successful_sitemaps = await fetch_all_sitemaps(
                set(VALID_ENCYCLOPEDIA_IDS)
            )
    except Exception as e:
        logger.error(f"Failed to fetch sitemaps: {e}", exc_info=True)
        return None

    logger.info(f"Fetched {len(sitemap_entries)} articles from sitemaps")
    logger.info(
        f"Successfully fetched {len(successful_sitemaps)}/{settings.SITEMAP_COUNT} sitemaps"
    )

    # Fetch database metadata
    logger.info("Fetching article metadata from database...")
    try:
        db_metadata = fetch_article_metadata()
    except Exception as e:
        logger.error(f"Failed to fetch database metadata: {e}", exc_info=True)
        return None

    logger.info(f"Found {len(db_metadata)} articles in database")

    # Categorize articles
    new_urls, modified_urls, deleted_ids, unchanged_count = categorize_articles(
        sitemap_entries, db_metadata, encyclopedia_ids
    )

    # If not all sitemaps were fetched successfully, skip deletions to avoid data loss
    deletions_skipped = len(successful_sitemaps) != settings.SITEMAP_COUNT
    if deletions_skipped:
        logger.warning(
            f"Only {len(successful_sitemaps)}/{settings.SITEMAP_COUNT} sitemaps fetched successfully. "
            "Skipping deletions to avoid incorrect removal of articles."
        )
        deleted_ids = []

    logger.info("Categorized articles:")
    logger.info(f"  New: {len(new_urls)}")
    logger.info(f"  Modified: {len(modified_urls)}")
    logger.info(f"  Unchanged: {unchanged_count}")
    logger.info(f"  Deleted: {len(deleted_ids)}")

    return CategorizedArticles(
        sitemap_entries=sitemap_entries,
        db_metadata=db_metadata,
        new_urls=new_urls,
        modified_urls=modified_urls,
        deleted_ids=deleted_ids,
        successful_sitemaps=successful_sitemaps,
        total_sitemaps=settings.SITEMAP_COUNT,
        unchanged_article_count=unchanged_count,
        skip_deletions=deletions_skipped,
    )


async def fetch_article_content(
    urls_to_fetch: list[str],
    batch_size: int,
) -> list[dict]:
    """
    Fetch article JSON content for the given URLs.

    Returns:
        List of successfully fetched article data dictionaries.
    """
    settings = get_settings()
    fetched_articles: list[dict] = []

    if not urls_to_fetch:
        return fetched_articles

    logger.info(f"Fetching {len(urls_to_fetch)} articles...")
    logger.info(f"Rate limit: {settings.SITEMAP_RATE_LIMIT} concurrent requests")

    for i in range(0, len(urls_to_fetch), batch_size):
        batch = urls_to_fetch[i : i + batch_size]
        logger.info(f"Fetching batch {i // batch_size + 1} ({len(batch)} articles)...")

        batch_articles = await fetch_articles_batch(
            batch, max_concurrent=settings.SITEMAP_RATE_LIMIT
        )
        fetched_articles.extend(batch_articles)

        logger.info(
            f"Successfully fetched {len(batch_articles)}/{len(batch)} articles in batch"
        )

    logger.info(f"Total fetched: {len(fetched_articles)}/{len(urls_to_fetch)} articles")
    return fetched_articles


def upsert_articles_to_db(
    articles: list[dict],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Upsert articles to the database.

    Returns:
        Tuple of (success_count, failure_count)
    """
    if not articles:
        return 0, 0

    if dry_run:
        logger.info(f"DRY RUN: Would upsert {len(articles)} articles")
        return len(articles), 0

    logger.info(f"Upserting {len(articles)} articles...")
    success_count = 0
    failure_count = 0

    for article in articles:
        if upsert_article(article):
            success_count += 1
        else:
            failure_count += 1

    logger.info(f"Upserted {success_count} articles successfully")
    if failure_count > 0:
        logger.warning(f"Failed to upsert {failure_count} articles")

    return success_count, failure_count


def delete_missing_articles_from_db(
    deleted_ids: list[int],
    dry_run: bool,
) -> int:
    """
    Delete articles that are no longer in sitemaps.

    Returns:
        Number of articles deleted.
    """
    if not deleted_ids:
        return 0

    if dry_run:
        logger.info(f"DRY RUN: Would delete {len(deleted_ids)} articles")
        return len(deleted_ids)

    logger.info(f"Deleting {len(deleted_ids)} articles no longer in sitemaps...")
    deleted_count = delete_articles(deleted_ids)
    logger.info(f"Deleted {deleted_count} articles")
    return deleted_count


def update_all_vector_indexes(
    batch_size: int,
    dry_run: bool,
) -> dict[str, VectorIndexStats]:
    """
    Update all vector indexes.

    Returns:
        Dictionary mapping index name to stats.
    """
    if dry_run:
        logger.info("DRY RUN: Would update vector indexes")
        return {}

    logger.info("Updating vector indexes...")
    vector_stats: dict[str, VectorIndexStats] = {}

    with get_db_connection() as conn:
        all_indexes = get_all_vector_index_metadata(conn)

        if not all_indexes:
            logger.info("No vector indexes found, skipping update")
            return vector_stats

        logger.info(f"Found {len(all_indexes)} vector index(es) to update")

        for index_meta in all_indexes:
            index_name = index_meta["index_name"]
            logger.info(f"Updating vector index: {index_name}")

            try:
                stats = update_vector_index(
                    db_conn=conn,
                    vector_index_name=index_name,
                    source_table=index_meta["source_table"],
                    text_column=index_meta["source_column"],
                    embedding_model_choice=EmbeddingModel(
                        index_meta["embedding_model"]
                    ),
                    updated_at_column=index_meta.get("updated_at_column", "changed_at"),
                    chunk_size=int(index_meta["chunk_size"]),
                    chunk_overlap=int(index_meta["chunk_overlap"]),
                    chunking_strategy=index_meta["chunking_strategy"],
                    batch_size=batch_size,
                )
                vector_stats[index_name] = VectorIndexStats(
                    created=stats["created"],
                    deleted=stats["deleted"],
                    errors=stats["errors"],
                )
                logger.info(
                    f"Updated {index_name}: "
                    f"{stats['created']} created, {stats['deleted']} deleted, "
                    f"{stats['errors']} errors"
                )
            except Exception as e:
                logger.error(
                    f"Failed to update vector index {index_name}: {e}",
                    exc_info=True,
                )
                vector_stats[index_name] = VectorIndexStats(error_message=str(e))

    return vector_stats


def log_sync_summary(
    stats: SyncStats,
    dry_run: bool,
    encyclopedia_ids: set[int] | None,
    batch_size: int,
) -> None:
    """Log a summary of the synchronization."""
    logger.info("=" * 60)
    logger.info("SYNCHRONIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Encyclopedia IDs: {encyclopedia_ids or 'all'}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")
    logger.info("Articles:")
    logger.info(f"  Total in sitemaps: {stats.sitemap_entries_count}")
    logger.info(f"  Total in database (before sync): {stats.db_articles_before}")
    logger.info(f"  New: {stats.new_count}")
    logger.info(f"  Modified: {stats.modified_count}")
    logger.info(f"  Unchanged: {stats.unchanged_count}")
    logger.info(f"  Deleted: {stats.articles_deleted}")
    if stats.deletions_skipped:
        logger.info(
            f"  Note: Deletions skipped due to incomplete sitemap fetching "
            f"({stats.successful_sitemaps}/{stats.total_sitemaps} sitemaps successful)"
        )
    logger.info("")
    logger.info("Fetch results:")
    logger.info(f"  Attempted: {stats.fetch_attempted}")
    logger.info(f"  Successful: {stats.fetch_successful}")
    logger.info(f"  Failed: {stats.fetch_attempted - stats.fetch_successful}")
    logger.info("")
    logger.info("Database operations:")
    logger.info(f"  Upserts successful: {stats.upsert_success}")
    logger.info(f"  Upserts failed: {stats.upsert_failure}")
    logger.info(f"  Articles deleted: {stats.articles_deleted}")

    if stats.vector_stats:
        logger.info("")
        logger.info("Vector index updates:")
        for index_name, index_stats in stats.vector_stats.items():
            if index_stats.is_error():
                logger.info(f"  {index_name}: ERROR - {index_stats.error_message}")
            else:
                logger.info(
                    f"  {index_name}: {index_stats.created} created, "
                    f"{index_stats.deleted} deleted, {index_stats.errors} errors"
                )

    logger.info("=" * 60)
    logger.info("Article synchronization completed successfully")


async def sync_articles_async(
    dry_run: bool,
    batch_size: int,
    encyclopedia_ids: set[int] | None,
) -> None:
    """
    Main async workflow for article synchronization.
    """

    logger.info("Starting article synchronization")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encyclopedia IDs: {encyclopedia_ids or 'all'}")

    # Step 1-2: Fetch and categorize articles
    categorized = await fetch_and_categorize_articles(encyclopedia_ids)
    if categorized is None:
        raise RuntimeError("Failed to fetch and categorize articles")

    # Initialize stats
    stats = SyncStats(
        sitemap_entries_count=len(categorized.sitemap_entries),
        successful_sitemaps=len(categorized.successful_sitemaps),
        total_sitemaps=categorized.total_sitemaps,
        new_count=len(categorized.new_urls),
        modified_count=len(categorized.modified_urls),
        unchanged_count=categorized.unchanged_article_count,
        db_articles_before=len(categorized.db_metadata),
        deletions_skipped=categorized.skip_deletions,
    )

    # Step 3: Fetch article JSON
    urls_to_fetch = categorized.new_urls + categorized.modified_urls
    stats.fetch_attempted = len(urls_to_fetch)

    fetched_articles = await fetch_article_content(urls_to_fetch, batch_size)
    stats.fetch_successful = len(fetched_articles)

    # Step 4: Upsert articles
    stats.upsert_success, stats.upsert_failure = upsert_articles_to_db(
        fetched_articles, dry_run
    )

    # Step 5: Delete missing articles
    stats.articles_deleted = delete_missing_articles_from_db(
        categorized.deleted_ids, dry_run
    )

    # Step 6: Update vector indexes
    if stats.upsert_success > 0 or stats.articles_deleted > 0:
        stats.vector_stats = update_all_vector_indexes(batch_size, dry_run)

    # Step 7: Log summary
    log_sync_summary(stats, dry_run, encyclopedia_ids, batch_size)


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
        asyncio.run(
            sync_articles_async(
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                encyclopedia_ids=encyclopedia_ids,
            )
        )
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Critical failure: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
