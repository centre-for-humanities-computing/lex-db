"""Sitemap fetching and parsing for lex.dk."""

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote, urlparse

import httpx

from lex_db.config import get_settings
from lex_db.utils import get_logger

logger = get_logger()


@dataclass
class SitemapEntry:
    """Represents a single entry from a sitemap."""

    url: str
    lastmod: datetime  # UTC timezone-aware
    encyclopedia_id: int
    permalink: str


def derive_encyclopedia_id(url: str) -> int:
    """
    Derive encyclopedia ID from URL subdomain.

    Args:
        url: Full article URL (e.g., "https://denstoredanske.lex.dk/article")

    Returns:
        Encyclopedia ID (1-20, excluding 13)

    Raises:
        ValueError: If subdomain is not recognized
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise ValueError(f"Invalid URL: {url}")

    # Map subdomain to encyclopedia ID (reverse of get_url_base())
    subdomain_map = {
        "denstoredanske.lex.dk": 1,
        "trap.lex.dk": 2,
        "biografiskleksikon.lex.dk": 3,
        "gyldendalogpolitikensdanmarkshistorie.lex.dk": 4,
        "danmarksoldtid.lex.dk": 5,
        "teaterleksikon.lex.dk": 6,
        "mytologi.lex.dk": 7,
        "pattedyratlas.lex.dk": 8,
        "dansklitteraturshistorie.lex.dk": 9,
        "bornelitteratur.lex.dk": 10,
        "symbolleksikon.lex.dk": 11,
        "naturenidanmark.lex.dk": 12,
        "om.lex.dk": 14,
        "lex.dk": 15,
        "kvindebiografiskleksikon.lex.dk": 16,
        "medicin.lex.dk": 17,
        "trap-groenland.lex.dk": 18,
        "trap-faeroeerne.lex.dk": 19,
        "danmarkshistorien.lex.dk": 20,
    }

    if hostname not in subdomain_map:
        raise ValueError(f"Unknown subdomain: {hostname}")

    return subdomain_map[hostname]


def derive_permalink(url: str) -> str:
    """
    Derive permalink from URL path.

    Args:
        url: Full article URL (e.g., "https://lex.dk/eksteri%C3%B8r")

    Returns:
        URL-decoded path without leading/trailing slashes (e.g., "eksteriør")
    """
    parsed = urlparse(url)
    path = parsed.path

    # URL decode and strip slashes
    decoded_path = unquote(path)
    permalink = decoded_path.strip("/")

    return permalink


def parse_sitemap(xml_content: str) -> list[SitemapEntry]:
    """
    Parse sitemap XML and extract entries.

    Args:
        xml_content: Raw XML content from sitemap

    Returns:
        List of SitemapEntry objects
    """
    entries: list[SitemapEntry] = []

    try:
        root = ET.fromstring(xml_content)

        # Handle XML namespace
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        for url_element in root.findall("ns:url", namespace):
            loc_element = url_element.find("ns:loc", namespace)
            lastmod_element = url_element.find("ns:lastmod", namespace)

            if loc_element is None or loc_element.text is None:
                logger.warning("Skipping entry without <loc> element")
                continue

            if lastmod_element is None or lastmod_element.text is None:
                logger.warning(f"Skipping entry without <lastmod>: {loc_element.text}")
                continue

            url = loc_element.text
            lastmod_str = lastmod_element.text

            try:
                # Parse ISO 8601 datetime and ensure it's UTC
                lastmod = datetime.fromisoformat(lastmod_str.replace("Z", "+00:00"))

                encyclopedia_id = derive_encyclopedia_id(url)
                permalink = derive_permalink(url)

                entry = SitemapEntry(
                    url=url,
                    lastmod=lastmod,
                    encyclopedia_id=encyclopedia_id,
                    permalink=permalink,
                )
                entries.append(entry)

            except (ValueError, Exception) as e:
                logger.warning(f"Error parsing entry {url}: {e}")
                continue

    except ET.ParseError as e:
        logger.error(f"Failed to parse sitemap XML: {e}")
        raise

    return entries


async def fetch_sitemap(url: str) -> str:
    """
    Fetch sitemap XML from URL.

    Args:
        url: Sitemap URL

    Returns:
        Raw XML content

    Raises:
        httpx.HTTPError: If request fails after retries
    """
    settings = get_settings()

    async with httpx.AsyncClient(timeout=settings.SITEMAP_REQUEST_TIMEOUT) as client:
        for attempt in range(settings.SITEMAP_MAX_RETRIES):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text

            except httpx.HTTPError as e:
                if attempt == settings.SITEMAP_MAX_RETRIES - 1:
                    logger.error(
                        f"Failed to fetch {url} after {settings.SITEMAP_MAX_RETRIES} attempts: {e}"
                    )
                    raise
                logger.warning(
                    f"Attempt {attempt + 1} failed for {url}: {e}, retrying..."
                )
                await asyncio.sleep(1)  # Brief delay before retry

    # This should never be reached due to the raise in the loop
    raise RuntimeError(f"Unexpected error fetching {url}")


async def fetch_article_json(url: str) -> dict:
    """
    Fetch full article JSON from endpoint.

    Args:
        url: Article URL (e.g., "https://lex.dk/eksteriør")

    Returns:
        Article data as dictionary

    Raises:
        httpx.HTTPError: If request fails
    """
    settings = get_settings()
    json_url = f"{url}.json"

    async with httpx.AsyncClient(timeout=settings.SITEMAP_REQUEST_TIMEOUT) as client:
        response = await client.get(json_url)
        response.raise_for_status()
        return dict(response.json())


async def fetch_all_sitemaps(
    encyclopedia_ids: set[int] = {14, 15, 18, 19, 20},
) -> list[SitemapEntry]:
    """
    Fetch and parse all sitemaps from lex.dk.

    Args:
        encyclopedia_ids: Set of encyclopedia IDs to filter by.

    Returns:
        List of SitemapEntry objects

    Raises:
        ValueError: If invalid encyclopedia_ids provided
    """
    settings = get_settings()

    # Validate encyclopedia_ids
    valid_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20}
    invalid_ids = encyclopedia_ids - valid_ids
    if invalid_ids:
        raise ValueError(f"Invalid encyclopedia IDs: {invalid_ids}")

    # Fetch all 6 sitemaps concurrently
    sitemap_urls = [f"{settings.SITEMAP_BASE_URL}/sitemap{i}.xml" for i in range(1, 7)]

    async def fetch_and_parse(url: str) -> list[SitemapEntry]:
        try:
            xml_content = await fetch_sitemap(url)
            return parse_sitemap(xml_content)
        except Exception as e:
            logger.error(f"Failed to process sitemap {url}: {e}")
            return []  # Continue with other sitemaps

    tasks = [fetch_and_parse(url) for url in sitemap_urls]
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_entries: list[SitemapEntry] = []
    for entries in results:
        all_entries.extend(entries)

    # Filter by encyclopedia_ids
    all_entries = [
        entry for entry in all_entries if entry.encyclopedia_id in encyclopedia_ids
    ]

    # Deduplicate by URL, keeping most recent lastmod
    url_to_entry: dict[str, SitemapEntry] = {}
    for entry in all_entries:
        if entry.url not in url_to_entry:
            url_to_entry[entry.url] = entry
        else:
            # Keep entry with most recent lastmod
            if entry.lastmod > url_to_entry[entry.url].lastmod:
                url_to_entry[entry.url] = entry

    deduplicated_entries = list(url_to_entry.values())

    logger.info(
        f"Fetched {len(deduplicated_entries)} unique entries from {len(sitemap_urls)} sitemaps"
    )

    return deduplicated_entries
