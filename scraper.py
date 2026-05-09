"""
SHL Product Catalog Scraper — Individual Test Solutions

Scrapes all Individual Test Solution entries from SHL's product catalog,
visits each detail page for full metadata, and saves as JSON + CSV.

Design Decisions:
- Static scraping with requests + BeautifulSoup (no Selenium/Playwright needed)
  because the catalog is fully server-side rendered (SSR).
- Pagination: type=1 is Individual Test Solutions, 12 items/page, 32 pages.
  We iterate start=0,12,24,...,372 and stop when a page yields 0 new items.
- Each listing page only gives name + URL. Detail pages provide: description,
  job levels, languages, assessment length, test type, remote testing, downloads.
- Rate limiting with 1-second delays between requests.
- Retry logic with exponential backoff for transient failures.
- Idempotent: re-running overwrites output files cleanly.

Test Type Legend (from SHL's catalog footer):
  A = Ability & Aptitude
  B = Biodata & Situational Judgement
  C = Competencies
  D = Development & 360
  E = Assessment Exercises
  K = Knowledge & Skills
  P = Personality & Behavior
  S = Simulations
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
import os
import sys
import logging
from urllib.parse import urljoin


BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"
ITEMS_PER_PAGE = 12
TYPE_INDIVIDUAL = 1  
MAX_PAGES = 40  
REQUEST_DELAY = 1.0  
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "shl_individual_tests.json")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "shl_individual_tests.csv")

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, "scraper.log"), mode="w", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a page with retry logic and return parsed BeautifulSoup, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            wait = RETRY_BACKOFF**attempt
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}. "
                f"Retrying in {wait:.0f}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
                return None


def collect_product_urls() -> list[dict]:
    """Paginate through the catalog and collect name + URL for each product."""
    products = []
    seen_urls = set()

    for page_num in range(MAX_PAGES):
        start = page_num * ITEMS_PER_PAGE
        url = f"{CATALOG_URL}?start={start}&type={TYPE_INDIVIDUAL}"
        logger.info(f"Fetching listing page {page_num + 1} (start={start})...")

        soup = fetch_page(url)
        if soup is None:
            logger.error(f"Failed to fetch listing page {page_num + 1}, stopping.")
            break

        links = soup.find_all("a", href=re.compile(r"/products/product-catalog/view/"))
        new_count = 0
        for link in links:
            href = urljoin(BASE_URL, link["href"])
            name = link.get_text(strip=True)
            if href not in seen_urls and name:
                seen_urls.add(href)
                products.append({"name": name, "url": href})
                new_count += 1

        logger.info(f"  Found {new_count} new products (total so far: {len(products)})")

        if new_count == 0:
            logger.info("No new products found — reached end of catalog.")
            break

        time.sleep(REQUEST_DELAY)

    return products

def scrape_detail(product: dict) -> dict:
    """Visit a product detail page and extract all metadata fields."""
    url = product["url"]
    soup = fetch_page(url)
    if soup is None:
        logger.warning(f"Could not fetch detail page: {url}")
        product.update(
            {
                "description": None,
                "job_levels": None,
                "languages": None,
                "assessment_length_minutes": None,
                "test_type": None,
                "test_type_name": None,
                "remote_testing": None,
                "adaptive_irt": None,
                "fact_sheet_url": None,
                "scrape_error": True,
            }
        )
        return product

    content_area = soup.find("div", class_=re.compile(r"product-catalog", re.I))
    if not content_area:
        content_area = soup
    h4_tags = content_area.find_all("h4")

    sections = {}
    for h4 in h4_tags:
        title = h4.get_text(strip=True).lower()
        content_parts = []
        for sibling in h4.find_next_siblings():
            if sibling.name == "h4":
                break
            text = sibling.get_text(strip=True)
            if text:
                content_parts.append(text)
        sections[title] = "\n".join(content_parts)

    description = sections.get("description", "")

    job_levels = sections.get("job levels", "").rstrip(",").strip()

    languages = sections.get("languages", "").rstrip(",").strip()

    assessment_section = sections.get("assessment length", "")

    duration_match = re.search(
        r"Completion Time in minutes\s*=\s*(\d+)", assessment_section
    )
    assessment_length = int(duration_match.group(1)) if duration_match else None

    test_type_match = re.search(r"Test Type:\s*([A-Z])", assessment_section)
    test_type = test_type_match.group(1) if test_type_match else None
    test_type_name = TEST_TYPE_MAP.get(test_type) if test_type else None

    remote_match = re.search(r"Remote Testing:\s*(Yes|No)?", assessment_section, re.I)
    if remote_match and remote_match.group(1):
        remote_testing = remote_match.group(1).capitalize()
    else:
        remote_testing = "Yes" if "Remote Testing" in assessment_section else None

    adaptive_match = re.search(
        r"Adaptive/IRT:\s*(Yes|No)?", assessment_section, re.I
    )
    adaptive_irt = None
    if adaptive_match and adaptive_match.group(1):
        adaptive_irt = adaptive_match.group(1).capitalize()

    fact_sheet_url = None
    downloads_section = sections.get("downloads", "")
    fact_sheet_link = content_area.find("a", href=re.compile(r"Fact_Sheet", re.I))
    if fact_sheet_link:
        fact_sheet_url = fact_sheet_link["href"]
    else:
        fact_sheet_link = content_area.find("a", string=re.compile(r"Fact Sheet", re.I))
        if fact_sheet_link:
            fact_sheet_url = fact_sheet_link.get("href")

    product.update(
        {
            "description": description or None,
            "job_levels": job_levels or None,
            "languages": languages or None,
            "assessment_length_minutes": assessment_length,
            "test_type": test_type,
            "test_type_name": test_type_name,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "fact_sheet_url": fact_sheet_url,
            "scrape_error": False,
        }
    )
    return product

def save_json(data: list[dict], path: str):
    """Save data as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} records to {path}")


def save_csv(data: list[dict], path: str):
    """Save data as CSV with all fields."""
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Saved {len(data)} records to {path}")

def main():
    logger.info("=" * 60)
    logger.info("SHL Product Catalog Scraper — Individual Test Solutions")
    logger.info("=" * 60)

    logger.info("\n--- Phase 1: Collecting product URLs from listing pages ---")
    products = collect_product_urls()
    logger.info(f"\nTotal products found: {len(products)}")

    if not products:
        logger.error("No products found! Exiting.")
        return

    logger.info("\n--- Phase 2: Scraping detail pages ---")
    total = len(products)
    for i, product in enumerate(products, 1):
        logger.info(f"[{i}/{total}] Scraping: {product['name']}")
        scrape_detail(product)
        if i < total:
            time.sleep(REQUEST_DELAY)

    successful = sum(1 for p in products if not p.get("scrape_error"))
    failed = sum(1 for p in products if p.get("scrape_error"))
    logger.info(f"\nScraping complete: {successful} succeeded, {failed} failed")

    logger.info("\n--- Phase 3: Saving output ---")
    save_json(products, JSON_OUTPUT)
    save_csv(products, CSV_OUTPUT)

    logger.info("\n--- Sample output (first 5 entries) ---")
    for p in products[:5]:
        logger.info(json.dumps(p, indent=2, ensure_ascii=False))

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
