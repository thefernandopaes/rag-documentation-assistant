"""
Async Document Processor - High-Performance Web Crawler

Performance improvements:
- httpx async client: Non-blocking HTTP requests
- Concurrent URL processing: 5 URLs in parallel
- Batch processing with asyncio.gather()
- Rate limiting with Semaphore
- Async sleep for respectful crawling

Expected performance: 5-10x faster than sync version
"""

import logging
import time
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
import httpx
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse
from config import Config
from data.sample_docs import (
    SAMPLE_REACT_DOCS,
    SAMPLE_PYTHON_DOCS,
    SAMPLE_FASTAPI_DOCS,
    SAMPLE_DOCKER_DOCS,
    SAMPLE_AWS_LAMBDA_DOCS
)

logger = logging.getLogger(__name__)


class AsyncDocumentProcessor:
    """Async document processor with concurrent crawling."""

    def __init__(self, max_concurrent: int = 5):
        """
        Initialize async document processor.

        Args:
            max_concurrent: Maximum concurrent HTTP requests (default: 5)
        """
        self.timeout = Config.DOC_CRAWL_TIMEOUT
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Headers for respectful crawling
        self.headers = {
            'User-Agent': 'DocRag Documentation Crawler/2.0 (+https://example.com)'
        }

        logger.info(f"AsyncDocumentProcessor initialized (max_concurrent={max_concurrent})")

    async def process_documentation_sources(self) -> List[Dict[str, Any]]:
        """
        Process all configured documentation sources concurrently.

        Performance: Processes multiple sources in parallel instead of sequentially.
        """
        logger.info("Starting async documentation processing...")
        start_time = time.time()

        # Create tasks for all sources (CONCURRENT PROCESSING)
        tasks = []
        for source_name, source_config in Config.DOC_SOURCES.items():
            task = self._process_source_with_error_handling(source_name, source_config)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_documents = []
        for docs in results:
            if docs:
                all_documents.extend(docs)

        elapsed = time.time() - start_time
        logger.info(f"Async processing complete: {len(all_documents)} documents in {elapsed:.2f}s")

        return all_documents

    async def _process_source_with_error_handling(
        self,
        source_name: str,
        source_config: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Process a single source with error handling."""
        logger.info(f"Processing {source_name} documentation...")
        try:
            docs = await self.process_source(source_config)
            logger.info(f"Processed {len(docs)} documents from {source_name}")
            return docs
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            return []

    async def process_source(self, source_config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process a single documentation source (async)."""
        if Config.DOC_USE_SAMPLE:
            # Use sample data (no async needed for in-memory data)
            doc_type = source_config['type']
            if doc_type == 'react':
                return await self._process_react_docs(source_config)
            elif doc_type == 'python':
                return await self._process_python_docs(source_config)
            elif doc_type == 'fastapi':
                return await self._process_fastapi_docs(source_config)
            elif doc_type == 'docker':
                return await self._process_docker_docs(source_config)
            elif doc_type == 'aws_lambda':
                return await self._process_aws_lambda_docs(source_config)
            return []

        # Real async crawling
        return await self._crawl_and_extract(source_config)

    async def _crawl_and_extract(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Crawl and extract content asynchronously (CONCURRENT).

        Performance: Processes multiple URLs in parallel with rate limiting.
        """
        base_url = config['base_url']
        start_url = config['docs_url']
        doc_type = config['type']

        logger.info(f"Async crawling {doc_type} docs: {start_url}")

        to_visit: List[str] = [start_url]
        visited: Set[str] = set()
        results: List[Dict[str, Any]] = []
        max_pages = Config.DOC_MAX_PAGES_PER_SOURCE

        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True
        ) as client:
            while to_visit and len(visited) < max_pages:
                # Process batch of URLs concurrently
                batch_size = min(self.max_concurrent, len(to_visit), max_pages - len(visited))
                batch = [to_visit.pop(0) for _ in range(batch_size)]

                # Filter already visited
                batch = [url for url in batch if url not in visited]

                if not batch:
                    continue

                # Mark as visited
                for url in batch:
                    visited.add(url)

                # Process batch concurrently
                batch_results = await self._process_url_batch(
                    client,
                    batch,
                    base_url,
                    doc_type
                )

                # Collect results and new URLs
                for url, doc, links in batch_results:
                    if doc:
                        results.append(doc)

                    # Add new links to queue
                    for link in links:
                        if link not in visited and link not in to_visit:
                            if len(visited) + len(to_visit) < max_pages:
                                to_visit.append(link)

                # Respectful crawling delay (async)
                if to_visit:
                    await asyncio.sleep(Config.DOC_CRAWL_DELAY)

        logger.info(f"Async crawled {len(visited)} pages, extracted {len(results)} documents for {doc_type}")
        return results

    async def _process_url_batch(
        self,
        client: httpx.AsyncClient,
        urls: List[str],
        base_url: str,
        doc_type: str
    ) -> List[tuple[str, Optional[Dict[str, Any]], List[str]]]:
        """
        Process a batch of URLs concurrently.

        Returns: List of (url, document, links) tuples
        """
        tasks = [
            self._process_single_url(client, url, base_url, doc_type)
            for url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning(f"Error processing {url}: {result}")
                valid_results.append((url, None, []))
            else:
                valid_results.append((url, *result))

        return valid_results

    async def _process_single_url(
        self,
        client: httpx.AsyncClient,
        url: str,
        base_url: str,
        doc_type: str
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Process a single URL with rate limiting.

        Returns: (document, links) tuple
        """
        async with self.semaphore:  # Rate limiting
            try:
                # Fetch HTML (ASYNC)
                html = await self._fetch_html(client, url)
                if not html:
                    return None, []

                # Extract content (CPU-bound, run in thread pool)
                text, title = await asyncio.to_thread(self._extract_content, html, url)
                cleaned = await asyncio.to_thread(self._clean_content, text)

                # Create document if content is substantial
                doc = None
                if cleaned and len(cleaned) > 300:
                    doc = {
                        'title': title or url,
                        'source_url': url,
                        'content': cleaned,
                        'doc_type': doc_type,
                        'version': 'latest',
                        'processed_at': time.time()
                    }

                # Extract links
                links = await asyncio.to_thread(self._extract_links, html, base_url)

                return doc, links

            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
                return None, []

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        """Fetch HTML content asynchronously."""
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            else:
                logger.debug(f"Non-200 status for {url}: {response.status_code}")
                return None
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML (sync, but fast)."""
        links: List[str] = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('#'):
                    continue
                abs_url = urljoin(base_url, href)
                if abs_url.startswith(base_url) and self._is_probably_doc_page(abs_url):
                    links.append(abs_url)
        except Exception:
            pass

        # Deduplicate while preserving order
        deduped = []
        seen: Set[str] = set()
        for u in links:
            if u not in seen:
                seen.add(u)
                deduped.append(u)

        return deduped

    def _is_probably_doc_page(self, url: str) -> bool:
        """Check if URL is likely a documentation page."""
        parsed = urlparse(url)
        blacklist_ext = (
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf',
            '.zip', '.tar', '.gz', '.mp4', '.mp3', '.ico'
        )
        if any(parsed.path.lower().endswith(ext) for ext in blacklist_ext):
            return False
        return True

    def _extract_content(self, html: str, url: str) -> tuple[str, str]:
        """Extract main content and title from HTML (sync)."""
        try:
            text = trafilatura.extract(filecontent=html, url=url) or ""
        except Exception:
            text = ""

        title = ""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = (soup.title.string or '').strip() if soup.title else ''
            if not text:
                # Fallback to raw text
                body = soup.body or soup
                text = body.get_text(separator='\n', strip=True)
        except Exception:
            pass

        return text or "", title or ""

    def _clean_content(self, text: str) -> str:
        """Clean extracted content (sync)."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove navigation/footer patterns
        text = re.sub(r'(?i)(skip to content|table of contents|next page|previous page)', '', text)

        return text.strip()

    # Sample data processors (async wrappers for consistency)

    async def _process_react_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process React documentation using sample data."""
        logger.info("Processing React documentation using sample data")
        return await asyncio.to_thread(self._process_sample_docs, SAMPLE_REACT_DOCS, 'react', 'latest')

    async def _process_python_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process Python documentation using sample data."""
        logger.info("Processing Python documentation using sample data")
        return await asyncio.to_thread(self._process_sample_docs, SAMPLE_PYTHON_DOCS, 'python', '3.11')

    async def _process_fastapi_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process FastAPI documentation using sample data."""
        logger.info("Processing FastAPI documentation using sample data")
        return await asyncio.to_thread(self._process_sample_docs, SAMPLE_FASTAPI_DOCS, 'fastapi', 'latest')

    async def _process_docker_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process Docker documentation using sample data."""
        logger.info("Processing Docker documentation using sample data")
        return await asyncio.to_thread(self._process_sample_docs, SAMPLE_DOCKER_DOCS, 'docker', 'latest')

    async def _process_aws_lambda_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process AWS Lambda documentation using sample data."""
        logger.info("Processing AWS Lambda documentation using sample data")
        return await asyncio.to_thread(self._process_sample_docs, SAMPLE_AWS_LAMBDA_DOCS, 'aws_lambda', 'latest')

    def _process_sample_docs(
        self,
        sample_docs: List[Dict[str, Any]],
        doc_type: str,
        version: str
    ) -> List[Dict[str, Any]]:
        """Process sample documentation data."""
        documents = []
        for doc_data in sample_docs:
            processed_doc = {
                'title': doc_data['title'],
                'source_url': doc_data['url'],
                'content': self._clean_content(doc_data['content']),
                'doc_type': doc_type,
                'version': version,
                'processed_at': time.time()
            }
            documents.append(processed_doc)
        return documents
