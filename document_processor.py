import logging
import time
import re
from typing import List, Dict, Any, Optional, Set
import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse
from config import Config
from data.sample_docs import SAMPLE_REACT_DOCS, SAMPLE_PYTHON_DOCS, SAMPLE_FASTAPI_DOCS

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocRag Documentation Crawler/1.0 (+https://example.com)'
        })
        self.timeout = Config.DOC_CRAWL_TIMEOUT
    
    def process_documentation_sources(self) -> List[Dict[str, Any]]:
        """Process all configured documentation sources"""
        all_documents = []
        
        for source_name, source_config in Config.DOC_SOURCES.items():
            logger.info(f"Processing {source_name} documentation...")
            try:
                docs = self.process_source(source_config)
                all_documents.extend(docs)
                logger.info(f"Processed {len(docs)} documents from {source_name}")
            except Exception as e:
                logger.error(f"Error processing {source_name}: {e}")
        
        return all_documents
    
    def process_source(self, source_config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process a single documentation source"""
        if Config.DOC_USE_SAMPLE:
            # Maintain current behavior for samples
            if source_config['type'] == 'react':
                return self._process_react_docs(source_config)
            if source_config['type'] == 'python':
                return self._process_python_docs(source_config)
            if source_config['type'] == 'fastapi':
                return self._process_fastapi_docs(source_config)
            return []

        # Real crawling path
        return self._crawl_and_extract(source_config)

    # ------------------ Real crawling helpers ------------------
    def _crawl_and_extract(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Crawl starting from docs_url within base_url scope and extract cleaned text."""
        base_url = config['base_url']
        start_url = config['docs_url']
        doc_type = config['type']

        logger.info(f"Crawling {doc_type} docs: {start_url}")

        to_visit: List[str] = [start_url]
        visited: Set[str] = set()
        results: List[Dict[str, Any]] = []
        max_pages = Config.DOC_MAX_PAGES_PER_SOURCE

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                html = self._fetch_html(url)
                if not html:
                    continue

                text, title = self._extract_content(html, url)
                cleaned = self._clean_content(text)
                if cleaned and len(cleaned) > 300:
                    results.append({
                        'title': title or url,
                        'source_url': url,
                        'content': cleaned,
                        'doc_type': doc_type,
                        'version': 'latest',
                        'processed_at': time.time()
                    })

                # Enfileirar links internos
                for link in self._extract_links(html, base_url):
                    if link not in visited and link not in to_visit and len(visited) + len(to_visit) < max_pages:
                        to_visit.append(link)

                # Respeitar intervalo entre requisições
                time.sleep(max(0.0, Config.DOC_CRAWL_DELAY))

            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
                continue

        logger.info(f"Crawled {len(visited)} pages, extracted {len(results)} documents for {doc_type}")
        return results

    def _fetch_html(self, url: str) -> Optional[str]:
        resp = self.session.get(url, timeout=self.timeout)
        if not resp.ok:
            return None
        return resp.text

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        links: List[str] = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('#'):
                    continue
                abs_url = urljoin(base_url, href)
                # restringe ao host/base
                if abs_url.startswith(base_url) and self._is_probably_doc_page(abs_url):
                    links.append(abs_url)
        except Exception:
            pass
        # dedup mantendo ordem
        deduped = []
        seen: Set[str] = set()
        for u in links:
            if u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped

    def _is_probably_doc_page(self, url: str) -> bool:
        parsed = urlparse(url)
        # heurística simples: evitar assets binários
        blacklist_ext = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mp3', '.ico')
        if any(parsed.path.lower().endswith(ext) for ext in blacklist_ext):
            return False
        return True

    def _extract_content(self, html: str, url: str) -> tuple[str, str]:
        """Extract main textual content and title from HTML."""
        try:
            text = trafilatura.extract(filecontent=html, url=url) or ""
        except Exception:
            text = ""
        title = ""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = (soup.title.string or '').strip() if soup.title else ''
            if not text:
                # fallback para texto bruto
                body = soup.body or soup
                text = body.get_text(separator='\n', strip=True)
        except Exception:
            pass
        return text or "", title or ""
    
    def _process_react_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process React documentation"""
        logger.info("Processing React documentation using sample data")
        
        documents = []
        for doc_data in SAMPLE_REACT_DOCS:
            processed_doc = {
                'title': doc_data['title'],
                'source_url': doc_data['url'],
                'content': self._clean_content(doc_data['content']),
                'doc_type': 'react',
                'version': 'latest',
                'processed_at': time.time()
            }
            documents.append(processed_doc)
        
        return documents
    
    def _process_python_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process Python documentation"""
        logger.info("Processing Python documentation using sample data")
        
        documents = []
        for doc_data in SAMPLE_PYTHON_DOCS:
            processed_doc = {
                'title': doc_data['title'],
                'source_url': doc_data['url'],
                'content': self._clean_content(doc_data['content']),
                'doc_type': 'python',
                'version': '3.11',
                'processed_at': time.time()
            }
            documents.append(processed_doc)
        
        return documents
    
    def _process_fastapi_docs(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process FastAPI documentation"""
        logger.info("Processing FastAPI documentation using sample data")
        
        documents = []
        for doc_data in SAMPLE_FASTAPI_DOCS:
            processed_doc = {
                'title': doc_data['title'],
                'source_url': doc_data['url'],
                'content': self._clean_content(doc_data['content']),
                'doc_type': 'fastapi',
                'version': '0.104+',
                'processed_at': time.time()
            }
            documents.append(processed_doc)
        
        return documents
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        # Remove HTML entities and tags if any
        content = re.sub(r'&[a-zA-Z0-9]+;', '', content)
        content = re.sub(r'<[^>]+>', '', content)
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip()
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape content from a URL using trafilatura
        This method can be used for live scraping when needed
        """
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Download the webpage
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"Failed to download content from {url}")
                return None
            
            # Extract text content
            text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not text_content:
                logger.warning(f"No text content extracted from {url}")
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata else "Unknown Title"
            
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'metadata': metadata.__dict__ if metadata else {},
                'scraped_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return None
    
    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from content"""
        code_blocks = []
        
        # Find code blocks with language specification
        code_pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        
        for match in matches:
            language = match[0] if match[0] else 'text'
            code = match[1].strip()
            
            if code:  # Only add non-empty code blocks
                code_blocks.append({
                    'language': language,
                    'code': code
                })
        
        return code_blocks
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            response = self.session.head(url, timeout=10)
            return response.status_code < 400
            
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            return False
    
    def get_url_links(self, base_url: str, max_depth: int = 2) -> List[str]:
        """
        Get all documentation links from a base URL
        This can be used for comprehensive documentation scraping
        """
        try:
            response = self.session.get(base_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {base_url}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                # Filter for documentation links
                if self._is_documentation_link(full_url, base_url):
                    links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting links from {base_url}: {e}")
            return []
    
    def _is_documentation_link(self, url: str, base_url: str) -> bool:
        """Check if a URL is likely a documentation page"""
        parsed = urlparse(url)
        base_parsed = urlparse(base_url)
        
        # Must be from the same domain
        if parsed.netloc != base_parsed.netloc:
            return False
        
        # Skip external links, images, downloads
        skip_extensions = ['.pdf', '.zip', '.tar', '.gz', '.exe', '.dmg', '.png', '.jpg', '.gif']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip fragments and queries for now
        if '#' in url or '?' in url:
            return False
        
        # Look for documentation patterns
        doc_patterns = ['/docs/', '/doc/', '/guide/', '/tutorial/', '/learn/', '/reference/']
        return any(pattern in url.lower() for pattern in doc_patterns)
    
    def batch_process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs in batch"""
        documents = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            
            scraped_data = self.scrape_url(url)
            if scraped_data:
                doc = {
                    'title': scraped_data['title'],
                    'source_url': scraped_data['url'],
                    'content': self._clean_content(scraped_data['content']),
                    'doc_type': self._detect_doc_type(url),
                    'version': 'latest',
                    'processed_at': time.time(),
                    'code_blocks': self.extract_code_blocks(scraped_data['content'])
                }
                documents.append(doc)
            
            # Add delay between requests to be respectful
            time.sleep(1)
        
        return documents
    
    def _detect_doc_type(self, url: str) -> str:
        """Detect documentation type from URL"""
        url_lower = url.lower()
        
        if 'react' in url_lower:
            return 'react'
        elif 'python.org' in url_lower:
            return 'python'
        elif 'fastapi' in url_lower:
            return 'fastapi'
        else:
            return 'unknown'
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        try:
            stats = {
                'sources_configured': len(Config.DOC_SOURCES),
                'supported_types': list(Config.DOC_SOURCES.keys()),
                'chunk_size': Config.CHUNK_SIZE,
                'chunk_overlap': Config.CHUNK_OVERLAP,
                'last_processed': time.time()
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
