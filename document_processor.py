import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Set
import requests
from bs4 import BeautifulSoup
import trafilatura
import yaml
from urllib.parse import urljoin, urlparse
from config import Config
from data.sample_docs import SAMPLE_REACT_DOCS, SAMPLE_PYTHON_DOCS, SAMPLE_FASTAPI_DOCS
from api_discovery import APIDiscoveryEngine, APIDocSource

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocRag Documentation Crawler/1.0 (+https://example.com)'
        })
        self.timeout = Config.DOC_CRAWL_TIMEOUT
        self.api_discovery = APIDiscoveryEngine()
    
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


class APIDocumentProcessor(DocumentProcessor):
    """Extended processor for different API documentation formats."""
    
    def process_api_documentation(self, base_url: str) -> List[Dict[str, Any]]:
        """
        Process API documentation from a base URL using discovery engine.
        
        Args:
            base_url: Base URL to discover API documentation from
            
        Returns:
            List of processed API documents
        """
        logger.info(f"Processing API documentation from: {base_url}")
        
        # Discover API documentation sources
        api_sources = self.api_discovery.discover_api_documentation(base_url)
        
        all_documents = []
        
        for source in api_sources:
            try:
                logger.info(f"Processing {source.doc_type} source: {source.url}")
                
                if source.doc_type == 'openapi':
                    documents = self.process_openapi_spec(source.url, source.format)
                elif source.doc_type == 'html':
                    documents = self.process_html_api_docs(source.url)
                else:
                    logger.warning(f"Unsupported API doc type: {source.doc_type}")
                    continue
                
                # Add source metadata
                for doc in documents:
                    doc['api_source'] = source.url
                    doc['discovery_confidence'] = source.confidence
                
                all_documents.extend(documents)
                logger.info(f"Processed {len(documents)} documents from {source.url}")
                
                # Rate limiting between sources
                time.sleep(Config.DOC_CRAWL_DELAY)
                
            except Exception as e:
                logger.error(f"Error processing API source {source.url}: {e}")
                continue
        
        logger.info(f"Total API documents processed: {len(all_documents)}")
        return all_documents
    
    def process_openapi_spec(self, spec_url: str, format: str = 'json') -> List[Dict[str, Any]]:
        """
        Process OpenAPI/Swagger specification.
        
        Args:
            spec_url: URL of the OpenAPI spec
            format: Format of the spec ('json' or 'yaml')
            
        Returns:
            List of processed endpoint documents
        """
        try:
            logger.info(f"Processing OpenAPI spec: {spec_url}")
            
            response = self.session.get(spec_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse specification
            if format == 'json' or spec_url.endswith('.json'):
                spec_data = response.json()
            else:  # yaml
                spec_data = yaml.safe_load(response.text)
            
            return self._process_openapi_data(spec_data, spec_url)
            
        except Exception as e:
            logger.error(f"Error processing OpenAPI spec {spec_url}: {e}")
            return []
    
    def _process_openapi_data(self, spec_data: Dict[str, Any], source_url: str) -> List[Dict[str, Any]]:
        """Process OpenAPI specification data into documents."""
        documents = []
        
        # Extract API info
        info = spec_data.get('info', {})
        base_path = spec_data.get('basePath', '')
        servers = spec_data.get('servers', [])
        
        # Get base URL from servers or construct from source
        base_url = ""
        if servers and isinstance(servers[0], dict):
            base_url = servers[0].get('url', '')
        
        # Process each endpoint
        paths = spec_data.get('paths', {})
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
                
            for method, endpoint_data in methods.items():
                if method.startswith('x-'):  # Skip extensions
                    continue
                    
                try:
                    doc = self._create_endpoint_document(
                        path, method, endpoint_data, info, base_url, source_url
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error processing endpoint {method.upper()} {path}: {e}")
                    continue
        
        # Create API overview document
        overview_doc = self._create_api_overview_document(spec_data, source_url)
        if overview_doc:
            documents.insert(0, overview_doc)
        
        return documents
    
    def _create_endpoint_document(self, path: str, method: str, endpoint_data: Dict[str, Any], 
                                 api_info: Dict[str, Any], base_url: str, source_url: str) -> Dict[str, Any]:
        """Create a document for a single API endpoint."""
        
        method_upper = method.upper()
        
        # Build endpoint content
        content_parts = []
        
        # Basic endpoint info
        content_parts.append(f"## {method_upper} {path}")
        
        # Summary and description
        if 'summary' in endpoint_data:
            content_parts.append(f"\n**Summary:** {endpoint_data['summary']}")
        
        if 'description' in endpoint_data:
            content_parts.append(f"\n**Description:** {endpoint_data['description']}")
        
        # Parameters
        parameters = endpoint_data.get('parameters', [])
        if parameters:
            content_parts.append("\n### Parameters")
            for param in parameters:
                param_name = param.get('name', 'unknown')
                param_type = param.get('type', param.get('schema', {}).get('type', 'string'))
                param_in = param.get('in', 'query')
                param_required = param.get('required', False)
                param_desc = param.get('description', '')
                
                required_text = " (required)" if param_required else " (optional)"
                content_parts.append(
                    f"- **{param_name}** ({param_in}, {param_type}){required_text}: {param_desc}"
                )
        
        # Request body
        request_body = endpoint_data.get('requestBody', {})
        if request_body:
            content_parts.append("\n### Request Body")
            if 'description' in request_body:
                content_parts.append(request_body['description'])
            
            # Add content type info
            content = request_body.get('content', {})
            for content_type, content_data in content.items():
                content_parts.append(f"\n**Content-Type:** {content_type}")
                
                # Add schema info if available
                schema = content_data.get('schema', {})
                if schema:
                    content_parts.append(f"**Schema:** {json.dumps(schema, indent=2)}")
        
        # Responses
        responses = endpoint_data.get('responses', {})
        if responses:
            content_parts.append("\n### Responses")
            for status_code, response_data in responses.items():
                description = response_data.get('description', '')
                content_parts.append(f"- **{status_code}**: {description}")
                
                # Add response schema if available
                content = response_data.get('content', {})
                for content_type, content_data in content.items():
                    schema = content_data.get('schema', {})
                    if schema:
                        content_parts.append(f"  - Content-Type: {content_type}")
                        content_parts.append(f"  - Schema: {json.dumps(schema, indent=2)}")
        
        # Security requirements
        security = endpoint_data.get('security', [])
        if security:
            content_parts.append("\n### Authentication")
            for security_req in security:
                for auth_name, scopes in security_req.items():
                    content_parts.append(f"- **{auth_name}**")
                    if scopes:
                        content_parts.append(f"  - Scopes: {', '.join(scopes)}")
        
        # Tags
        tags = endpoint_data.get('tags', [])
        if tags:
            content_parts.append(f"\n**Tags:** {', '.join(tags)}")
        
        full_content = "\n".join(content_parts)
        
        return {
            'title': f"{method_upper} {path}",
            'source_url': source_url,
            'content': self._clean_content(full_content),
            'doc_type': 'api_endpoint',
            'api_method': method_upper,
            'api_path': path,
            'api_tags': tags,
            'version': api_info.get('version', '1.0'),
            'processed_at': time.time(),
            'endpoint_data': {
                'method': method_upper,
                'path': path,
                'summary': endpoint_data.get('summary', ''),
                'description': endpoint_data.get('description', ''),
                'parameters': parameters,
                'responses': list(responses.keys()) if responses else [],
                'tags': tags,
                'security': security
            }
        }
    
    def _create_api_overview_document(self, spec_data: Dict[str, Any], source_url: str) -> Optional[Dict[str, Any]]:
        """Create an overview document for the entire API."""
        try:
            info = spec_data.get('info', {})
            
            content_parts = []
            content_parts.append(f"# {info.get('title', 'API Documentation')}")
            
            if 'description' in info:
                content_parts.append(f"\n{info['description']}")
            
            if 'version' in info:
                content_parts.append(f"\n**Version:** {info['version']}")
            
            # Contact info
            contact = info.get('contact', {})
            if contact:
                content_parts.append("\n## Contact Information")
                if 'name' in contact:
                    content_parts.append(f"**Name:** {contact['name']}")
                if 'email' in contact:
                    content_parts.append(f"**Email:** {contact['email']}")
                if 'url' in contact:
                    content_parts.append(f"**URL:** {contact['url']}")
            
            # License info
            license_info = info.get('license', {})
            if license_info:
                content_parts.append("\n## License")
                content_parts.append(f"**Name:** {license_info.get('name', 'Unknown')}")
                if 'url' in license_info:
                    content_parts.append(f"**URL:** {license_info['url']}")
            
            # Servers
            servers = spec_data.get('servers', [])
            if servers:
                content_parts.append("\n## Servers")
                for server in servers:
                    server_url = server.get('url', '')
                    server_desc = server.get('description', '')
                    content_parts.append(f"- **{server_url}**: {server_desc}")
            
            # Tags (categories)
            tags = spec_data.get('tags', [])
            if tags:
                content_parts.append("\n## API Categories")
                for tag in tags:
                    tag_name = tag.get('name', '')
                    tag_desc = tag.get('description', '')
                    content_parts.append(f"- **{tag_name}**: {tag_desc}")
            
            full_content = "\n".join(content_parts)
            
            return {
                'title': f"API Overview: {info.get('title', 'Unknown API')}",
                'source_url': source_url,
                'content': self._clean_content(full_content),
                'doc_type': 'api_overview',
                'version': info.get('version', '1.0'),
                'processed_at': time.time(),
                'api_info': info
            }
            
        except Exception as e:
            logger.error(f"Error creating API overview: {e}")
            return None
    
    def process_html_api_docs(self, html_url: str) -> List[Dict[str, Any]]:
        """
        Process HTML API documentation pages.
        
        Args:
            html_url: URL of the HTML documentation
            
        Returns:
            List of processed documents
        """
        try:
            logger.info(f"Processing HTML API docs: {html_url}")
            
            response = self.session.get(html_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Extract content using trafilatura (existing method)
            text_content = trafilatura.extract(response.text, include_comments=False, include_tables=True)
            
            if not text_content:
                logger.warning(f"No content extracted from {html_url}")
                return []
            
            # Extract additional API-specific information
            soup = BeautifulSoup(response.text, 'html.parser')
            title = self._extract_page_title_from_soup(soup) or "API Documentation"
            
            # Look for API endpoints in the content
            endpoints = self._extract_endpoints_from_html(text_content, soup)
            
            # Create main document
            doc = {
                'title': title,
                'source_url': html_url,
                'content': self._clean_content(text_content),
                'doc_type': 'api_html',
                'version': 'latest',
                'processed_at': time.time(),
                'code_blocks': self.extract_code_blocks(text_content),
                'endpoints': endpoints
            }
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Error processing HTML API docs {html_url}: {e}")
            return []
    
    def _extract_page_title_from_soup(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title from BeautifulSoup object."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 as fallback
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return None
    
    def _extract_endpoints_from_html(self, text_content: str, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract API endpoints from HTML content."""
        endpoints = []
        
        # Pattern to match HTTP methods and paths
        endpoint_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+([/\w\-\{\}]+)'
        matches = re.findall(endpoint_pattern, text_content, re.IGNORECASE)
        
        for match in matches:
            method, path = match
            endpoints.append({
                'method': method.upper(),
                'path': path.strip()
            })
        
        return endpoints
    
    def process_postman_collection(self, collection_data: Dict[str, Any], source_url: str = '') -> List[Dict[str, Any]]:
        """
        Process Postman Collection data.
        
        Args:
            collection_data: Postman collection JSON data
            source_url: Source URL of the collection (optional)
            
        Returns:
            List of processed request documents
        """
        try:
            logger.info("Processing Postman collection")
            
            documents = []
            collection_info = collection_data.get('info', {})
            collection_name = collection_info.get('name', 'Postman Collection')
            
            # Process collection items
            items = collection_data.get('item', [])
            for item in items:
                if 'request' in item:
                    doc = self._create_postman_document(item, collection_name, source_url)
                    if doc:
                        documents.append(doc)
                elif 'item' in item:  # Nested folders
                    nested_docs = self._process_postman_folder(item, collection_name, source_url)
                    documents.extend(nested_docs)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Postman collection: {e}")
            return []
    
    def _create_postman_document(self, item: Dict[str, Any], collection_name: str, source_url: str) -> Optional[Dict[str, Any]]:
        """Create document from Postman request item."""
        try:
            request_data = item.get('request', {})
            
            # Extract request details
            method = request_data.get('method', 'GET')
            url = request_data.get('url', {})
            
            if isinstance(url, str):
                url_string = url
            else:
                url_string = url.get('raw', '') if isinstance(url, dict) else ''
            
            item_name = item.get('name', f"{method} Request")
            description = item.get('description', '')
            
            # Build content
            content_parts = []
            content_parts.append(f"# {item_name}")
            
            if description:
                content_parts.append(f"\n{description}")
            
            content_parts.append(f"\n**Method:** {method}")
            content_parts.append(f"**URL:** {url_string}")
            
            # Headers
            headers = request_data.get('header', [])
            if headers:
                content_parts.append("\n## Headers")
                for header in headers:
                    if isinstance(header, dict):
                        key = header.get('key', '')
                        value = header.get('value', '')
                        content_parts.append(f"- {key}: {value}")
            
            # Body
            body = request_data.get('body', {})
            if body:
                content_parts.append("\n## Request Body")
                mode = body.get('mode', '')
                if mode == 'raw':
                    raw_body = body.get('raw', '')
                    content_parts.append(f"```json\n{raw_body}\n```")
                elif mode == 'formdata':
                    formdata = body.get('formdata', [])
                    for field in formdata:
                        if isinstance(field, dict):
                            key = field.get('key', '')
                            value = field.get('value', '')
                            content_parts.append(f"- {key}: {value}")
            
            full_content = "\n".join(content_parts)
            
            return {
                'title': item_name,
                'source_url': source_url,
                'content': self._clean_content(full_content),
                'doc_type': 'postman_request',
                'api_method': method,
                'api_url': url_string,
                'collection': collection_name,
                'version': 'latest',
                'processed_at': time.time(),
                'request_data': {
                    'method': method,
                    'url': url_string,
                    'headers': headers,
                    'body': body
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating Postman document: {e}")
            return None
    
    def _process_postman_folder(self, folder: Dict[str, Any], collection_name: str, source_url: str) -> List[Dict[str, Any]]:
        """Process Postman folder (nested items)."""
        documents = []
        
        items = folder.get('item', [])
        for item in items:
            if 'request' in item:
                doc = self._create_postman_document(item, collection_name, source_url)
                if doc:
                    documents.append(doc)
            elif 'item' in item:  # Further nesting
                nested_docs = self._process_postman_folder(item, collection_name, source_url)
                documents.extend(nested_docs)
        
        return documents
