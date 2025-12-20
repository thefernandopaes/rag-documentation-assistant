"""
API Discovery Engine for automatically finding API documentation sources.

This module discovers various types of API documentation including:
- OpenAPI/Swagger specifications (JSON/YAML)
- Common API documentation patterns
- Sitemap-based discovery
"""

import logging
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class APIDocSource:
    """Represents a discovered API documentation source."""
    url: str
    doc_type: str  # 'openapi', 'html', 'postman', 'graphql'
    format: str   # 'json', 'yaml', 'html'
    title: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    confidence: float = 0.8  # Confidence in discovery accuracy


class APIDiscoveryEngine:
    """Discovers automatically API documentation sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocRag-APIBot/1.0 (API Documentation Discovery)'
        })
        
        # Common OpenAPI/Swagger paths
        self.openapi_paths = [
            '/swagger.json', '/openapi.json', '/api-docs.json',
            '/swagger.yaml', '/openapi.yaml', '/docs/openapi.yaml',
            '/api/swagger.json', '/api/openapi.json',
            '/docs/swagger.json', '/docs/openapi.json',
            '/v1/swagger.json', '/v2/swagger.json', '/v3/swagger.json',
            '/swagger-ui/swagger.json', '/api-docs'
        ]
        
        # Common API documentation URL patterns
        self.doc_patterns = [
            '/docs/api/', '/api/reference/', '/developers/',
            '/api-docs/', '/documentation/', '/reference/',
            '/docs/reference/', '/api/docs/', '/developer/',
            '/docs/', '/guides/api/', '/api-guide/',
            '/rest-api/', '/graphql/', '/webhooks/'
        ]
        
    def discover_api_documentation(self, base_url: str) -> List[APIDocSource]:
        """
        Discover API documentation sources from a base URL.
        
        Args:
            base_url: The base URL to start discovery from
            
        Returns:
            List of discovered API documentation sources
        """
        logger.info(f"Starting API documentation discovery for: {base_url}")
        
        # Normalize base URL
        if not base_url.startswith(('http://', 'https://')):
            base_url = f'https://{base_url}'
        
        base_url = base_url.rstrip('/')
        
        sources = []
        
        try:
            # 1. Discover OpenAPI/Swagger specifications
            openapi_sources = self._discover_openapi_specs(base_url)
            sources.extend(openapi_sources)
            logger.info(f"Found {len(openapi_sources)} OpenAPI sources")
            
            # 2. Discover common documentation patterns
            doc_sources = self._discover_common_doc_patterns(base_url)
            sources.extend(doc_sources)
            logger.info(f"Found {len(doc_sources)} documentation pattern sources")
            
            # 3. Discover via sitemap
            sitemap_sources = self._discover_via_sitemap(base_url)
            sources.extend(sitemap_sources)
            logger.info(f"Found {len(sitemap_sources)} sitemap sources")
            
            # 4. Discover via robots.txt
            robots_sources = self._discover_via_robots(base_url)
            sources.extend(robots_sources)
            logger.info(f"Found {len(robots_sources)} robots.txt sources")
            
        except Exception as e:
            logger.error(f"Error during API discovery for {base_url}: {e}")
        
        # Remove duplicates and sort by confidence
        unique_sources = self._deduplicate_sources(sources)
        unique_sources.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Total unique API documentation sources found: {len(unique_sources)}")
        
        return unique_sources
    
    def _discover_openapi_specs(self, base_url: str) -> List[APIDocSource]:
        """Discover OpenAPI/Swagger specifications."""
        sources = []
        
        for path in self.openapi_paths:
            full_url = urljoin(base_url, path)
            
            try:
                response = self._make_request(full_url)
                if response and response.status_code == 200:
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'json' in content_type:
                        spec_data = response.json()
                        if self._is_valid_openapi_spec(spec_data):
                            source = self._create_openapi_source(full_url, spec_data, 'json')
                            sources.append(source)
                            
                    elif 'yaml' in content_type or full_url.endswith(('.yaml', '.yml')):
                        try:
                            spec_data = yaml.safe_load(response.text)
                            if self._is_valid_openapi_spec(spec_data):
                                source = self._create_openapi_source(full_url, spec_data, 'yaml')
                                sources.append(source)
                        except yaml.YAMLError:
                            continue
                            
            except Exception as e:
                logger.debug(f"Failed to check OpenAPI spec at {full_url}: {e}")
                continue
                
            # Rate limiting
            time.sleep(Config.DOC_CRAWL_DELAY)
        
        return sources
    
    def _discover_common_doc_patterns(self, base_url: str) -> List[APIDocSource]:
        """Discover documentation following common patterns."""
        sources = []
        
        for pattern in self.doc_patterns:
            full_url = urljoin(base_url, pattern)
            
            try:
                response = self._make_request(full_url)
                if response and response.status_code == 200:
                    
                    # Check if this looks like API documentation
                    if self._is_api_documentation_page(response.text):
                        source = APIDocSource(
                            url=full_url,
                            doc_type='html',
                            format='html',
                            title=self._extract_page_title(response.text),
                            confidence=0.7
                        )
                        sources.append(source)
                        
            except Exception as e:
                logger.debug(f"Failed to check documentation pattern at {full_url}: {e}")
                continue
                
            # Rate limiting
            time.sleep(Config.DOC_CRAWL_DELAY)
        
        return sources
    
    def _discover_via_sitemap(self, base_url: str) -> List[APIDocSource]:
        """Discover API documentation via sitemap.xml."""
        sources = []
        
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap-docs.xml'),
            urljoin(base_url, '/docs/sitemap.xml'),
            urljoin(base_url, '/api/sitemap.xml')
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                response = self._make_request(sitemap_url)
                if response and response.status_code == 200:
                    urls = self._parse_sitemap(response.text)
                    
                    for url in urls:
                        # Filter URLs that look like API documentation
                        if self._url_looks_like_api_docs(url):
                            source = APIDocSource(
                                url=url,
                                doc_type='html',
                                format='html',
                                confidence=0.6
                            )
                            sources.append(source)
                            
            except Exception as e:
                logger.debug(f"Failed to parse sitemap at {sitemap_url}: {e}")
                continue
                
            # Rate limiting
            time.sleep(Config.DOC_CRAWL_DELAY)
        
        return sources
    
    def _discover_via_robots(self, base_url: str) -> List[APIDocSource]:
        """Discover API documentation hints from robots.txt."""
        sources = []
        
        robots_url = urljoin(base_url, '/robots.txt')
        
        try:
            response = self._make_request(robots_url)
            if response and response.status_code == 200:
                
                # Look for sitemap references
                for line in response.text.split('\n'):
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        if sitemap_url:
                            sitemap_sources = self._discover_via_sitemap(sitemap_url)
                            sources.extend(sitemap_sources)
                            
        except Exception as e:
            logger.debug(f"Failed to check robots.txt at {robots_url}: {e}")
        
        return sources
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with timeout and error handling."""
        try:
            response = self.session.get(
                url,
                timeout=Config.DOC_CRAWL_TIMEOUT,
                allow_redirects=True
            )
            return response
        except requests.RequestException as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None
    
    def _is_valid_openapi_spec(self, data: Dict[str, Any]) -> bool:
        """Check if data is a valid OpenAPI/Swagger specification."""
        if not isinstance(data, dict):
            return False
        
        # Check for OpenAPI 3.x
        if 'openapi' in data and isinstance(data['openapi'], str):
            return data['openapi'].startswith('3.')
        
        # Check for Swagger 2.x
        if 'swagger' in data and isinstance(data['swagger'], str):
            return data['swagger'].startswith('2.')
        
        # Check for required fields
        return 'paths' in data and isinstance(data['paths'], dict)
    
    def _create_openapi_source(self, url: str, spec_data: Dict[str, Any], format: str) -> APIDocSource:
        """Create APIDocSource from OpenAPI specification data."""
        info = spec_data.get('info', {})
        
        return APIDocSource(
            url=url,
            doc_type='openapi',
            format=format,
            title=info.get('title', 'API Documentation'),
            description=info.get('description'),
            version=info.get('version'),
            confidence=0.9
        )
    
    def _is_api_documentation_page(self, html_content: str) -> bool:
        """Check if HTML content looks like API documentation."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for API-related keywords in title and headers
            api_keywords = [
                'api', 'rest', 'graphql', 'endpoint', 'webhook',
                'authentication', 'authorization', 'curl', 'json',
                'swagger', 'openapi', 'postman'
            ]
            
            # Check title
            title = soup.find('title')
            if title:
                title_text = title.get_text().lower()
                if any(keyword in title_text for keyword in api_keywords):
                    return True
            
            # Check headers
            headers = soup.find_all(['h1', 'h2', 'h3'])
            for header in headers:
                header_text = header.get_text().lower()
                if any(keyword in header_text for keyword in api_keywords):
                    return True
            
            # Check for code blocks (common in API docs)
            code_blocks = soup.find_all(['pre', 'code'])
            if len(code_blocks) > 3:  # Likely has examples
                return True
                
            # Check for specific API documentation indicators
            api_indicators = soup.find_all(text=re.compile(r'(GET|POST|PUT|DELETE|PATCH)\s+/', re.IGNORECASE))
            if api_indicators:
                return True
            
        except Exception:
            pass
        
        return False
    
    def _extract_page_title(self, html_content: str) -> Optional[str]:
        """Extract page title from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            if title:
                return title.get_text().strip()
        except Exception:
            pass
        return None
    
    def _parse_sitemap(self, sitemap_content: str) -> List[str]:
        """Parse sitemap XML and extract URLs."""
        urls = []
        
        try:
            soup = BeautifulSoup(sitemap_content, 'xml')
            
            # Handle sitemap index
            sitemap_tags = soup.find_all('sitemap')
            for sitemap_tag in sitemap_tags:
                loc = sitemap_tag.find('loc')
                if loc:
                    # Recursively parse sub-sitemaps
                    sub_urls = self._discover_via_sitemap(loc.get_text())
                    urls.extend([source.url for source in sub_urls])
            
            # Handle URL set
            url_tags = soup.find_all('url')
            for url_tag in url_tags:
                loc = url_tag.find('loc')
                if loc:
                    urls.append(loc.get_text())
                    
        except Exception as e:
            logger.debug(f"Failed to parse sitemap: {e}")
        
        return urls
    
    def _url_looks_like_api_docs(self, url: str) -> bool:
        """Check if URL looks like API documentation."""
        url_lower = url.lower()
        
        api_indicators = [
            '/api/', '/docs/', '/reference/', '/developer/',
            '/rest/', '/graphql/', '/webhook/', '/guide/',
            'swagger', 'openapi', 'postman'
        ]
        
        return any(indicator in url_lower for indicator in api_indicators)
    
    def _deduplicate_sources(self, sources: List[APIDocSource]) -> List[APIDocSource]:
        """Remove duplicate sources based on URL."""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        return unique_sources
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about the discovery process."""
        return {
            'openapi_paths_checked': len(self.openapi_paths),
            'doc_patterns_checked': len(self.doc_patterns),
            'discovery_methods': ['openapi_specs', 'doc_patterns', 'sitemap', 'robots'],
            'supported_formats': ['json', 'yaml', 'html'],
            'supported_doc_types': ['openapi', 'html', 'postman', 'graphql']
        }