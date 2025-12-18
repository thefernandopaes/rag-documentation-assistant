"""
Async RAG Engine - Performance-optimized with AsyncOpenAI

Critical performance improvements:
- AsyncOpenAI client: Non-blocking API calls (2-4s savings per query)
- Async embedding generation: Concurrent processing
- Async LLM response: Non-blocking chat completions
- ChromaDB wrapped in asyncio.to_thread() (no native async support)

Expected performance: 2-4x faster than sync version
"""

import json
import logging
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from cache_manager_async import AsyncCacheManager
from code_generator import CodeExampleGenerator
from hashlib import sha256

logger = logging.getLogger(__name__)


class AsyncRAGEngine:
    """Async RAG Engine with AsyncOpenAI for high-performance document retrieval."""

    def __init__(self):
        """Initialize the async RAG engine"""
        Config.validate_config()

        # Initialize AsyncOpenAI client (NON-BLOCKING)
        self.openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        logger.info("AsyncOpenAI client initialized")

        # Initialize ChromaDB (sync, will wrap calls in to_thread)
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Initialize async cache
        self.cache = AsyncCacheManager()

        # Initialize code generator (sync, but fast)
        self.code_generator = CodeExampleGenerator()

        logger.info("Async RAG Engine initialized successfully")

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store (async)"""
        try:
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc['content'])

                # Process chunks concurrently (MAJOR SPEEDUP)
                chunk_tasks = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue
                    chunk_tasks.append(self._process_chunk_async(doc, chunk, i))

                # Wait for all chunks to be processed
                processed_chunks = await asyncio.gather(*chunk_tasks)

                # Collect results
                for result in processed_chunks:
                    if result:
                        all_chunks.append(result['chunk'])
                        all_embeddings.append(result['embedding'])
                        all_metadatas.append(result['metadata'])
                        all_ids.append(result['id'])

            # Add to ChromaDB (wrapped in to_thread)
            if all_chunks:
                await asyncio.to_thread(
                    self.collection.add,
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                logger.info(f"Added {len(all_chunks)} chunks to vector store (async)")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def _process_chunk_async(self, doc: Dict[str, Any], chunk: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single chunk asynchronously"""
        try:
            # Generate embedding (ASYNC - NON-BLOCKING)
            embedding = await self._get_embedding(chunk)

            # Create metadata
            metadata = {
                'source_url': doc.get('source_url', ''),
                'title': doc.get('title', ''),
                'doc_type': doc.get('doc_type', ''),
                'chunk_index': index,
                'version': doc.get('version', ''),
                'content_hash': sha256(chunk.encode('utf-8')).hexdigest(),
            }

            # Create unique ID
            doc_id = (
                f"{doc.get('doc_type', 'unknown')}_"
                f"{sha256((doc.get('source_url', '') + str(index)).encode('utf-8')).hexdigest()}_"
                f"{metadata['content_hash'][:16]}"
            )

            return {
                'chunk': chunk,
                'embedding': embedding,
                'metadata': metadata,
                'id': doc_id
            }

        except Exception as e:
            logger.error(f"Error processing chunk {index}: {e}")
            return None

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using AsyncOpenAI (NON-BLOCKING).

        Performance: 500ms-2s per call (async allows concurrent processing)
        """
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents (async)"""
        try:
            # Check cache first
            cache_key = f"search_{hash(query)}_{n_results}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for search: {query[:50]}...")
                return cached_result

            # Generate query embedding (ASYNC)
            query_embedding = await self._get_embedding(query)

            # Search ChromaDB (wrapped in to_thread since ChromaDB is sync)
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]
                }
                formatted_results.append(result)

            # Cache results (async)
            await self.cache.set(cache_key, formatted_results)

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def generate_response(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate response using async RAG.

        Performance improvement: 2-4x faster than sync version
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"response_{hash(query + str(conversation_history))}"
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                cached_response['cached'] = True
                cached_response['response_time'] = time.time() - start_time
                return cached_response

            # Search for relevant documents (ASYNC)
            relevant_docs = await self.search_documents(query, n_results=5)

            if not relevant_docs:
                return {
                    'response': "I couldn't find relevant information in the documentation. Please try rephrasing your question.",
                    'sources': [],
                    'code_examples': [],
                    'response_time': time.time() - start_time,
                    'cached': False
                }

            # Build context
            context = self._build_context(relevant_docs)

            # Build conversation history
            history_context = ""
            if conversation_history:
                history_context = "\n".join([
                    f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                    for msg in conversation_history[-3:]
                ])

            # Generate response (ASYNC)
            response_data = await self._generate_llm_response(query, context, history_context)

            # Extract sources
            sources = self._extract_sources(relevant_docs)

            # Calculate response time
            response_time = time.time() - start_time

            result = {
                'response': response_data.get('response', ''),
                'code_examples': response_data.get('code_examples', []),
                'sources': sources,
                'related_questions': response_data.get('related_questions', []),
                'response_time': response_time,
                'cached': False
            }

            # Cache the result (async)
            await self.cache.set(cache_key, result)

            logger.info(f"Generated response in {response_time:.2f}s (async)")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"I encountered an error: {str(e)}. Please try again.",
                'sources': [],
                'code_examples': [],
                'response_time': time.time() - start_time,
                'cached': False
            }

    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Build context string from relevant documents"""
        context_parts = []
        for doc in relevant_docs:
            source_info = f"Source: {doc['metadata'].get('title', 'Unknown')} ({doc['metadata'].get('doc_type', 'unknown')})"
            context_parts.append(f"{source_info}\n{doc['content']}\n")

        return "\n---\n".join(context_parts)

    async def _generate_llm_response(self, query: str, context: str, history: str) -> Dict[str, Any]:
        """
        Generate response using AsyncOpenAI (NON-BLOCKING).

        Performance: 1-3s per call (async allows concurrent processing)
        """
        try:
            # Detect if API-related query
            is_api_query = self._is_api_related_query(query, context)

            if is_api_query:
                system_prompt = self._get_api_specialized_prompt()
            else:
                system_prompt = self._get_standard_prompt()

            user_prompt = self._build_user_prompt(query, context, history, is_api_query)

            # AsyncOpenAI call (NON-BLOCKING)
            response = await self.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=Config.MAX_RESPONSE_TOKENS,
                temperature=Config.TEMPERATURE,
                response_format={"type": "json_object"}
            )

            try:
                content = response.choices[0].message.content
                if content:
                    result = json.loads(content)

                    # Enhance with code examples for API queries
                    if is_api_query and 'endpoints' in result:
                        result = self._enhance_with_code_examples(result, context)

                    return result

            except json.JSONDecodeError:
                return {
                    'response': response.choices[0].message.content or '',
                    'code_examples': [],
                    'related_questions': []
                }

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    def _extract_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract and format source information"""
        sources = []
        seen_sources = set()

        for doc in relevant_docs:
            metadata = doc['metadata']
            source_key = f"{metadata.get('title', 'Unknown')}_{metadata.get('source_url', '')}"

            if source_key not in seen_sources:
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'url': metadata.get('source_url', ''),
                    'type': metadata.get('doc_type', 'unknown'),
                    'relevance': round(doc.get('relevance_score', 0), 2)
                })
                seen_sources.add(source_key)

        return sources[:3]

    def _is_api_related_query(self, query: str, context: str) -> bool:
        """Detect if query is API-related"""
        api_keywords = [
            'api', 'endpoint', 'rest', 'post', 'get', 'put', 'delete', 'patch',
            'authentication', 'auth', 'token', 'bearer', 'api key', 'header',
            'parameter', 'request', 'response', 'status code', 'curl', 'json'
        ]

        query_lower = query.lower()
        context_lower = context.lower()

        query_has_api_keywords = any(keyword in query_lower for keyword in api_keywords)
        context_has_api_content = any([
            'api_endpoint' in context_lower,
            'method:' in context_lower and any(method in context_lower for method in ['get', 'post', 'put', 'delete']),
            'endpoint:' in context_lower
        ])

        return query_has_api_keywords or context_has_api_content

    def _get_api_specialized_prompt(self) -> str:
        """Get system prompt for API documentation"""
        return """You are DocRag, an expert API documentation assistant.

Your expertise:
1. API Endpoints: HTTP methods, paths, parameters, responses
2. Authentication: OAuth, API keys, Bearer tokens
3. Request/Response: JSON schemas, headers, status codes
4. Code Examples: Multi-language examples
5. Integration: SDKs, rate limiting, best practices

Response format (JSON):
- answer: Technical explanation (markdown)
- examples: Code examples array
- endpoints: Relevant API endpoints
- authentication: Auth requirements
- parameters: Key parameters
- response_format: Response structure
- error_codes: Common errors
- related_concepts: Related topics"""

    def _get_standard_prompt(self) -> str:
        """Get system prompt for general documentation"""
        return """You are DocRag, a technical documentation assistant.

Guidelines:
- Provide accurate answers based on context
- Generate working code examples
- Explain concepts clearly
- Cite sources properly
- Suggest related questions

Response format (JSON):
- response: Main answer (markdown)
- code_examples: Array of code blocks
- related_questions: Follow-up questions"""

    def _build_user_prompt(self, query: str, context: str, history: str, is_api_query: bool) -> str:
        """Build user prompt"""
        if is_api_query:
            return f"""Context from API documentation:
{context}

Previous conversation:
{history}

API question: {query}

Provide comprehensive API answer with:
1. Clear explanation
2. Authentication requirements
3. Multi-language examples
4. Parameter details
5. Error codes
6. Best practices

Format as JSON with API-specific fields."""
        else:
            return f"""Context:
{context}

Previous conversation:
{history}

Question: {query}

Provide comprehensive answer with code examples. Format as JSON."""

    def _enhance_with_code_examples(self, result: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Enhance response with auto-generated code examples"""
        try:
            endpoint_info = self._extract_endpoint_info(result, context)

            if endpoint_info:
                auto_examples = self.code_generator.generate_multi_language_examples(endpoint_info)
                existing_examples = result.get('examples', [])
                all_examples = existing_examples + auto_examples
                unique_examples = self._deduplicate_examples(all_examples)
                result['examples'] = unique_examples
                logger.info(f"Enhanced with {len(auto_examples)} code examples")

            return result

        except Exception as e:
            logger.warning(f"Failed to enhance with code examples: {e}")
            return result

    def _extract_endpoint_info(self, result: Dict[str, Any], context: str) -> Optional[Dict[str, Any]]:
        """Extract endpoint info for code generation"""
        try:
            endpoints = result.get('endpoints', [])
            if not endpoints:
                return None

            endpoint = endpoints[0] if isinstance(endpoints, list) else endpoints

            if isinstance(endpoint, str):
                parts = endpoint.split(' ', 1)
                method, path = (parts[0], parts[1]) if len(parts) == 2 else ('GET', endpoint)
            else:
                method = endpoint.get('method', 'GET')
                path = endpoint.get('path', '/')

            return {
                'method': method,
                'path': path,
                'base_url': 'https://api.example.com',
                'parameters': [],
                'headers': {'Content-Type': 'application/json'},
                'auth': self._extract_auth_from_result(result)
            }

        except Exception as e:
            logger.warning(f"Failed to extract endpoint info: {e}")
            return None

    def _extract_auth_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract auth info from result"""
        auth_info = {}
        auth_section = result.get('authentication', '')

        if isinstance(auth_section, str):
            auth_lower = auth_section.lower()
            if 'bearer' in auth_lower or 'token' in auth_lower:
                auth_info = {'type': 'bearer', 'description': 'Bearer token'}
            elif 'api key' in auth_lower:
                auth_info = {'type': 'api_key', 'location': 'header', 'name': 'X-API-Key'}

        return auth_info

    def _deduplicate_examples(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate examples by language"""
        seen = set()
        unique = []

        for example in examples:
            lang = example.get('language', 'unknown')
            if lang not in seen:
                seen.add(lang)
                unique.append(example)

        return unique

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics (async)"""
        try:
            count = await asyncio.to_thread(self.collection.count)
            return {
                'document_count': count,
                'collection_name': Config.COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'document_count': 0, 'collection_name': Config.COLLECTION_NAME}
