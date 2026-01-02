"""
Async RAG Engine - Support for OpenAI, OpenRouter, and Gemini

Multi-provider support:
- OpenAI (GPT-4o, etc.)
- OpenRouter (Various models)
- Google Gemini (Gemini 1.5 Pro/Flash)

High-performance architecture with async client support.
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
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from app.core.cache import AsyncInMemoryCache
from app.core.code_generator import CodeExampleGenerator
from hashlib import sha256

logger = logging.getLogger(__name__)


class AsyncRAGEngine:
    """Async RAG Engine supporting multiple AI providers."""

    def __init__(self):
        """Initialize the async RAG engine"""
        Config.validate_config()
        self.provider = Config.AI_PROVIDER

        # --- Chat Client Initialization ---
        if self.provider == "gemini":
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.chat_model_name = Config.GEMINI_MODEL
            # Gemini client is stateless, configured globally
            logger.info(f"Gemini client initialized with model: {self.chat_model_name}")
            
        elif Config.USE_OPENROUTER:
            self.chat_client = AsyncOpenAI(
                api_key=Config.OPENROUTER_API_KEY,
                base_url=Config.OPENROUTER_BASE_URL
            )
            self.chat_model_name = Config.OPENROUTER_MODEL
            logger.info(f"OpenRouter client initialized with model: {self.chat_model_name}")
            
        else: # OpenAI
            self.chat_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            self.chat_model_name = Config.OPENAI_MODEL
            logger.info(f"OpenAI client initialized with model: {self.chat_model_name}")


        # --- Embedding Client Initialization ---
        if Config.USE_OPENAI_EMBEDDINGS:
            # Use OpenAI for embeddings (Standard)
            self.embedding_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            self.embedding_model = Config.OPENAI_EMBEDDING_MODEL
            self.embedding_provider = "openai"
            logger.info(f"OpenAI client initialized for embeddings: {self.embedding_model}")
            
        elif self.provider == "gemini":
            # Use Gemini for embeddings
            self.embedding_model = Config.GEMINI_EMBEDDING_MODEL
            self.embedding_provider = "gemini"
            logger.info(f"Gemini configured for embeddings: {self.embedding_model}")
            
        else:
            # OpenRouter (if not OpenAI embeddings)
            self.embedding_client = self.chat_client
            self.embedding_model = Config.OPENROUTER_EMBEDDING_MODEL
            self.embedding_provider = "openai_compatible" # OpenRouter uses OpenAI format


        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        self.cache = AsyncInMemoryCache(ttl=Config.CACHE_TTL, max_size=1000)
        self.code_generator = CodeExampleGenerator()

        logger.info(f"Async RAG Engine initialized (Provider: {self.provider})")

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding based on configured provider"""
        start_time = time.time()
        try:
            if self.embedding_provider == "gemini":
                # Gemini Embedding
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
            
            else:
                # OpenAI / OpenRouter Embedding
                response = await self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text.replace("\n", " ")
                )
                embedding = response.data[0].embedding

            self._last_embedding_time = time.time() - start_time
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding ({self.embedding_provider}): {e}")
            raise

    async def _generate_llm_response(self, query: str, context: str, history: str) -> Dict[str, Any]:
        """Generate response handling multiple providers"""
        try:
            is_api_query = self._is_api_related_query(query, context)
            
            if is_api_query:
                system_prompt = self._get_api_specialized_prompt()
            else:
                system_prompt = self._get_standard_prompt()

            user_prompt = self._build_user_prompt(query, context, history, is_api_query)

            if self.provider == "gemini":
                return await self._generate_gemini_response(system_prompt, user_prompt, is_api_query, context)
            else:
                return await self._generate_openai_response(system_prompt, user_prompt, is_api_query, context)

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    async def _generate_gemini_response(self, system_prompt: str, user_prompt: str, is_api_query: bool, context: str) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        try:
            model = genai.GenerativeModel(self.chat_model_name)
            
            # Gemini generic prompt structure
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nPlease output valid JSON."
            
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=Config.TEMPERATURE,
                    response_mime_type="application/json"
                )
            )
            
            content = response.text
            result = json.loads(content)
            
            if is_api_query and 'endpoints' in result:
                result = self._enhance_with_code_examples(result, context)
                
            return result
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise

    async def _generate_openai_response(self, system_prompt: str, user_prompt: str, is_api_query: bool, context: str) -> Dict[str, Any]:
        """Generate response using OpenAI/OpenRouter API"""
        max_tokens = min(2000 if is_api_query else 1500, Config.MAX_RESPONSE_TOKENS)
        
        response = await self.chat_client.chat.completions.create(
            model=self.chat_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=Config.TEMPERATURE,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        if not content:
            return {'response': '', 'code_examples': []}
            
        result = json.loads(content)
        
        if is_api_query and 'endpoints' in result:
            result = self._enhance_with_code_examples(result, context)
            
        return result

    # --- Standard RAG Methods (Unchanged mostly) ---

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store (async)"""
        try:
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            for doc in documents:
                chunks = self.text_splitter.split_text(doc['content'])
                chunk_tasks = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50: continue
                    chunk_tasks.append(self._process_chunk_async(doc, chunk, i))

                processed_chunks = await asyncio.gather(*chunk_tasks)

                for result in processed_chunks:
                    if result:
                        all_chunks.append(result['chunk'])
                        all_embeddings.append(result['embedding'])
                        all_metadatas.append(result['metadata'])
                        all_ids.append(result['id'])

            if all_chunks:
                await asyncio.to_thread(
                    self.collection.add,
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                logger.info(f"Added {len(all_chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def _process_chunk_async(self, doc: Dict[str, Any], chunk: str, index: int) -> Optional[Dict[str, Any]]:
        try:
            embedding = await self._get_embedding(chunk)
            metadata = {
                'source_url': doc.get('source_url', ''),
                'title': doc.get('title', ''),
                'doc_type': doc.get('doc_type', ''),
                'chunk_index': index,
                'version': doc.get('version', ''),
                'content_hash': sha256(chunk.encode('utf-8')).hexdigest(),
            }
            doc_id = f"{doc.get('doc_type', 'unknown')}_{sha256((doc.get('source_url', '') + str(index)).encode('utf-8')).hexdigest()}_{metadata['content_hash'][:16]}"
            return {'chunk': chunk, 'embedding': embedding, 'metadata': metadata, 'id': doc_id}
        except Exception as e:
            logger.error(f"Error processing chunk {index}: {e}")
            return None

    async def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            cache_key = f"search_{hash(query)}_{n_results}"
            cached_result = await self.cache.get(cache_key)
            if cached_result: return cached_result

            query_embedding = await self._get_embedding(query)
            
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'relevance_score': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)

            await self.cache.set(cache_key, formatted_results)
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def generate_response(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        perf_metrics = {}
        total_start = time.time()

        try:
            # Cache Check
            cache_key = f"response_{hash(query + str(conversation_history))}"
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                cached_response['cached'] = True
                cached_response['response_time'] = time.time() - total_start
                return cached_response

            # Query Enhancement
            enhanced_query = query
            if self._is_self_query(query):
                enhanced_query = self._enhance_self_query(query)

            # Document Search
            relevant_docs = await self.search_documents(enhanced_query, n_results=3)

            if not relevant_docs:
                return {
                    'response': "I couldn't find relevant information in the documentation.",
                    'sources': [],
                    'code_examples': [],
                    'response_time': time.time() - total_start,
                    'cached': False
                }

            # Context Building
            context = self._build_context(relevant_docs)
            history_context = ""
            if conversation_history:
                history_context = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history[-3:]])

            # LLM Generation
            response_data = await self._generate_llm_response(query, context, history_context)

            # Sources
            sources = self._extract_sources(relevant_docs)

            result = {
                'response': response_data.get('answer', response_data.get('response', '')),
                'code_examples': response_data.get('examples', response_data.get('code_examples', [])),
                'sources': sources,
                'related_questions': response_data.get('related_questions', response_data.get('related_concepts', [])),
                'response_time': time.time() - total_start,
                'cached': False
            }

            # Copy API fields
            for field in ['endpoints', 'authentication', 'parameters', 'response_format', 'error_codes']:
                if field in response_data:
                    result[field] = response_data[field]

            await self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Error: {str(e)}",
                'sources': [],
                'code_examples': [],
                'response_time': time.time() - total_start,
                'cached': False
            }

    # --- Helper Methods ---
    def _is_self_query(self, query: str) -> bool:
        self_keywords = ['this api', 'docrag', 'your system']
        return any(k in query.lower() for k in self_keywords)

    def _enhance_self_query(self, query: str) -> str:
        return query.replace('this api', 'the DocRag API')

    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        parts = []
        for doc in relevant_docs:
            parts.append(f"Source: {doc['metadata'].get('title')} ({doc['metadata'].get('doc_type')})\n{doc['content']}\n")
        return "\n---\n".join(parts)

    def _extract_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen = set()
        for doc in relevant_docs:
            key = doc['metadata'].get('source_url')
            if key not in seen:
                sources.append({
                    'title': doc['metadata'].get('title'),
                    'url': key,
                    'type': doc['metadata'].get('doc_type'),
                    'relevance': round(doc.get('relevance_score', 0), 2)
                })
                seen.add(key)
        return sources[:3]

    def _is_api_related_query(self, query: str, context: str) -> bool:
        api_keywords = ['api', 'endpoint', 'rest', 'auth', 'json']
        return any(k in query.lower() for k in api_keywords) or 'api_endpoint' in context.lower()

    def _get_api_specialized_prompt(self) -> str:
        return """You are DocRag, an API documentation expert.
Response format (JSON) - BE CONCISE:
- answer: Clear explanation (markdown)
- examples: Code examples (max 3)
- endpoints: Relevant endpoints
- authentication: Auth requirements
- parameters: Key parameters
- error_codes: Common errors
- related_concepts: Array of related topics"""

    def _get_standard_prompt(self) -> str:
        return """You are DocRag, a technical documentation expert.
Response format (JSON) - BE CONCISE:
- response: Clear answer (markdown)
- code_examples: Working examples
- related_questions: Follow-up questions"""

    def _build_user_prompt(self, query: str, context: str, history: str, is_api_query: bool) -> str:
        type_str = "API question" if is_api_query else "Question"
        return f"""Context:\n{context}\n\nHistory:\n{history}\n\n{type_str}: {query}\n\nProvide comprehensive answer in JSON."""

    def _enhance_with_code_examples(self, result: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Delegate to code generator (stub implementation)"""
        # In a full implementation, this calls self.code_generator
        return result

