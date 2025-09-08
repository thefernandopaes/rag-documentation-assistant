import json
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
from cache_manager import CacheManager
from hashlib import sha256

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine with ChromaDB and OpenAI"""
        Config.validate_config()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize ChromaDB
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
        
        # Initialize cache
        self.cache = CacheManager()
        
        logger.info("RAG Engine initialized successfully")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        try:
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []
            
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    # Generate embedding
                    embedding = self._get_embedding(chunk)
                    
                    # Create metadata
                    metadata = {
                        'source_url': doc.get('source_url', ''),
                        'title': doc.get('title', ''),
                        'doc_type': doc.get('doc_type', ''),
                        'chunk_index': i,
                        'version': doc.get('version', ''),
                        'content_hash': sha256(chunk.encode('utf-8')).hexdigest(),
                    }
                    
                    # Create unique ID
                    # deterministic idempotent ID using source + chunk hash + index
                    doc_id = (
                        f"{doc.get('doc_type', 'unknown')}_"
                        f"{sha256((doc.get('source_url','') + str(i)).encode('utf-8')).hexdigest()}_"
                        f"{metadata['content_hash'][:16]}"
                    )
                    
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    all_metadatas.append(metadata)
                    all_ids.append(doc_id)
            
            # Add to ChromaDB
            if all_chunks:
                self.collection.add(
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                logger.info(f"Added {len(all_chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Replace existing documents for given sources with new content (idempotent refresh).

        For each document, removes previous vectors by source_url+doc_type and inserts fresh chunks.
        """
        try:
            # First, delete existing vectors for each unique (source_url, doc_type)
            unique_keys = set(
                (
                    doc.get('source_url', ''),
                    doc.get('doc_type', ''),
                )
                for doc in documents
            )

            for source_url, doc_type in unique_keys:
                if not source_url:
                    continue
                try:
                    self.collection.delete(where={
                        'source_url': source_url,
                        'doc_type': doc_type,
                    })
                except Exception as del_err:
                    logger.warning(f"Delete failed for {source_url} ({doc_type}): {del_err}")

            # Then add new vectors
            self.add_documents(documents)

        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Check cache first
            cache_key = f"search_{hash(query)}_{n_results}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Search ChromaDB
            results = self.collection.query(
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
                    'relevance_score': 1 - results['distances'][0][i]  # Convert distance to relevance
                }
                formatted_results.append(result)
            
            # Cache results
            self.cache.set(cache_key, formatted_results)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def generate_response(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate response using RAG"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"response_{hash(query + str(conversation_history))}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                cached_response['cached'] = True
                return cached_response
            
            # Search for relevant documents
            relevant_docs = self.search_documents(query, n_results=5)
            
            if not relevant_docs:
                return {
                    'response': "I couldn't find relevant information in the documentation. Please try rephrasing your question or asking about React, Python, or FastAPI topics.",
                    'sources': [],
                    'code_examples': [],
                    'response_time': time.time() - start_time,
                    'cached': False
                }
            
            # Build context from relevant documents
            context = self._build_context(relevant_docs)
            
            # Build conversation history
            history_context = ""
            if conversation_history:
                history_context = "\n".join([
                    f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                    for msg in conversation_history[-3:]  # Last 3 exchanges
                ])
            
            # Generate response
            response_data = self._generate_llm_response(query, context, history_context)
            
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
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"I encountered an error while processing your question: {str(e)}. Please try again.",
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
    
    def _generate_llm_response(self, query: str, context: str, history: str) -> Dict[str, Any]:
        """Generate response using OpenAI GPT"""
        try:
            system_prompt = """You are DocRag, an expert technical documentation assistant specializing in React, Python, and FastAPI. 

Your role is to:
1. Provide accurate, contextual answers based on official documentation
2. Generate functional code examples when appropriate
3. Explain concepts clearly for both beginners and experienced developers
4. Cite sources properly
5. Suggest related questions

Guidelines:
- Always base your answers on the provided context
- Generate working code examples with proper syntax highlighting
- Explain code step-by-step when helpful
- Be concise but comprehensive
- If you don't know something, say so clearly
- Format code examples properly with language tags

Response format should be JSON with these fields:
- response: Main answer (markdown format)
- code_examples: Array of code blocks with language and explanation
- related_questions: Array of 2-3 suggested follow-up questions"""

            user_prompt = f"""Context from documentation:
{context}

Previous conversation:
{history}

User question: {query}

Please provide a comprehensive answer with code examples if applicable. Format your response as JSON."""

            response = self.openai_client.chat.completions.create(
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
                    return json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'response': response.choices[0].message.content,
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
        
        return sources[:3]  # Limit to top 3 sources
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': Config.COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'document_count': 0, 'collection_name': Config.COLLECTION_NAME}


class APIChunker:
    """Chunking strategy specialized for API documentation."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """Initialize API chunker with configurable parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # API-specific separators for better chunking
        self.api_separators = [
            "\n## ",      # Headers (endpoints)
            "\n### ",     # Sub-headers (parameters, responses)
            "\n#### ",    # Sub-sub-headers
            "\n\n",       # Paragraph breaks
            "\n",         # Line breaks
            ". ",         # Sentence endings
            ", ",         # Comma separations
            " ",          # Word boundaries
            ""            # Character fallback
        ]
        
        # Initialize standard text splitter as fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.api_separators
        )
        
        logger.info(f"APIChunker initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_api_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk an API document using specialized strategies.
        
        Args:
            document: Document with API-specific metadata
            
        Returns:
            List of chunked documents with preserved metadata
        """
        doc_type = document.get('doc_type', 'unknown')
        
        if doc_type == 'api_endpoint':
            return self.chunk_by_endpoint(document)
        elif doc_type == 'api_overview':
            return self.chunk_by_sections(document)
        elif doc_type == 'openapi':
            return self.chunk_openapi_spec(document)
        elif doc_type in ['api_html', 'postman_request']:
            return self.chunk_by_structure(document)
        else:
            # Fallback to standard chunking
            return self.chunk_standard(document)
    
    def chunk_by_endpoint(self, api_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks specialized for API endpoints."""
        chunks = []
        content = api_doc.get('content', '')
        
        # Get endpoint metadata
        endpoint_data = api_doc.get('endpoint_data', {})
        method = endpoint_data.get('method', 'GET')
        path = endpoint_data.get('path', '/')
        
        # Main endpoint chunk
        main_chunk_content = self._create_endpoint_summary(api_doc)
        main_chunk = {
            **api_doc,
            'content': main_chunk_content,
            'chunk_type': 'endpoint_summary',
            'chunk_index': 0,
            'content_hash': self._generate_content_hash(main_chunk_content),
            'api_method': method,
            'api_path': path
        }
        chunks.append(main_chunk)
        
        # Parameter chunks (if parameters are complex)
        parameters = endpoint_data.get('parameters', [])
        if len(parameters) > 3:  # Many parameters deserve separate chunk
            param_content = self._create_parameters_chunk(parameters, method, path)
            param_chunk = {
                **api_doc,
                'content': param_content,
                'chunk_type': 'parameters',
                'chunk_index': 1,
                'content_hash': self._generate_content_hash(param_content),
                'api_method': method,
                'api_path': path,
                'title': f"{method} {path} - Parameters"
            }
            chunks.append(param_chunk)
        
        # Response chunks (if responses are detailed)
        responses = endpoint_data.get('responses', [])
        if len(responses) > 2:  # Multiple responses deserve separate chunk
            response_content = self._create_responses_chunk(responses, method, path, content)
            response_chunk = {
                **api_doc,
                'content': response_content,
                'chunk_type': 'responses',
                'chunk_index': 2,
                'content_hash': self._generate_content_hash(response_content),
                'api_method': method,
                'api_path': path,
                'title': f"{method} {path} - Responses"
            }
            chunks.append(response_chunk)
        
        # Code example chunks
        if 'code_examples' in api_doc:
            for i, example in enumerate(api_doc.get('code_examples', [])):
                if len(str(example)) > 200:  # Large examples get their own chunk
                    example_content = self._create_example_chunk(example, method, path)
                    example_chunk = {
                        **api_doc,
                        'content': example_content,
                        'chunk_type': 'code_example',
                        'chunk_index': 3 + i,
                        'content_hash': self._generate_content_hash(example_content),
                        'api_method': method,
                        'api_path': path,
                        'title': f"{method} {path} - {example.get('language', 'Code')} Example"
                    }
                    chunks.append(example_chunk)
        
        return chunks
    
    def chunk_by_sections(self, api_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk API overview document by logical sections."""
        content = api_doc.get('content', '')
        chunks = []
        
        # Split by headers
        sections = self._split_by_headers(content)
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very small sections
                continue
                
            chunk = {
                **api_doc,
                'content': section.strip(),
                'chunk_type': 'api_section',
                'chunk_index': i,
                'content_hash': self._generate_content_hash(section.strip()),
                'title': f"{api_doc.get('title', 'API Overview')} - Section {i+1}"
            }
            chunks.append(chunk)
        
        return chunks if chunks else [api_doc]  # Return original if no sections found
    
    def chunk_openapi_spec(self, api_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk OpenAPI specification documents."""
        # For OpenAPI specs, we typically want to preserve endpoint integrity
        return self.chunk_by_endpoint(api_doc)
    
    def chunk_by_structure(self, api_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk by document structure (headers, code blocks, etc.)."""
        content = api_doc.get('content', '')
        
        # Use standard text splitter but with API-aware separators
        text_chunks = self.text_splitter.split_text(content)
        
        chunks = []
        for i, chunk_content in enumerate(text_chunks):
            if len(chunk_content.strip()) < 50:  # Skip tiny chunks
                continue
                
            chunk = {
                **api_doc,
                'content': chunk_content.strip(),
                'chunk_type': 'structured',
                'chunk_index': i,
                'content_hash': self._generate_content_hash(chunk_content.strip())
            }
            chunks.append(chunk)
        
        return chunks if chunks else [api_doc]
    
    def chunk_standard(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Standard chunking for non-API documents."""
        content = document.get('content', '')
        text_chunks = self.text_splitter.split_text(content)
        
        chunks = []
        for i, chunk_content in enumerate(text_chunks):
            if len(chunk_content.strip()) < 50:
                continue
                
            chunk = {
                **document,
                'content': chunk_content.strip(),
                'chunk_type': 'standard',
                'chunk_index': i,
                'content_hash': self._generate_content_hash(chunk_content.strip())
            }
            chunks.append(chunk)
        
        return chunks if chunks else [document]
    
    def _create_endpoint_summary(self, api_doc: Dict[str, Any]) -> str:
        """Create a concise endpoint summary."""
        endpoint_data = api_doc.get('endpoint_data', {})
        method = endpoint_data.get('method', 'GET')
        path = endpoint_data.get('path', '/')
        summary = endpoint_data.get('summary', '')
        description = endpoint_data.get('description', '')
        
        summary_parts = [f"# {method} {path}"]
        
        if summary:
            summary_parts.append(f"\n**Summary:** {summary}")
        
        if description:
            summary_parts.append(f"\n**Description:** {description}")
        
        # Add key information
        parameters = endpoint_data.get('parameters', [])
        if parameters:
            required_params = [p for p in parameters if p.get('required', False)]
            if required_params:
                param_names = [p.get('name', 'unknown') for p in required_params]
                summary_parts.append(f"\n**Required Parameters:** {', '.join(param_names)}")
        
        tags = endpoint_data.get('tags', [])
        if tags:
            summary_parts.append(f"\n**Categories:** {', '.join(tags)}")
        
        return '\n'.join(summary_parts)
    
    def _create_parameters_chunk(self, parameters: List[Dict[str, Any]], method: str, path: str) -> str:
        """Create a detailed parameters chunk."""
        content_parts = [f"# {method} {path} - Parameters"]
        
        for param in parameters:
            param_name = param.get('name', 'unknown')
            param_type = param.get('type', param.get('schema', {}).get('type', 'string'))
            param_in = param.get('in', 'query')
            param_required = param.get('required', False)
            param_desc = param.get('description', '')
            
            required_text = " (required)" if param_required else " (optional)"
            content_parts.append(f"\n## {param_name}")
            content_parts.append(f"- **Type:** {param_type}")
            content_parts.append(f"- **Location:** {param_in}")
            content_parts.append(f"- **Required:** {'Yes' if param_required else 'No'}")
            if param_desc:
                content_parts.append(f"- **Description:** {param_desc}")
        
        return '\n'.join(content_parts)
    
    def _create_responses_chunk(self, responses: List[str], method: str, path: str, full_content: str) -> str:
        """Create a detailed responses chunk."""
        content_parts = [f"# {method} {path} - Responses"]
        
        # Extract response information from full content
        # This is a simplified extraction - in practice, you'd want more sophisticated parsing
        response_section = ""
        if "### Responses" in full_content:
            response_section = full_content.split("### Responses")[1].split("###")[0]
        
        if response_section:
            content_parts.append(response_section.strip())
        else:
            # Fallback: list response codes
            content_parts.append(f"\n**Response Codes:** {', '.join(responses)}")
        
        return '\n'.join(content_parts)
    
    def _create_example_chunk(self, example: Dict[str, Any], method: str, path: str) -> str:
        """Create a code example chunk."""
        language = example.get('language', 'code')
        code = example.get('code', str(example))
        title = example.get('title', f"{language} Example")
        
        content_parts = [
            f"# {method} {path} - {title}",
            f"\n```{language}",
            code,
            "```"
        ]
        
        if 'explanation' in example:
            content_parts.append(f"\n**Explanation:** {example['explanation']}")
        
        return '\n'.join(content_parts)
    
    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by headers (# ## ###)."""
        import re
        
        # Split by headers while preserving them
        header_pattern = r'(^#{1,4}\s+.*$)'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        sections = []
        current_section = ""
        
        for part in parts:
            if re.match(r'^#{1,4}\s+', part):  # This is a header
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part
            else:
                current_section += part
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunked documents."""
        if not chunks:
            return {}
        
        chunk_types = {}
        chunk_sizes = []
        
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            chunk_sizes.append(len(chunk.get('content', '')))
        
        return {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types,
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0
        }
