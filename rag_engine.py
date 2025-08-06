import json
import logging
import time
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
from cache_manager import CacheManager

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
                    }
                    
                    # Create unique ID
                    doc_id = f"{doc.get('doc_type', 'unknown')}_{hash(doc.get('source_url', ''))}_chunk_{i}"
                    
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
