"""
RAG Search Tool - Documentation Search

LangChain tool for searching API documentation using the existing RAG engine.
"""

from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool"""
    query: str = Field(
        description="The search query for documentation. Be specific about what you're looking for."
    )


class RAGSearchTool(BaseTool):
    """
    Tool for searching API documentation using RAG (Retrieval-Augmented Generation).

    This tool searches through the vector database to find relevant documentation
    sections based on semantic similarity.

    Example usage by agent:
        Action: documentation_search
        Action Input: "How to create a FastAPI endpoint with async"
    """

    name: str = "documentation_search"
    description: str = (
        "Searches API documentation for relevant information. "
        "Use this when you need to find information about APIs, "
        "endpoints, parameters, authentication, or usage examples. "
        "Input should be a specific search query about what you're looking for."
    )
    args_schema: Type[BaseModel] = RAGSearchInput
    rag_engine: Optional[Any] = None  # RAG engine instance

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'rag_engine') or self.rag_engine is None:
            object.__setattr__(self, 'rag_engine', None)

    def _get_rag_engine(self):
        """Lazy load RAG engine to avoid circular imports"""
        if self.rag_engine is None:
            from rag_engine_async import AsyncRAGEngine
            self.rag_engine = AsyncRAGEngine()
            logger.info("RAG engine initialized in tool")
        return self.rag_engine

    def _run(self, query: str) -> str:
        """Sync version (not used, but required by BaseTool interface)"""
        raise NotImplementedError("Use async version (_arun)")

    async def _arun(self, query: str) -> str:
        """
        Execute RAG search (async).

        Args:
            query: Search query

        Returns:
            Formatted search results with sources
        """
        try:
            logger.info(f"RAG tool searching for: {query}")

            engine = self._get_rag_engine()

            # Search documents
            results = await engine.search_documents(query, n_results=5)

            # Check if results are empty (handle both no results and empty inner list)
            if (not results or 'documents' not in results or not results['documents'] or
                not results['documents'][0] or len(results['documents'][0]) == 0):
                return "No relevant documentation found for this query. Try rephrasing or being more specific."

            # Format results
            formatted = "Found relevant documentation:\n\n"

            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            ), 1):
                title = metadata.get('title', 'Unknown Source')
                url = metadata.get('source_url', '')
                doc_type = metadata.get('doc_type', '')

                formatted += f"[{i}] {title}"
                if doc_type:
                    formatted += f" ({doc_type})"
                formatted += f"\n"

                # Truncate document content for clarity
                content = doc[:400] if len(doc) > 400 else doc
                formatted += f"{content}...\n"

                if url:
                    formatted += f"Source: {url}\n"

                formatted += "\n"

            logger.info(f"RAG tool found {len(results['documents'][0])} results")

            return formatted

        except Exception as e:
            logger.error(f"RAG search error: {e}", exc_info=True)
            return f"Error searching documentation: {str(e)}. Please try rephrasing your query."
