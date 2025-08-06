# DocRag - Technical Documentation Assistant

## Overview

DocRag is an AI-powered Retrieval-Augmented Generation (RAG) system that helps developers find answers from technical documentation using natural language queries. The system currently supports React, Python, and FastAPI documentation, providing semantic search capabilities with contextual responses and code generation.

The application uses a Flask-based web interface where users can ask questions about technical topics and receive AI-generated responses based on relevant documentation chunks retrieved from a vector database. The system processes documentation from multiple sources, chunks them for efficient retrieval, and uses OpenAI's GPT models to generate contextual responses.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

**Web Framework**: Flask-based application with SQLAlchemy ORM for database operations. The app follows a modular blueprint structure with separate route handlers and uses session management for conversation tracking.

**Database Layer**: SQLite database (configurable to other SQL databases) with three main entities:
- `Conversation`: Stores chat history with user queries, AI responses, sources, and feedback
- `DocumentChunk`: Stores processed documentation chunks with metadata and embedding references
- `RateLimit`: Tracks API usage per IP address for rate limiting

**RAG Engine**: Core retrieval-augmented generation system built with:
- ChromaDB for vector storage and similarity search
- OpenAI embeddings for document vectorization
- LangChain's RecursiveCharacterTextSplitter for intelligent text chunking
- Custom caching layer for performance optimization

**Document Processing**: Automated system that processes documentation from configured sources (React, Python, FastAPI) using web scraping with BeautifulSoup and trafilatura. Documents are chunked, embedded, and stored in both the vector database and SQL database for metadata tracking.

**Frontend Architecture**: Server-side rendered templates using Bootstrap with dark theme, featuring:
- Real-time chat interface with WebSocket-like behavior via AJAX
- Syntax highlighting for code blocks using Prism.js
- Responsive design with mobile support
- Character count tracking and input validation

**Caching System**: File-based cache manager with TTL (time-to-live) support for frequently accessed queries, reducing API calls and improving response times.

**Rate Limiting**: IP-based rate limiting system that tracks requests per minute and stores limits in the database, preventing API abuse and managing costs.

**Security & Validation**: Input sanitization and validation utilities that clean user queries, validate API keys, and prevent common security issues like XSS and injection attacks.

## External Dependencies

**AI/ML Services**:
- OpenAI API for GPT-4o model access (text generation and embeddings)
- ChromaDB for vector database operations and similarity search

**Web Scraping & Processing**:
- BeautifulSoup4 for HTML parsing
- trafilatura for content extraction from web pages
- requests for HTTP operations

**Python Libraries**:
- Flask with SQLAlchemy for web framework and ORM
- LangChain for text processing and chunking
- python-dotenv for environment configuration

**Frontend Assets**:
- Bootstrap CSS framework with Replit dark theme
- Font Awesome for icons
- Prism.js for syntax highlighting

**Documentation Sources**:
- React.dev official documentation
- Python.org official documentation  
- FastAPI official documentation

The system is designed to be extensible, allowing additional documentation sources to be configured through the Config class and processed by the DocumentProcessor module.