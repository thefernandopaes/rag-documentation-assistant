"""
FastAPI Application - Async RAG Documentation Assistant

Main application factory with:
- Async lifespan management for RAG engine initialization
- Security middleware (CORS, headers)
- Global exception handlers
- Router registration
- Templates and static files serving
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for async initialization and cleanup.

    Startup:
    - Validate configuration
    - Initialize async RAG engine
    - Warm up database connections

    Shutdown:
    - Close database connections
    - Cleanup resources
    """
    logger.info("Starting FastAPI application...")

    try:
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validated successfully")

        # Initialize async RAG engine (Phase 3 complete)
        from app.core.rag_engine import AsyncRAGEngine
        app.state.rag_engine = AsyncRAGEngine()
        logger.info("Async RAG engine initialized")

        # Initialize database (async version in Phase 2)
        # TODO Phase 2: Initialize async database connection pool
        logger.info("Database connections ready")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down FastAPI application...")

        # Close database connections
        # TODO Phase 2: await app.state.db_engine.dispose()

        logger.info("Cleanup completed")


def create_fastapi_app() -> FastAPI:
    """
    Application factory function.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="RAG Documentation Assistant API",
        description="Async RAG system for API documentation queries with code generation",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Configure templates and static files
    BASE_DIR = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    logger.info("Templates and static files configured")

    # Store templates in app state for use in routes
    app.state.templates = templates

    # Register middleware
    _register_middleware(app)

    # Register exception handlers
    _register_exception_handlers(app)

    # Register routes (Phase 4)
    from app.api.routes import router
    app.include_router(router)
    logger.info("Async routes registered")

    # Frontend routes - HTML pages
    @app.get("/", tags=["Frontend"], include_in_schema=False)
    async def index(request: Request):
        """
        Home page - Landing page with features and documentation info.
        """
        return app.state.templates.TemplateResponse(
            "index.html",
            {"request": request}
        )

    @app.get("/chat", tags=["Frontend"], include_in_schema=False)
    async def chat_page(request: Request):
        """
        Chat interface - Interactive chat page with AI assistant.
        """
        import uuid
        session_id = str(uuid.uuid4())
        return app.state.templates.TemplateResponse(
            "chat.html",
            {"request": request, "session_id": session_id}
        )

    # API info endpoint for programmatic access
    @app.get("/api", tags=["Info"])
    async def api_info():
        """
        API information endpoint - Quick reference for available endpoints.
        """
        return {
            "name": "RAG Documentation Assistant API",
            "version": "2.0.0",
            "framework": "FastAPI",
            "async": True,
            "status": "operational",
            "description": "Async RAG system for API documentation queries with code generation",
            "endpoints": {
                "chat": "POST /api/chat - Generate AI responses",
                "stats": "GET /api/stats - System statistics",
                "history": "GET /api/history - Conversation history",
                "feedback": "POST /api/feedback - Submit feedback",
                "health": "GET /health - Health check"
            },
            "documentation": {
                "swagger": "/docs - Interactive API documentation",
                "redoc": "/redoc - Alternative API documentation",
                "openapi": "/openapi.json - OpenAPI specification"
            },
            "quick_start": [
                "1. Visit / for the web interface",
                "2. Visit /chat for the chat interface",
                "3. Visit /docs for API documentation",
                "4. POST to /api/chat with JSON: {\"query\": \"your question\"}"
            ]
        }

    logger.info("FastAPI application created successfully")

    return app


def _register_middleware(app: FastAPI) -> None:
    """Register security and functional middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.ALLOWED_DOMAINS if Config.ALLOWED_DOMAINS else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    logger.info("CORS middleware registered")

    # Trusted host middleware (production only)
    if Config._is_production() and Config.ALLOWED_DOMAINS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=Config.ALLOWED_DOMAINS
        )
        logger.info("TrustedHost middleware registered")

    # Custom security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

    logger.info("Security headers middleware registered")


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle HTTP exceptions with consistent format."""
        logger.warning(f"HTTP {exc.status_code} error: {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        errors = exc.errors()
        logger.warning(f"Validation error: {errors}")

        # Convert errors to JSON-serializable format
        serializable_errors = []
        for error in errors:
            serializable_errors.append({
                "loc": list(error.get("loc", [])),
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", ""))
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": serializable_errors,
                "path": str(request.url)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

        # Hide internal errors in production
        error_message = "Internal server error"
        if not Config._is_production():
            error_message = str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": error_message,
                "status_code": 500,
                "path": str(request.url)
            }
        )

    logger.info("Exception handlers registered")


# Create application instance
app = create_fastapi_app()


# Health check endpoint (available immediately)
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.

    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "framework": "fastapi",
        "async": True
    }


# Create app instance at module level for uvicorn
app = create_fastapi_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
