"""
Async Database Configuration for FastAPI

Provides async SQLAlchemy engine and session management with:
- Automatic URI conversion (sqlite:// → sqlite+aiosqlite://, postgres:// → postgresql+asyncpg://)
- Connection pooling for PostgreSQL
- Session dependency injection for FastAPI
- Compatible with existing Alembic migrations
"""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from config import Config

logger = logging.getLogger(__name__)

# Declarative base for async models
Base = declarative_base()


def get_async_database_uri() -> str:
    """
    Convert sync database URI to async version.

    Conversions:
    - sqlite:/// → sqlite+aiosqlite:///
    - postgresql:// → postgresql+asyncpg://
    - postgresql+psycopg2:// → postgresql+asyncpg://
    """
    uri = Config.get_database_uri()

    # SQLite conversion
    if uri.startswith("sqlite:///"):
        async_uri = uri.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        logger.info("Using SQLite async driver (aiosqlite)")
        return async_uri

    # PostgreSQL conversion
    if uri.startswith("postgresql://"):
        async_uri = uri.replace("postgresql://", "postgresql+asyncpg://", 1)
        logger.info("Using PostgreSQL async driver (asyncpg)")
        return async_uri

    # PostgreSQL psycopg2 conversion
    if uri.startswith("postgresql+psycopg2://"):
        async_uri = uri.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        logger.info("Using PostgreSQL async driver (asyncpg)")
        return async_uri

    # Fallback: return as-is
    logger.warning(f"Unknown database URI scheme, using as-is: {uri[:20]}...")
    return uri


def create_async_db_engine() -> AsyncEngine:
    """
    Create async SQLAlchemy engine with appropriate settings.

    Returns:
        AsyncEngine configured for the database backend
    """
    uri = get_async_database_uri()
    is_sqlite = uri.startswith("sqlite+aiosqlite://")
    is_postgres = uri.startswith("postgresql+asyncpg://")

    # Base engine options
    engine_options = {
        "echo": False,  # Set to True for SQL query logging
        "pool_pre_ping": True,  # Verify connections before using
    }

    if is_sqlite:
        # SQLite-specific settings
        engine_options.update({
            "connect_args": {
                "check_same_thread": False,  # Allow multiple threads
            }
        })
        logger.info("Creating async SQLite engine")

    elif is_postgres:
        # PostgreSQL-specific settings
        engine_options.update({
            "pool_size": Config.DB_POOL_SIZE,
            "max_overflow": Config.DB_MAX_OVERFLOW,
            "pool_recycle": 300,  # Recycle connections every 5 minutes
        })

        # SSL mode for PostgreSQL
        if Config.DB_SSLMODE:
            engine_options["connect_args"] = {"ssl": Config.DB_SSLMODE}

        logger.info(f"Creating async PostgreSQL engine (pool_size={Config.DB_POOL_SIZE})")

    engine = create_async_engine(uri, **engine_options)
    return engine


# Global async engine and session factory
async_engine: AsyncEngine = create_async_db_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autocommit=False,
    autoflush=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database sessions.

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Model))
            ...

    Yields:
        AsyncSession that is automatically closed after request
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_async_db():
    """
    Initialize async database tables.

    Creates all tables defined in Base metadata.
    Should only be called once during application startup.
    """
    async with async_engine.begin() as conn:
        # Import models to register them with Base
        from app.db.models import Conversation, DocumentChunk, RateLimit

        # Create tables (idempotent - won't recreate existing tables)
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Async database tables initialized")


async def dispose_async_db():
    """
    Dispose of async database engine.

    Closes all database connections.
    Should be called during application shutdown.
    """
    await async_engine.dispose()
    logger.info("Async database engine disposed")


# Health check function
async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
