# DEV_2: Conversation Memory

**Desenvolvedor:** DEV_2
**Fase:** 9B - Conversation Memory
**Prioridade:** ⭐⭐⭐ CRÍTICA
**Estimativa:** 2-3 horas
**Dependências:** Nenhuma (pode começar imediatamente)

---

## 🎯 Objetivo

Implementar **sistema de memória conversacional** com múltiplas estratégias para manter contexto de longo prazo nas conversas.

### O que você vai construir:

```
User Query + Memory → Agent → Response
         ↓
   Store in Memory
```

**Estratégias:**
1. **Buffer Window** - Últimas N mensagens
2. **Summary** - Resumo de conversas antigas
3. **Token Buffer** - Limite por tokens

---

## 📦 Entregas

### Arquivos a criar:

1. **`conversation_memory.py`** - Memory managers
2. **`models_memory.py`** - Database models
3. **`migrations/add_memory_tables.py`** - Alembic migration
4. **`routes_memory.py`** - Memory endpoints
5. **`schemas_memory.py`** - Pydantic models
6. **`test_memory.py`** - Memory tests

### Arquivos a modificar:

- `fastapi_app.py` - Registrar memory router
- `models_async.py` - Adicionar relationship (se necessário)

---

## 🏗️ Arquitetura

```python
┌────────────────────────────────────┐
│    Memory Management Layer         │
│  ┌──────────────────────────────┐  │
│  │  ConversationMemoryManager   │  │
│  └────────┬─────────────────────┘  │
│           │                         │
│  ┌────────▼─────────────────────┐  │
│  │    Strategy Selection        │  │
│  │  (buffer | summary | token)  │  │
│  └────────┬─────────────────────┘  │
└───────────┼──────────────────────┘
            │
    ┌───────┼───────┐
    │       │       │
┌───▼───┐ ┌─▼────┐ ┌─▼────┐
│Buffer │ │Summary│ │Token │
│Window │ │Memory │ │Buffer│
└───┬───┘ └───┬──┘ └───┬──┘
    │         │        │
    └─────────▼────────┘
         PostgreSQL
      (memory table)
```

---

## 📝 Especificação Detalhada

### 1. `models_memory.py`

```python
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from models_async import Base

class ConversationMemory(Base):
    """
    Stores conversation messages for memory management.

    Supports different memory strategies:
    - buffer_window: Keep last N messages
    - summary: Summarize old messages
    - token_buffer: Keep messages within token limit
    """
    __tablename__ = "conversation_memory"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), nullable=False, index=True)

    # Message content
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    tokens = Column(Integer, default=0)  # Estimated token count

    # Metadata
    message_index = Column(Integer, nullable=False)  # Order in conversation
    memory_strategy = Column(String(50), default='buffer_window')
    is_summarized = Column(Boolean, default=False)
    summary_of = Column(String(500), nullable=True)  # IDs of messages this summarizes

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes for fast retrieval
    __table_args__ = (
        Index('idx_session_strategy', 'session_id', 'memory_strategy'),
        Index('idx_session_index', 'session_id', 'message_index'),
    )


class MemoryConfiguration(Base):
    """
    User/session-specific memory configuration.
    """
    __tablename__ = "memory_configuration"

    session_id = Column(String(36), primary_key=True)

    # Strategy selection
    strategy = Column(String(50), default='buffer_window')

    # Strategy parameters
    buffer_window_size = Column(Integer, default=20)  # For buffer_window
    token_limit = Column(Integer, default=2000)  # For token_buffer
    summary_interval = Column(Integer, default=10)  # For summary (messages before summarizing)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

---

### 2. `conversation_memory.py`

```python
import logging
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime
import tiktoken

from models_memory import ConversationMemory, MemoryConfiguration
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
from config import Config

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Manages conversation memory with multiple strategies.
    """

    def __init__(self, session_id: str, db: AsyncSession):
        """
        Initialize memory manager for a session.

        Args:
            session_id: Session identifier
            db: Async database session
        """
        self.session_id = session_id
        self.db = db
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def get_memory_config(self) -> MemoryConfiguration:
        """Get or create memory configuration for session"""
        result = await self.db.execute(
            select(MemoryConfiguration).where(
                MemoryConfiguration.session_id == self.session_id
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            config = MemoryConfiguration(
                session_id=self.session_id,
                strategy=Config.MEMORY_STRATEGY,
                buffer_window_size=Config.MEMORY_MAX_MESSAGES,
                token_limit=2000,
                summary_interval=10
            )
            self.db.add(config)
            await self.db.commit()

        return config

    async def add_message(self, role: str, content: str) -> None:
        """
        Add message to memory.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        # Get message index (next in sequence)
        result = await self.db.execute(
            select(func.max(ConversationMemory.message_index))
            .where(ConversationMemory.session_id == self.session_id)
        )
        max_index = result.scalar() or 0
        next_index = max_index + 1

        # Count tokens
        tokens = len(self.encoding.encode(content))

        # Get config
        config = await self.get_memory_config()

        # Create memory entry
        memory = ConversationMemory(
            session_id=self.session_id,
            role=role,
            content=content,
            tokens=tokens,
            message_index=next_index,
            memory_strategy=config.strategy
        )

        self.db.add(memory)
        await self.db.commit()

        logger.info(f"Added {role} message to memory (session: {self.session_id}, index: {next_index})")

        # Apply strategy-specific cleanup
        await self._apply_strategy(config)

    async def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation messages based on memory strategy.

        Args:
            limit: Optional message limit (overrides strategy)

        Returns:
            List of messages in format [{'role': 'user', 'content': '...'}, ...]
        """
        config = await self.get_memory_config()

        if config.strategy == 'buffer_window':
            return await self._get_buffer_window_messages(
                limit or config.buffer_window_size
            )
        elif config.strategy == 'summary':
            return await self._get_summary_messages()
        elif config.strategy == 'token_buffer':
            return await self._get_token_buffer_messages(config.token_limit)
        else:
            logger.warning(f"Unknown strategy: {config.strategy}, using buffer_window")
            return await self._get_buffer_window_messages(20)

    async def _get_buffer_window_messages(self, window_size: int) -> List[Dict]:
        """Get last N messages"""
        result = await self.db.execute(
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.session_id == self.session_id,
                    ConversationMemory.is_summarized == False
                )
            )
            .order_by(ConversationMemory.message_index.desc())
            .limit(window_size)
        )

        messages = result.scalars().all()

        # Reverse to chronological order
        return [
            {'role': msg.role, 'content': msg.content}
            for msg in reversed(messages)
        ]

    async def _get_summary_messages(self) -> List[Dict]:
        """Get messages with summaries"""
        # Get all messages (including summaries)
        result = await self.db.execute(
            select(ConversationMemory)
            .where(ConversationMemory.session_id == self.session_id)
            .order_by(ConversationMemory.message_index.asc())
        )

        messages = result.scalars().all()

        formatted = []
        for msg in messages:
            if msg.is_summarized:
                # This is a summary
                formatted.append({
                    'role': 'system',
                    'content': f"[Summary of previous messages]: {msg.content}"
                })
            else:
                formatted.append({
                    'role': msg.role,
                    'content': msg.content
                })

        return formatted

    async def _get_token_buffer_messages(self, token_limit: int) -> List[Dict]:
        """Get messages within token limit (most recent first)"""
        result = await self.db.execute(
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.session_id == self.session_id,
                    ConversationMemory.is_summarized == False
                )
            )
            .order_by(ConversationMemory.message_index.desc())
        )

        messages = result.scalars().all()

        # Accumulate messages until token limit
        selected = []
        total_tokens = 0

        for msg in messages:
            if total_tokens + msg.tokens > token_limit:
                break
            selected.append(msg)
            total_tokens += msg.tokens

        # Reverse to chronological order
        return [
            {'role': msg.role, 'content': msg.content}
            for msg in reversed(selected)
        ]

    async def _apply_strategy(self, config: MemoryConfiguration) -> None:
        """Apply memory strategy cleanup/summarization"""

        if config.strategy == 'summary':
            # Check if we should create a summary
            count_result = await self.db.execute(
                select(func.count(ConversationMemory.id))
                .where(
                    and_(
                        ConversationMemory.session_id == self.session_id,
                        ConversationMemory.is_summarized == False
                    )
                )
            )
            message_count = count_result.scalar()

            if message_count >= config.summary_interval:
                await self._create_summary(config.summary_interval)

    async def _create_summary(self, message_count: int) -> None:
        """Create summary of old messages"""
        # Get messages to summarize
        result = await self.db.execute(
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.session_id == self.session_id,
                    ConversationMemory.is_summarized == False
                )
            )
            .order_by(ConversationMemory.message_index.asc())
            .limit(message_count)
        )

        messages = result.scalars().all()

        if not messages:
            return

        # Create conversation text
        conversation = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])

        # Use LLM to summarize
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        summary_prompt = f"Summarize the following conversation concisely:\n\n{conversation}"

        summary = await llm.ainvoke(summary_prompt)
        summary_text = summary.content

        # Create summary memory entry
        summary_memory = ConversationMemory(
            session_id=self.session_id,
            role='system',
            content=summary_text,
            tokens=len(self.encoding.encode(summary_text)),
            message_index=-1,  # Special index for summaries
            memory_strategy='summary',
            is_summarized=True,
            summary_of=",".join([msg.id for msg in messages])
        )

        self.db.add(summary_memory)

        # Mark original messages as summarized
        for msg in messages:
            msg.is_summarized = True

        await self.db.commit()

        logger.info(f"Created summary for {len(messages)} messages (session: {self.session_id})")

    async def clear_memory(self) -> None:
        """Clear all memory for this session"""
        await self.db.execute(
            delete(ConversationMemory).where(
                ConversationMemory.session_id == self.session_id
            )
        )
        await self.db.commit()
        logger.info(f"Cleared memory for session: {self.session_id}")

    async def get_statistics(self) -> Dict:
        """Get memory statistics"""
        # Total messages
        count_result = await self.db.execute(
            select(func.count(ConversationMemory.id))
            .where(ConversationMemory.session_id == self.session_id)
        )
        total_messages = count_result.scalar()

        # Total tokens
        tokens_result = await self.db.execute(
            select(func.sum(ConversationMemory.tokens))
            .where(ConversationMemory.session_id == self.session_id)
        )
        total_tokens = tokens_result.scalar() or 0

        # Strategy
        config = await self.get_memory_config()

        return {
            'total_messages': total_messages,
            'total_tokens': total_tokens,
            'strategy': config.strategy,
            'config': {
                'buffer_window_size': config.buffer_window_size,
                'token_limit': config.token_limit,
                'summary_interval': config.summary_interval
            }
        }
```

---

### 3. `routes_memory.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from conversation_memory import ConversationMemoryManager
from schemas_memory import (
    MemoryConfigRequest,
    MemoryConfigResponse,
    MessageHistoryResponse,
    MemoryStatsResponse
)
from database_async import get_async_db
from dependencies import get_session_id, validate_rate_limit

router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.get("/config", response_model=MemoryConfigResponse)
async def get_memory_config(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db)
):
    """Get current memory configuration"""
    manager = ConversationMemoryManager(session_id, db)
    config = await manager.get_memory_config()

    return MemoryConfigResponse(
        session_id=session_id,
        strategy=config.strategy,
        buffer_window_size=config.buffer_window_size,
        token_limit=config.token_limit,
        summary_interval=config.summary_interval
    )


@router.put("/config", response_model=MemoryConfigResponse)
async def update_memory_config(
    request: MemoryConfigRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db)
):
    """Update memory configuration"""
    manager = ConversationMemoryManager(session_id, db)
    config = await manager.get_memory_config()

    # Update config
    if request.strategy:
        config.strategy = request.strategy
    if request.buffer_window_size:
        config.buffer_window_size = request.buffer_window_size
    if request.token_limit:
        config.token_limit = request.token_limit
    if request.summary_interval:
        config.summary_interval = request.summary_interval

    await db.commit()

    return MemoryConfigResponse(
        session_id=session_id,
        strategy=config.strategy,
        buffer_window_size=config.buffer_window_size,
        token_limit=config.token_limit,
        summary_interval=config.summary_interval
    )


@router.get("/messages", response_model=MessageHistoryResponse)
async def get_memory_messages(
    limit: int = 20,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db)
):
    """Get conversation messages from memory"""
    manager = ConversationMemoryManager(session_id, db)
    messages = await manager.get_messages(limit=limit)

    return MessageHistoryResponse(
        session_id=session_id,
        messages=messages,
        count=len(messages)
    )


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db)
):
    """Get memory statistics"""
    manager = ConversationMemoryManager(session_id, db)
    stats = await manager.get_statistics()

    return MemoryStatsResponse(**stats)


@router.delete("/clear")
async def clear_memory(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db),
    _rate_limit: None = Depends(validate_rate_limit)
):
    """Clear all memory for current session"""
    manager = ConversationMemoryManager(session_id, db)
    await manager.clear_memory()

    return {"message": "Memory cleared successfully", "session_id": session_id}
```

---

### 4. `schemas_memory.py`

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class MemoryConfigRequest(BaseModel):
    """Request to update memory configuration"""
    strategy: Optional[str] = Field(None, description="Memory strategy: buffer_window | summary | token_buffer")
    buffer_window_size: Optional[int] = Field(None, ge=1, le=100)
    token_limit: Optional[int] = Field(None, ge=100, le=10000)
    summary_interval: Optional[int] = Field(None, ge=5, le=50)

class MemoryConfigResponse(BaseModel):
    """Memory configuration"""
    session_id: str
    strategy: str
    buffer_window_size: int
    token_limit: int
    summary_interval: int

class MessageHistoryResponse(BaseModel):
    """Memory messages"""
    session_id: str
    messages: List[Dict]
    count: int

class MemoryStatsResponse(BaseModel):
    """Memory statistics"""
    total_messages: int
    total_tokens: int
    strategy: str
    config: Dict
```

---

### 5. `migrations/add_memory_tables.py`

```bash
# Run this to create migration:
alembic revision --autogenerate -m "Add conversation memory tables"

# Then edit the generated migration file to ensure:
# - conversation_memory table created
# - memory_configuration table created
# - Indexes created
```

---

### 6. `test_memory.py`

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from conversation_memory import ConversationMemoryManager
from models_memory import Base, ConversationMemory


@pytest.fixture
async def db_session():
    """Create test database session"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_add_message(db_session):
    """Test adding message to memory"""
    manager = ConversationMemoryManager("test-session", db_session)

    await manager.add_message("user", "Hello!")
    await manager.add_message("assistant", "Hi there!")

    messages = await manager.get_messages()

    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == 'Hello!'


@pytest.mark.asyncio
async def test_buffer_window_strategy(db_session):
    """Test buffer window strategy"""
    manager = ConversationMemoryManager("test-session", db_session)

    # Add 30 messages
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        await manager.add_message(role, f"Message {i}")

    # Get with window size 10
    messages = await manager._get_buffer_window_messages(10)

    assert len(messages) == 10
    # Should be last 10 messages
    assert messages[-1]['content'] == "Message 29"


@pytest.mark.asyncio
async def test_memory_statistics(db_session):
    """Test memory statistics"""
    manager = ConversationMemoryManager("test-session", db_session)

    await manager.add_message("user", "Test message")

    stats = await manager.get_statistics()

    assert stats['total_messages'] == 1
    assert stats['total_tokens'] > 0
    assert stats['strategy'] in ['buffer_window', 'summary', 'token_buffer']


@pytest.mark.asyncio
async def test_clear_memory(db_session):
    """Test clearing memory"""
    manager = ConversationMemoryManager("test-session", db_session)

    await manager.add_message("user", "Test")
    await manager.clear_memory()

    messages = await manager.get_messages()
    assert len(messages) == 0
```

---

## ✅ Critérios de Aceitação

- [ ] Todas as 3 estratégias funcionando
- [ ] Database migration criada
- [ ] Endpoints de memory funcionando
- [ ] Testes passando (>80% coverage)
- [ ] Integração com agent (DEV_1)

---

**Boa sorte, DEV_2! 🚀**
