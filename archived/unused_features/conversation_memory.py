"""
Conversation Memory Manager - Multi-Strategy Memory Management

Manages conversation history with multiple memory strategies:
- Buffer Window: Keep last N messages
- Summary: Summarize old conversations
- Token Buffer: Keep messages within token limit
"""

import logging
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, delete
from datetime import datetime
import tiktoken

from models_memory import ConversationMemory, MemoryConfiguration
from langchain_openai import ChatOpenAI
from config import Config

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Manages conversation memory with multiple strategies.

    Strategies:
    1. buffer_window: Keep last N messages (simple, fast)
    2. summary: Periodically summarize old messages (saves context)
    3. token_buffer: Keep messages within token limit (precise)
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

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding: {e}, using approximate counting")
            self.encoding = None

    async def get_memory_config(self) -> MemoryConfiguration:
        """
        Get or create memory configuration for session.

        Returns:
            MemoryConfiguration instance
        """
        result = await self.db.execute(
            select(MemoryConfiguration).where(
                MemoryConfiguration.session_id == self.session_id
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            # Create default configuration
            config = MemoryConfiguration(
                session_id=self.session_id,
                strategy=getattr(Config, 'MEMORY_STRATEGY', 'buffer_window'),
                buffer_window_size=getattr(Config, 'MEMORY_MAX_MESSAGES', 20),
                token_limit=2000,
                summary_interval=10
            )
            self.db.add(config)
            await self.db.commit()
            await self.db.refresh(config)

            logger.info(f"Created memory config for session {self.session_id}")

        return config

    async def add_message(self, role: str, content: str) -> None:
        """
        Add message to memory.

        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content

        Raises:
            ValueError: If role is invalid
        """
        if role not in ['user', 'assistant', 'system']:
            raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")

        # Get message index (next in sequence)
        result = await self.db.execute(
            select(func.max(ConversationMemory.message_index))
            .where(ConversationMemory.session_id == self.session_id)
        )
        max_index = result.scalar() or 0
        next_index = max_index + 1

        # Count tokens
        tokens = self._count_tokens(content)

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

        logger.info(f"Added {role} message to memory (session: {self.session_id}, index: {next_index}, tokens: {tokens})")

        # Apply strategy-specific cleanup/processing
        await self._apply_strategy(config)

    async def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation messages based on memory strategy.

        Args:
            limit: Optional message limit (overrides strategy default)

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
        """
        Get last N messages (Buffer Window strategy).

        Args:
            window_size: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
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
        """
        Get messages with summaries (Summary strategy).

        Returns messages including summary placeholders.

        Returns:
            List of messages with summaries
        """
        # Get all messages (including summaries)
        result = await self.db.execute(
            select(ConversationMemory)
            .where(ConversationMemory.session_id == self.session_id)
            .order_by(ConversationMemory.message_index.asc())
        )

        messages = result.scalars().all()

        formatted = []
        for msg in messages:
            if msg.is_summarized and msg.role == 'system':
                # This is a summary
                formatted.append({
                    'role': 'system',
                    'content': f"[Summary of previous messages]: {msg.content}"
                })
            elif not msg.is_summarized:
                # Regular message
                formatted.append({
                    'role': msg.role,
                    'content': msg.content
                })

        return formatted

    async def _get_token_buffer_messages(self, token_limit: int) -> List[Dict]:
        """
        Get messages within token limit (Token Buffer strategy).

        Retrieves most recent messages that fit within token limit.

        Args:
            token_limit: Maximum tokens to include

        Returns:
            List of messages within token limit
        """
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
        """
        Apply memory strategy cleanup/summarization.

        Args:
            config: Memory configuration
        """
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
        """
        Create summary of old messages.

        Uses LLM to summarize oldest unsummarized messages.

        Args:
            message_count: Number of messages to summarize
        """
        # Get messages to summarize (oldest first)
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

        try:
            # Use LLM to summarize
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=Config.OPENAI_API_KEY
            )

            summary_prompt = f"Summarize the following conversation concisely, preserving key information:\n\n{conversation}"

            response = await llm.ainvoke(summary_prompt)
            summary_text = response.content

            # Count summary tokens
            summary_tokens = self._count_tokens(summary_text)

            # Create summary memory entry
            summary_memory = ConversationMemory(
                session_id=self.session_id,
                role='system',
                content=summary_text,
                tokens=summary_tokens,
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

        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            # Don't raise - summarization is optional

    async def clear_memory(self) -> None:
        """
        Clear all memory for this session.

        Deletes all conversation messages.
        """
        await self.db.execute(
            delete(ConversationMemory).where(
                ConversationMemory.session_id == self.session_id
            )
        )
        await self.db.commit()

        logger.info(f"Cleared memory for session: {self.session_id}")

    async def get_statistics(self) -> Dict:
        """
        Get memory statistics for this session.

        Returns:
            Dictionary with statistics
        """
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

        # Get configuration
        config = await self.get_memory_config()

        return {
            'total_messages': total_messages,
            'total_tokens': int(total_tokens),
            'strategy': config.strategy,
            'config': {
                'buffer_window_size': config.buffer_window_size,
                'token_limit': config.token_limit,
                'summary_interval': config.summary_interval
            }
        }

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token counting error: {e}")

        # Fallback: approximate as ~4 characters per token
        return len(text) // 4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_memory_manager(
    session_id: str,
    db: AsyncSession
) -> ConversationMemoryManager:
    """
    Factory function to get memory manager instance.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        ConversationMemoryManager instance
    """
    return ConversationMemoryManager(session_id, db)
