"""
Stream Handler - Server-Sent Events (SSE) Streaming

Handles streaming of agent responses token-by-token using SSE for real-time UX.
"""

from typing import AsyncGenerator
import asyncio
import json
import logging
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for streaming agent responses.

    Captures LLM tokens as they're generated and provides async iteration.
    """

    def __init__(self):
        """Initialize streaming callback handler"""
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Called when LLM generates a new token.

        Args:
            token: The newly generated token
            **kwargs: Additional arguments
        """
        # Put token in queue for streaming
        asyncio.create_task(self.queue.put(token))
        logger.debug(f"New token: {token}")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """
        Called when LLM starts.

        Args:
            serialized: Serialized LLM config
            prompts: Input prompts
            **kwargs: Additional arguments
        """
        logger.info("LLM streaming started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Called when LLM finishes.

        Args:
            response: LLM response
            **kwargs: Additional arguments
        """
        self.done = True
        asyncio.create_task(self.queue.put(None))  # Sentinel value
        logger.info("LLM streaming complete")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """
        Called when LLM encounters an error.

        Args:
            error: The exception
            **kwargs: Additional arguments
        """
        self.done = True
        error_msg = f"ERROR: {str(error)}"
        asyncio.create_task(self.queue.put(error_msg))
        asyncio.create_task(self.queue.put(None))  # Sentinel
        logger.error(f"LLM streaming error: {error}")

    async def aiter(self) -> AsyncGenerator[str, None]:
        """
        Async iterator for tokens.

        Yields:
            Token strings as they're generated
        """
        while not self.done or not self.queue.empty():
            try:
                # Wait for token with timeout to prevent hanging
                token = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=30.0  # 30 second timeout
                )

                if token is None:
                    # Sentinel value - stream is done
                    break

                yield token

            except asyncio.TimeoutError:
                logger.warning("Stream timeout waiting for token")
                break
            except Exception as e:
                logger.error(f"Error in stream iteration: {e}")
                break


async def stream_agent_response(
    agent,
    query: str,
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """
    Stream agent response token by token.

    Args:
        agent: DocumentationAgent instance
        query: User query
        session_id: Optional session ID for context

    Yields:
        SSE-formatted strings with tokens

    Example SSE events:
        data: {"token": "Hello"}
        data: {"token": " world"}
        data: {"done": true}
    """
    try:
        logger.info(f"Starting agent stream for query: {query[:50]}...")

        # Create streaming callback
        callback = StreamingCallbackHandler()

        # Enable streaming on LLM
        original_streaming = agent.llm.streaming
        original_callbacks = agent.llm.callbacks

        agent.llm.streaming = True
        agent.llm.callbacks = [callback]

        try:
            # Start agent execution in background
            task = asyncio.create_task(agent.arun(query))

            # Stream tokens as they arrive
            async for token in callback.aiter():
                # Format as SSE event
                event = f"data: {json.dumps({'token': token})}\n\n"
                yield event

            # Wait for agent to complete
            result = await task

            # Send completion event with metadata
            yield f"data: {json.dumps({'done': True, 'sources': result.get('sources', [])})}\n\n"

            logger.info("Agent stream completed successfully")

        finally:
            # Restore original LLM settings
            agent.llm.streaming = original_streaming
            agent.llm.callbacks = original_callbacks

    except Exception as e:
        logger.error(f"Error in agent stream: {e}", exc_info=True)

        # Send error event
        error_event = f"data: {json.dumps({'error': str(e)})}\n\n"
        yield error_event

        # Send done event
        yield f"data: {json.dumps({'done': True})}\n\n"


async def stream_text_response(text: str, chunk_size: int = 10) -> AsyncGenerator[str, None]:
    """
    Stream pre-generated text response.

    Useful for testing or when response is already generated.

    Args:
        text: Text to stream
        chunk_size: Characters per chunk

    Yields:
        SSE-formatted strings with text chunks
    """
    try:
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]

            # Send chunk
            event = f"data: {json.dumps({'token': chunk})}\n\n"
            yield event

            # Small delay to simulate streaming
            await asyncio.sleep(0.05)

        # Send completion
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        logger.error(f"Error streaming text: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def stream_with_progress(
    generator: AsyncGenerator,
    progress_callback=None
) -> AsyncGenerator[str, None]:
    """
    Stream with progress tracking.

    Args:
        generator: Async generator to wrap
        progress_callback: Optional callback for progress updates

    Yields:
        SSE events from wrapped generator plus progress events
    """
    token_count = 0

    async for event in generator:
        yield event

        # Track progress
        if '"token"' in event:
            token_count += 1

            # Send progress update every 10 tokens
            if progress_callback and token_count % 10 == 0:
                progress_callback(token_count)

            # Or send as SSE event
            if token_count % 20 == 0:
                progress_event = f"data: {json.dumps({'progress': token_count})}\n\n"
                yield progress_event
