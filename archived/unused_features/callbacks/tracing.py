"""
Tracing Callbacks - Custom Callbacks for Metrics Collection

Custom LangChain callback handlers for collecting metrics on LLM calls,
tokens, costs, and performance.
"""

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for collecting LLM metrics.

    Tracks:
    - Number of LLM calls
    - Token usage (input/output/total)
    - Estimated costs
    - Latency per call
    """

    # Pricing (per 1K tokens) - Update based on current pricing
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'default': {'input': 0.01, 'output': 0.03}
    }

    def __init__(self):
        """Initialize metrics collector"""
        super().__init__()

        # Counters
        self.llm_calls = 0
        self.tool_calls = 0
        self.chain_calls = 0

        # Token tracking
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # Cost tracking
        self.total_cost = 0.0

        # Timing
        self.start_times = {}
        self.call_latencies = []

        # Errors
        self.errors = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """
        Called when LLM starts running.

        Args:
            serialized: Serialized LLM configuration
            prompts: List of prompts being sent
            **kwargs: Additional arguments
        """
        self.llm_calls += 1
        run_id = kwargs.get('run_id', f'llm_{self.llm_calls}')
        self.start_times[run_id] = time.time()

        model = serialized.get('name', 'unknown')
        logger.info(f"LLM call #{self.llm_calls} started (model: {model}, prompts: {len(prompts)})")

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any
    ) -> None:
        """
        Called when LLM finishes running.

        Args:
            response: LLM response object
            **kwargs: Additional arguments
        """
        run_id = kwargs.get('run_id', f'llm_{self.llm_calls}')

        # Calculate latency
        if run_id in self.start_times:
            elapsed = time.time() - self.start_times[run_id]
            self.call_latencies.append(elapsed)
            del self.start_times[run_id]
        else:
            elapsed = 0.0

        # Extract token usage
        tokens = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            tokens = response.llm_output.get('token_usage', {})
        elif hasattr(response, 'token_usage'):
            tokens = response.token_usage

        prompt_tokens = tokens.get('prompt_tokens', 0)
        completion_tokens = tokens.get('completion_tokens', 0)
        total_tokens = tokens.get('total_tokens', prompt_tokens + completion_tokens)

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        # Estimate cost
        model = self._extract_model_name(response)
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        self.total_cost += cost

        logger.info(
            f"LLM call completed: {elapsed:.2f}s, "
            f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}), "
            f"Cost: ${cost:.4f}"
        )

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """
        Called when LLM encounters an error.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        run_id = kwargs.get('run_id', 'unknown')
        self.errors.append({
            'type': 'llm',
            'run_id': run_id,
            'error': str(error)
        })

        logger.error(f"LLM error in run {run_id}: {error}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """
        Called when a tool starts running.

        Args:
            serialized: Serialized tool configuration
            input_str: Input to the tool
            **kwargs: Additional arguments
        """
        self.tool_calls += 1
        tool_name = serialized.get('name', 'unknown')

        logger.info(f"Tool #{self.tool_calls} started: {tool_name}")

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """
        Called when a tool finishes running.

        Args:
            output: Tool output
            **kwargs: Additional arguments
        """
        logger.info(f"Tool completed")

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """
        Called when a tool encounters an error.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        run_id = kwargs.get('run_id', 'unknown')
        self.errors.append({
            'type': 'tool',
            'run_id': run_id,
            'error': str(error)
        })

        logger.error(f"Tool error in run {run_id}: {error}")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called when a chain starts running.

        Args:
            serialized: Serialized chain configuration
            inputs: Chain inputs
            **kwargs: Additional arguments
        """
        self.chain_calls += 1
        chain_name = serialized.get('name', 'unknown')

        logger.info(f"Chain #{self.chain_calls} started: {chain_name}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected metrics.

        Returns:
            Dictionary with all collected metrics
        """
        avg_latency = (
            sum(self.call_latencies) / len(self.call_latencies)
            if self.call_latencies else 0.0
        )

        max_latency = max(self.call_latencies) if self.call_latencies else 0.0
        min_latency = min(self.call_latencies) if self.call_latencies else 0.0

        return {
            'calls': {
                'llm': self.llm_calls,
                'tool': self.tool_calls,
                'chain': self.chain_calls,
                'total': self.llm_calls + self.tool_calls + self.chain_calls
            },
            'tokens': {
                'prompt': self.prompt_tokens,
                'completion': self.completion_tokens,
                'total': self.total_tokens
            },
            'cost': {
                'total': round(self.total_cost, 4),
                'per_call': round(self.total_cost / self.llm_calls, 4) if self.llm_calls > 0 else 0.0
            },
            'latency': {
                'avg': round(avg_latency, 2),
                'min': round(min_latency, 2),
                'max': round(max_latency, 2),
                'total': round(sum(self.call_latencies), 2)
            },
            'errors': {
                'count': len(self.errors),
                'details': self.errors
            }
        }

    def reset(self) -> None:
        """Reset all metrics counters"""
        self.llm_calls = 0
        self.tool_calls = 0
        self.chain_calls = 0

        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.total_cost = 0.0

        self.start_times = {}
        self.call_latencies = []

        self.errors = []

        logger.info("Metrics reset")

    def _extract_model_name(self, response: Any) -> str:
        """
        Extract model name from LLM response.

        Args:
            response: LLM response object

        Returns:
            Model name string
        """
        # Try to get model from response
        if hasattr(response, 'llm_output') and response.llm_output:
            model = response.llm_output.get('model_name', 'default')
            return model

        # Default
        return 'default'

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate estimated cost for LLM call.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        # Get pricing for model (or use default)
        pricing = self.PRICING.get(model, self.PRICING['default'])

        # Calculate cost (pricing is per 1K tokens)
        cost = (
            (prompt_tokens * pricing['input'] / 1000) +
            (completion_tokens * pricing['output'] / 1000)
        )

        return cost


class DebugCallbackHandler(BaseCallbackHandler):
    """
    Debug callback handler for detailed logging.

    Useful for development and troubleshooting.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize debug handler.

        Args:
            verbose: If True, log detailed information
        """
        super().__init__()
        self.verbose = verbose

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Log LLM start"""
        if self.verbose:
            logger.debug(f"[LLM START] Model: {serialized.get('name')}")
            logger.debug(f"[LLM START] Prompts: {prompts[0][:100]}..." if prompts else "[LLM START] No prompts")

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any
    ) -> None:
        """Log LLM end"""
        if self.verbose:
            logger.debug(f"[LLM END] Response received")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Log tool start"""
        if self.verbose:
            tool_name = serialized.get('name', 'unknown')
            logger.debug(f"[TOOL START] {tool_name}: {input_str[:100]}...")

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """Log tool end"""
        if self.verbose:
            logger.debug(f"[TOOL END] Output: {output[:100]}...")
