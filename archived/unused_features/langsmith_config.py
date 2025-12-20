"""
LangSmith Configuration - Observability and Tracing Setup

Configures LangSmith for tracing, debugging, and monitoring LLM calls.

Setup:
1. Create account at https://smith.langchain.com
2. Create project "rag-documentation-assistant"
3. Get API key
4. Set environment variables:
   - LANGSMITH_API_KEY=your-key
   - LANGSMITH_PROJECT=rag-documentation-assistant
   - LANGSMITH_TRACING=true
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LangSmithConfig:
    """
    LangSmith observability configuration.

    Provides centralized configuration for LangSmith tracing and monitoring.
    """

    @staticmethod
    def is_enabled() -> bool:
        """
        Check if LangSmith tracing is enabled.

        Returns:
            True if LangSmith is enabled, False otherwise
        """
        enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

        if enabled:
            # Verify API key is set
            if not os.getenv("LANGSMITH_API_KEY"):
                logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY not set. Disabling tracing.")
                return False

        return enabled

    @staticmethod
    def get_client():
        """
        Get LangSmith client for API access.

        Returns:
            LangSmith Client instance or None if disabled
        """
        if not LangSmithConfig.is_enabled():
            return None

        try:
            from langsmith import Client

            client = Client(
                api_key=os.getenv("LANGSMITH_API_KEY"),
                api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            )

            logger.info("LangSmith client initialized")
            return client

        except ImportError:
            logger.warning("langsmith package not installed. Install with: pip install langsmith")
            return None
        except Exception as e:
            logger.error(f"Error initializing LangSmith client: {e}")
            return None

    @staticmethod
    def get_tracer(run_name: str = "default"):
        """
        Get LangChain tracer for callbacks.

        Args:
            run_name: Name for the trace run

        Returns:
            LangChainTracer instance or None if disabled
        """
        if not LangSmithConfig.is_enabled():
            return None

        try:
            from langchain.callbacks.tracers import LangChainTracer

            client = LangSmithConfig.get_client()
            if not client:
                return None

            tracer = LangChainTracer(
                project_name=os.getenv("LANGSMITH_PROJECT", "rag-assistant"),
                client=client
            )

            logger.info(f"LangChain tracer created for run: {run_name}")
            return tracer

        except ImportError:
            logger.warning("LangChain tracer not available. Install langchain with: pip install langchain")
            return None
        except Exception as e:
            logger.error(f"Error creating tracer: {e}")
            return None

    @staticmethod
    def get_callbacks(run_name: str = "agent_run") -> List:
        """
        Get callback list for agent/chain execution.

        Args:
            run_name: Name for the trace run

        Returns:
            List of callback handlers (empty list if disabled)
        """
        if not LangSmithConfig.is_enabled():
            return []

        try:
            tracer = LangSmithConfig.get_tracer(run_name)
            callbacks = [tracer] if tracer else []

            # Add custom metrics callback
            try:
                from callbacks.tracing import MetricsCallbackHandler
                callbacks.append(MetricsCallbackHandler())
            except ImportError:
                logger.warning("MetricsCallbackHandler not available")

            logger.info(f"Created {len(callbacks)} callbacks for run: {run_name}")
            return callbacks

        except Exception as e:
            logger.error(f"Error getting callbacks: {e}")
            return []

    @staticmethod
    def get_environment_config() -> dict:
        """
        Get current LangSmith environment configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            'enabled': LangSmithConfig.is_enabled(),
            'project': os.getenv("LANGSMITH_PROJECT", "rag-assistant"),
            'endpoint': os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            'api_key_set': bool(os.getenv("LANGSMITH_API_KEY")),
            'tracing': os.getenv("LANGSMITH_TRACING", "false")
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_langsmith() -> bool:
    """
    Setup LangSmith tracing for the application.

    Call this during application startup to enable tracing.

    Returns:
        True if setup successful, False otherwise
    """
    if not LangSmithConfig.is_enabled():
        logger.info("LangSmith tracing is disabled (LANGSMITH_TRACING=false)")
        return False

    try:
        # Test connection
        client = LangSmithConfig.get_client()
        if not client:
            logger.warning("Failed to initialize LangSmith client")
            return False

        # Log configuration
        config = LangSmithConfig.get_environment_config()
        logger.info(f"LangSmith tracing enabled: project={config['project']}")

        return True

    except Exception as e:
        logger.error(f"Error setting up LangSmith: {e}")
        return False


def get_trace_url(run_id: str) -> Optional[str]:
    """
    Get URL to view trace in LangSmith dashboard.

    Args:
        run_id: Trace run ID

    Returns:
        URL string or None if unavailable
    """
    if not LangSmithConfig.is_enabled():
        return None

    project = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    # Note: Actual URL format may vary, adjust based on LangSmith dashboard
    return f"https://smith.langchain.com/projects/{project}/runs/{run_id}"
