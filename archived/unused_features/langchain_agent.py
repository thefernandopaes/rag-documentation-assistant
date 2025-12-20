"""
LangChain ReAct Agent - Documentation Assistant

Multi-tool orchestration for complex documentation queries.

Capabilities:
- Search documentation using RAG
- Generate code examples
- Validate generated code
- Multi-step reasoning with ReAct pattern
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.rag_tool import RAGSearchTool
from tools.code_generator_tool import CodeGeneratorTool
from tools.validator_tool import CodeValidatorTool
from config import Config

logger = logging.getLogger(__name__)


class DocumentationAgent:
    """
    LangChain ReAct agent for API documentation queries.

    The agent uses the ReAct (Reasoning + Acting) pattern to:
    1. Think about what to do
    2. Act by using tools
    3. Observe results
    4. Repeat until answer is found

    Tools available:
    - documentation_search: Search documentation via RAG
    - code_generator: Generate code examples
    - code_validator: Validate generated code
    """

    def __init__(self):
        """Initialize agent with tools and LLM"""
        logger.info("Initializing DocumentationAgent...")

        # Initialize LLM (GPT-4 for best reasoning, or gpt-3.5-turbo for cost savings)
        self.llm = ChatOpenAI(
            model=Config.AGENT_MODEL if hasattr(Config, 'AGENT_MODEL') else "gpt-4",
            temperature=0.2,
            api_key=Config.OPENAI_API_KEY,
            timeout=60,
            max_retries=2
        )

        # Initialize tools
        self.tools = [
            RAGSearchTool(),
            CodeGeneratorTool(),
            CodeValidatorTool()
        ]

        logger.info(f"Initialized {len(self.tools)} tools")

        # Create LangGraph agent
        self.agent_executor = create_react_agent(self.llm, self.tools)

        # Backward compatibility: alias for tests
        self.agent = self.agent_executor

        logger.info("DocumentationAgent initialized successfully")


    async def arun(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run agent on query (async).

        Args:
            query: User query
            conversation_history: Previous messages for context (optional)

        Returns:
            {
                'output': str,  # Final answer
                'intermediate_steps': List[tuple],  # (action, observation) pairs
                'tool_calls': List[Dict]  # Formatted tool calls
            }
        """
        try:
            logger.info(f"Agent processing query: {query[:100]}...")

            # Prepare input for LangGraph
            messages = [{"role": "user", "content": query}]

            # Run agent
            result = await self.agent_executor.ainvoke({"messages": messages})

            # Extract final message
            if result and 'messages' in result:
                final_message = result['messages'][-1]
                output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                output = "No response generated"

            # Format response
            response = {
                'output': output,
                'intermediate_steps': [],
                'tool_calls': []
            }

            logger.info(f"Agent completed successfully")

            return response

        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            # Return error response instead of raising
            return {
                'output': f"Error: {str(e)}",
                'intermediate_steps': [],
                'tool_calls': []
            }

    def _extract_tool_calls(self, steps: List[tuple]) -> List[Dict]:
        """
        Extract and format tool calls from intermediate steps.

        Args:
            steps: List of (AgentAction, observation) tuples

        Returns:
            List of formatted tool call dictionaries
        """
        tool_calls = []

        for action, observation in steps:
            # Extract tool info
            tool_call = {
                'tool': action.tool if hasattr(action, 'tool') else str(action),
                'input': action.tool_input if hasattr(action, 'tool_input') else {},
                'output': str(observation)[:500]  # Truncate to 500 chars
            }
            tool_calls.append(tool_call)

        return tool_calls

    def get_tools_info(self) -> List[Dict]:
        """
        Get information about available tools.

        Returns:
            List of tool info dictionaries
        """
        return [
            {
                'name': tool.name,
                'description': tool.description
            }
            for tool in self.tools
        ]


# Global agent instance (singleton pattern for performance)
_agent_instance: Optional[DocumentationAgent] = None


def get_agent() -> DocumentationAgent:
    """
    Get or create global agent instance.

    Returns:
        DocumentationAgent instance
    """
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = DocumentationAgent()

    return _agent_instance
