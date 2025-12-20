"""
Code Validator Tool - Validate Generated Code

LangChain tool for validating code syntax and basic correctness.
"""

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import asyncio
import logging
import ast
import re

logger = logging.getLogger(__name__)


class CodeValidatorInput(BaseModel):
    """Input schema for code validator tool"""
    code: str = Field(
        description="Code to validate. Should include language identifier if possible."
    )
    language: Optional[str] = Field(
        default=None,
        description="Programming language (python, javascript, etc.). Auto-detected if not provided."
    )


class CodeValidatorTool(BaseTool):
    """
    Tool for validating code syntax and basic correctness.

    Supports:
    - Python (syntax via ast.parse)
    - JavaScript/TypeScript (basic syntax checks)
    - Generic validation (brackets, quotes)

    Example usage by agent:
        Action: code_validator
        Action Input: {"code": "def foo():\n    return 42", "language": "python"}
    """

    name: str = "code_validator"
    description: str = (
        "Validates code syntax and checks for basic errors. "
        "Use this after generating code to ensure it's syntactically correct. "
        "Input should be the code to validate and optionally the language."
    )
    args_schema: Type[BaseModel] = CodeValidatorInput

    def _run(self, code: str, language: Optional[str] = None) -> str:
        """
        Validate code (sync version).

        Args:
            code: Code to validate
            language: Programming language (optional, auto-detected)

        Returns:
            Validation result message
        """
        try:
            logger.info(f"Validating {language or 'auto-detected'} code")

            # Auto-detect language if not provided
            if not language:
                language = self._detect_language(code)

            # Validate based on language
            if language == "python":
                result = self._validate_python(code)
            elif language in ["javascript", "typescript"]:
                result = self._validate_javascript(code)
            else:
                result = self._validate_generic(code)

            logger.info(f"Validation result: {result['valid']}")

            # Format response
            if result['valid']:
                return f"✅ Code validation passed ({language}).\n{result.get('message', '')}"
            else:
                return f"❌ Code validation failed ({language}):\n{result['error']}\n\nSuggestion: {result.get('suggestion', 'Check syntax and try again.')}"

        except Exception as e:
            logger.error(f"Code validation error: {e}", exc_info=True)
            return f"Error during validation: {str(e)}"

    async def _arun(self, code: str, language: Optional[str] = None) -> str:
        """Async version"""
        return await asyncio.to_thread(self._run, code, language)

    def _detect_language(self, code: str) -> str:
        """
        Auto-detect programming language from code.

        Args:
            code: Code to analyze

        Returns:
            Detected language
        """
        code_lower = code.lower()

        # Python indicators
        if any(keyword in code for keyword in ['def ', 'import ', 'from ', 'async def', 'await ']):
            return "python"

        # JavaScript/TypeScript indicators
        if any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'async ', 'await ']):
            return "javascript"

        # Default to generic
        return "generic"

    def _validate_python(self, code: str) -> dict:
        """
        Validate Python code syntax.

        Args:
            code: Python code

        Returns:
            Validation result dict
        """
        try:
            # Remove markdown code fences if present
            code = self._strip_markdown_fences(code)

            # Try to parse as Python AST
            ast.parse(code)

            # Additional checks
            warnings = []

            # Check for common issues
            if 'print(' in code and code.count('print(') > 5:
                warnings.append("Many print statements (consider logging)")

            if len(code.splitlines()) > 100:
                warnings.append("Code is quite long (consider splitting)")

            message = "Code is syntactically correct."
            if warnings:
                message += "\n\nWarnings:\n- " + "\n- ".join(warnings)

            return {
                'valid': True,
                'message': message,
                'warnings': warnings
            }

        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error at line {e.lineno}: {e.msg}",
                'suggestion': "Check for missing colons, incorrect indentation, or unclosed brackets."
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Parse error: {str(e)}",
                'suggestion': "Verify the code is valid Python."
            }

    def _validate_javascript(self, code: str) -> dict:
        """
        Validate JavaScript/TypeScript code (basic checks).

        Note: Without a JS parser, we do basic bracket/brace matching.

        Args:
            code: JavaScript code

        Returns:
            Validation result dict
        """
        try:
            # Remove markdown code fences if present
            code = self._strip_markdown_fences(code)

            # Basic bracket/brace matching
            bracket_check = self._check_brackets(code)
            if not bracket_check['valid']:
                return {
                    'valid': False,
                    'error': bracket_check['error'],
                    'suggestion': "Check for missing or extra brackets/braces/parentheses."
                }

            # Check for common syntax patterns
            warnings = []

            # Missing semicolons (optional warning)
            lines = [line.strip() for line in code.splitlines() if line.strip()]
            non_semicolon_lines = [
                line for line in lines
                if line and not line.endswith((';', '{', '}', ','))
                and not line.startswith(('//', '/*', '*', 'import', 'export'))
            ]
            if len(non_semicolon_lines) > len(lines) * 0.5:
                warnings.append("Many lines without semicolons (style preference)")

            # console.log count
            if code.count('console.log') > 5:
                warnings.append("Many console.log statements (consider proper logging)")

            message = "Code appears syntactically correct (basic validation)."
            if warnings:
                message += "\n\nWarnings:\n- " + "\n- ".join(warnings)

            return {
                'valid': True,
                'message': message,
                'warnings': warnings
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}",
                'suggestion': "Verify the code structure."
            }

    def _validate_generic(self, code: str) -> dict:
        """
        Generic code validation (bracket matching, basic structure).

        Args:
            code: Code in any language

        Returns:
            Validation result dict
        """
        try:
            # Remove markdown code fences if present
            code = self._strip_markdown_fences(code)

            # Check bracket matching
            bracket_check = self._check_brackets(code)
            if not bracket_check['valid']:
                return {
                    'valid': False,
                    'error': bracket_check['error'],
                    'suggestion': "Check for balanced brackets, braces, and parentheses."
                }

            # Check quote matching
            quote_check = self._check_quotes(code)
            if not quote_check['valid']:
                return {
                    'valid': False,
                    'error': quote_check['error'],
                    'suggestion': "Check for unclosed strings."
                }

            return {
                'valid': True,
                'message': "Basic validation passed (brackets and quotes balanced)."
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}",
                'suggestion': "Review code structure."
            }

    def _check_brackets(self, code: str) -> dict:
        """
        Check if brackets, braces, and parentheses are balanced.

        Args:
            code: Code to check

        Returns:
            Dict with 'valid' and optional 'error'
        """
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        opening = set(pairs.keys())
        closing = set(pairs.values())

        # Remove string literals to avoid false positives
        code_no_strings = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '', code)

        for i, char in enumerate(code_no_strings):
            if char in opening:
                stack.append((char, i))
            elif char in closing:
                if not stack:
                    return {
                        'valid': False,
                        'error': f"Unexpected closing bracket '{char}' at position {i}"
                    }
                open_char, open_pos = stack.pop()
                if pairs[open_char] != char:
                    return {
                        'valid': False,
                        'error': f"Mismatched brackets: '{open_char}' at {open_pos} and '{char}' at {i}"
                    }

        if stack:
            open_char, open_pos = stack[0]
            return {
                'valid': False,
                'error': f"Unclosed bracket '{open_char}' at position {open_pos}"
            }

        return {'valid': True}

    def _check_quotes(self, code: str) -> dict:
        """
        Check if quotes are balanced.

        Args:
            code: Code to check

        Returns:
            Dict with 'valid' and optional 'error'
        """
        # Count quotes (simple check)
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')

        if single_quotes % 2 != 0:
            return {
                'valid': False,
                'error': "Unbalanced single quotes"
            }

        if double_quotes % 2 != 0:
            return {
                'valid': False,
                'error': "Unbalanced double quotes"
            }

        return {'valid': True}

    def _strip_markdown_fences(self, code: str) -> str:
        """
        Remove markdown code fences (```python, ```, etc.).

        Args:
            code: Code potentially with markdown fences

        Returns:
            Code without fences
        """
        # Remove opening fence
        code = re.sub(r'^```[a-z]*\n?', '', code, flags=re.MULTILINE)
        # Remove closing fence
        code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
        return code.strip()
