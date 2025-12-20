"""
Code Generator Tool - Generate Code Examples

LangChain tool for generating code examples based on specifications.
"""

from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)


class CodeGenInput(BaseModel):
    """Input schema for code generation tool"""
    specification: str = Field(
        description=(
            "Specification for what code to generate. "
            "Should include: programming language, functionality, "
            "and any specific requirements or constraints."
        )
    )


class CodeGeneratorTool(BaseTool):
    """
    Tool for generating code examples in various languages.

    Generates working code examples based on specifications, supporting:
    - Python (FastAPI, async/await, etc.)
    - JavaScript/TypeScript
    - cURL commands
    - And more

    Example usage by agent:
        Action: code_generator
        Action Input: "Python function to create a FastAPI POST endpoint that accepts JSON"
    """

    name: str = "code_generator"
    description: str = (
        "Generates code examples in various programming languages. "
        "Use this when the user asks for code examples, implementations, or wants to see how to use an API. "
        "Input should be a clear specification of what code to generate, including the language and functionality."
    )
    args_schema: Type[BaseModel] = CodeGenInput
    generator: Optional[Any] = None  # Generator instance

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'generator') or self.generator is None:
            object.__setattr__(self, 'generator', None)

    def _get_generator(self):
        """Lazy load code generator"""
        if self.generator is None:
            try:
                from code_generator import CodeExampleGenerator
                self.generator = CodeExampleGenerator()
                logger.info("Code generator initialized in tool")
            except ImportError:
                # Fallback: use simple template-based generation
                logger.warning("CodeExampleGenerator not found, using fallback")
                self.generator = SimpleFallbackGenerator()
        return self.generator

    def _run(self, specification: str) -> str:
        """
        Generate code (sync version).

        Args:
            specification: What code to generate

        Returns:
            Generated code as string
        """
        try:
            logger.info(f"Code generator creating code for: {specification[:100]}...")

            generator = self._get_generator()

            # Generate code
            code = generator.generate_example(specification)

            logger.info("Code generation successful")

            return code

        except Exception as e:
            logger.error(f"Code generation error: {e}", exc_info=True)
            return f"Error generating code: {str(e)}"

    async def _arun(self, specification: str) -> str:
        """Async version"""
        return await asyncio.to_thread(self._run, specification)


class SimpleFallbackGenerator:
    """
    Fallback code generator using simple templates.

    Used when CodeExampleGenerator is not available.
    """

    def generate_example(self, specification: str) -> str:
        """Generate simple code example based on specification"""

        spec_lower = specification.lower()

        # Python FastAPI examples
        if 'fastapi' in spec_lower and 'python' in spec_lower:
            if 'post' in spec_lower or 'create' in spec_lower:
                return self._fastapi_post_example()
            elif 'get' in spec_lower:
                return self._fastapi_get_example()
            else:
                return self._fastapi_basic_example()

        # Python async examples
        elif 'async' in spec_lower and 'python' in spec_lower:
            return self._python_async_example()

        # cURL examples
        elif 'curl' in spec_lower:
            return self._curl_example()

        # JavaScript/TypeScript examples
        elif 'javascript' in spec_lower or 'typescript' in spec_lower:
            return self._javascript_example()

        # Generic Python
        else:
            return self._generic_python_example(specification)

    def _fastapi_post_example(self) -> str:
        return '''```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float

@app.post("/items/")
async def create_item(item: Item):
    """Create a new item"""
    # Process the item
    return {"item_id": 1, "name": item.name, "price": item.price}
```'''

    def _fastapi_get_example(self) -> str:
        return '''```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """Get an item by ID"""
    return {"item_id": item_id, "name": "Sample Item"}
```'''

    def _fastapi_basic_example(self) -> str:
        return '''```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```'''

    def _python_async_example(self) -> str:
        return '''```python
import asyncio

async def fetch_data(url: str):
    """Fetch data asynchronously"""
    # Simulate async operation
    await asyncio.sleep(1)
    return {"url": url, "data": "result"}

async def main():
    """Main async function"""
    tasks = [fetch_data(f"https://api.example.com/{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

# Run
asyncio.run(main())
```'''

    def _curl_example(self) -> str:
        return '''```bash
# GET request
curl -X GET "https://api.example.com/items/1" \\
  -H "Content-Type: application/json"

# POST request
curl -X POST "https://api.example.com/items" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "Item Name", "price": 29.99}'
```'''

    def _javascript_example(self) -> str:
        return '''```javascript
// Fetch API example
async function fetchData(url) {
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Usage
fetchData('https://api.example.com/items/1')
    .then(data => console.log(data));
```'''

    def _generic_python_example(self, specification: str) -> str:
        return f'''```python
# Generated code based on: {specification}

def example_function(param):
    """
    Example function based on your specification.

    Args:
        param: Input parameter

    Returns:
        Result
    """
    # Implementation here
    result = f"Processing: {{param}}"
    return result

# Usage
if __name__ == "__main__":
    result = example_function("test")
    print(result)
```'''
