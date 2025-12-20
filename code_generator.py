"""
Code Example Generator for API documentation.

This module generates multi-language code examples for API endpoints,
including cURL, Python, JavaScript and other popular languages.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode, urlparse, parse_qs

logger = logging.getLogger(__name__)


class CodeExampleGenerator:
    """Generates code examples for APIs in multiple programming languages."""
    
    def __init__(self):
        """Initialize the code example generator."""
        self.supported_languages = [
            'curl', 'python', 'javascript', 'node', 'php', 'ruby', 'go', 'java'
        ]
        
        # Common headers that should be included
        self.common_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'API-Client/1.0'
        }
        
        logger.info(f"CodeExampleGenerator initialized with support for: {', '.join(self.supported_languages)}")
    
    def generate_multi_language_examples(self, endpoint_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate examples in multiple programming languages.
        
        Args:
            endpoint_info: Dictionary containing endpoint information
                - method: HTTP method (GET, POST, etc.)
                - path: API endpoint path
                - base_url: Base URL for the API
                - parameters: List of parameters
                - request_body: Request body schema/example
                - headers: Required headers
                - auth: Authentication information
        
        Returns:
            List of code examples with language, title, and code
        """
        examples = []
        
        try:
            # Generate examples for supported languages
            examples.append(self._generate_curl_example(endpoint_info))
            examples.append(self._generate_python_example(endpoint_info))
            examples.append(self._generate_javascript_example(endpoint_info))
            examples.append(self._generate_node_example(endpoint_info))
            
            # Add additional languages for common HTTP methods
            if endpoint_info.get('method', 'GET') in ['POST', 'PUT', 'PATCH']:
                examples.append(self._generate_php_example(endpoint_info))
            
            return [ex for ex in examples if ex is not None]
            
        except Exception as e:
            logger.error(f"Error generating multi-language examples: {e}")
            return []
    
    def _generate_curl_example(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate cURL example."""
        try:
            method = endpoint_info.get('method', 'GET').upper()
            base_url = endpoint_info.get('base_url', 'https://api.example.com')
            path = endpoint_info.get('path', '/')
            
            # Construct full URL
            full_url = f"{base_url.rstrip('/')}{path}"
            
            # Start with basic curl command
            curl_parts = [f"curl -X {method}"]
            
            # Add headers
            headers = endpoint_info.get('headers', {})
            auth_info = endpoint_info.get('auth', {})
            
            # Merge with common headers
            all_headers = {**self.common_headers, **headers}
            
            # Add authentication header if present
            if auth_info.get('type') == 'bearer':
                all_headers['Authorization'] = 'Bearer YOUR_ACCESS_TOKEN'
            elif auth_info.get('type') == 'api_key':
                if auth_info.get('location') == 'header':
                    all_headers[auth_info.get('name', 'X-API-Key')] = 'YOUR_API_KEY'
            
            for key, value in all_headers.items():
                curl_parts.append(f"-H '{key}: {value}'")
            
            # Add query parameters for GET requests
            parameters = endpoint_info.get('parameters', [])
            query_params = {}
            path_params = {}
            
            for param in parameters:
                param_name = param.get('name', '')
                param_location = param.get('in', 'query')
                param_example = self._get_param_example(param)
                
                if param_location == 'query':
                    query_params[param_name] = param_example
                elif param_location == 'path':
                    path_params[param_name] = param_example
            
            # Replace path parameters
            for param_name, param_value in path_params.items():
                full_url = full_url.replace(f'{{{param_name}}}', str(param_value))
            
            # Add query parameters
            if query_params and method == 'GET':
                query_string = urlencode(query_params)
                full_url += f"?{query_string}"
            
            # Add request body for POST/PUT/PATCH
            request_body = endpoint_info.get('request_body')
            if method in ['POST', 'PUT', 'PATCH'] and request_body:
                if isinstance(request_body, dict):
                    body_json = json.dumps(request_body, indent=2)
                    curl_parts.append(f"-d '{body_json}'")
                elif isinstance(request_body, str):
                    curl_parts.append(f"-d '{request_body}'")
            
            # Add URL as final parameter
            curl_parts.append(f"'{full_url}'")
            
            return {
                'language': 'curl',
                'title': 'cURL',
                'code': " \\\n  ".join(curl_parts)
            }
            
        except Exception as e:
            logger.error(f"Error generating cURL example: {e}")
            return None
    
    def _generate_python_example(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate Python example using requests library."""
        try:
            method = endpoint_info.get('method', 'GET').lower()
            base_url = endpoint_info.get('base_url', 'https://api.example.com')
            path = endpoint_info.get('path', '/')
            
            # Build Python code
            code_lines = [
                "import requests",
                "import json",
                "",
                f"# API endpoint",
                f'base_url = "{base_url}"',
                f'endpoint = "{path}"',
                f'url = base_url + endpoint',
                ""
            ]
            
            # Headers
            headers = endpoint_info.get('headers', {})
            auth_info = endpoint_info.get('auth', {})
            all_headers = {**self.common_headers, **headers}
            
            # Add authentication
            if auth_info.get('type') == 'bearer':
                all_headers['Authorization'] = 'YOUR_ACCESS_TOKEN'
            elif auth_info.get('type') == 'api_key':
                if auth_info.get('location') == 'header':
                    all_headers[auth_info.get('name', 'X-API-Key')] = 'YOUR_API_KEY'
            
            code_lines.append("# Headers")
            code_lines.append("headers = {")
            for key, value in all_headers.items():
                code_lines.append(f'    "{key}": "{value}",')
            code_lines.append("}")
            code_lines.append("")
            
            # Parameters
            parameters = endpoint_info.get('parameters', [])
            query_params = {}
            
            for param in parameters:
                if param.get('in') == 'query':
                    param_name = param.get('name', '')
                    param_example = self._get_param_example(param)
                    query_params[param_name] = param_example
            
            if query_params:
                code_lines.append("# Query parameters")
                code_lines.append("params = {")
                for key, value in query_params.items():
                    if isinstance(value, str):
                        code_lines.append(f'    "{key}": "{value}",')
                    else:
                        code_lines.append(f'    "{key}": {value},')
                code_lines.append("}")
                code_lines.append("")
            
            # Request body
            request_body = endpoint_info.get('request_body')
            if method in ['post', 'put', 'patch'] and request_body:
                code_lines.append("# Request body")
                code_lines.append("data = {")
                if isinstance(request_body, dict):
                    for key, value in request_body.items():
                        if isinstance(value, str):
                            code_lines.append(f'    "{key}": "{value}",')
                        else:
                            code_lines.append(f'    "{key}": {json.dumps(value)},')
                code_lines.append("}")
                code_lines.append("")
            
            # Make request
            code_lines.append("# Make request")
            request_args = ["url", "headers=headers"]
            
            if query_params:
                request_args.append("params=params")
            
            if method in ['post', 'put', 'patch'] and request_body:
                request_args.append("json=data")
            
            request_line = f'response = requests.{method}({", ".join(request_args)})'
            code_lines.append(request_line)
            code_lines.append("")
            
            # Handle response
            code_lines.extend([
                "# Handle response",
                "if response.status_code == 200:",
                "    result = response.json()",
                "    print(json.dumps(result, indent=2))",
                "else:",
                "    print(f'Error: {response.status_code} - {response.text}')"
            ])
            
            return {
                'language': 'python',
                'title': 'Python (requests)',
                'code': '\n'.join(code_lines)
            }
            
        except Exception as e:
            logger.error(f"Error generating Python example: {e}")
            return None
    
    def _generate_javascript_example(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate JavaScript example using fetch API."""
        try:
            method = endpoint_info.get('method', 'GET').upper()
            base_url = endpoint_info.get('base_url', 'https://api.example.com')
            path = endpoint_info.get('path', '/')
            
            code_lines = [
                "// API endpoint",
                f'const baseUrl = "{base_url}";',
                f'const endpoint = "{path}";',
                f'const url = baseUrl + endpoint;',
                ""
            ]
            
            # Headers
            headers = endpoint_info.get('headers', {})
            auth_info = endpoint_info.get('auth', {})
            all_headers = {**self.common_headers, **headers}
            
            if auth_info.get('type') == 'bearer':
                all_headers['Authorization'] = 'Bearer YOUR_ACCESS_TOKEN'
            elif auth_info.get('type') == 'api_key':
                if auth_info.get('location') == 'header':
                    all_headers[auth_info.get('name', 'X-API-Key')] = 'YOUR_API_KEY'
            
            code_lines.append("// Request options")
            code_lines.append("const options = {")
            code_lines.append(f'  method: "{method}",')
            code_lines.append("  headers: {")
            for key, value in all_headers.items():
                code_lines.append(f'    "{key}": "{value}",')
            code_lines.append("  }")
            
            # Request body
            request_body = endpoint_info.get('request_body')
            if method in ['POST', 'PUT', 'PATCH'] and request_body:
                code_lines.append("  body: JSON.stringify({")
                if isinstance(request_body, dict):
                    for key, value in request_body.items():
                        if isinstance(value, str):
                            code_lines.append(f'    "{key}": "{value}",')
                        else:
                            code_lines.append(f'    "{key}": {json.dumps(value)},')
                code_lines.append("  })")
            
            code_lines.append("};")
            code_lines.append("")
            
            # Query parameters
            parameters = endpoint_info.get('parameters', [])
            query_params = {}
            
            for param in parameters:
                if param.get('in') == 'query':
                    param_name = param.get('name', '')
                    param_example = self._get_param_example(param)
                    query_params[param_name] = param_example
            
            if query_params:
                code_lines.append("// Add query parameters")
                code_lines.append("const params = new URLSearchParams({")
                for key, value in query_params.items():
                    code_lines.append(f'  "{key}": "{value}",')
                code_lines.append("});")
                code_lines.append("const urlWithParams = `${url}?${params}`;")
                code_lines.append("")
            
            # Make request
            final_url = "urlWithParams" if query_params else "url"
            code_lines.extend([
                "// Make request",
                f"fetch({final_url}, options)",
                "  .then(response => {",
                "    if (!response.ok) {",
                "      throw new Error(`HTTP error! status: ${response.status}`);",
                "    }",
                "    return response.json();",
                "  })",
                "  .then(data => {",
                "    console.log('Success:', data);",
                "  })",
                "  .catch(error => {",
                "    console.error('Error:', error);",
                "  });"
            ])
            
            return {
                'language': 'javascript',
                'title': 'JavaScript (fetch)',
                'code': '\n'.join(code_lines)
            }
            
        except Exception as e:
            logger.error(f"Error generating JavaScript example: {e}")
            return None
    
    def _generate_node_example(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate Node.js example using axios."""
        try:
            method = endpoint_info.get('method', 'GET').lower()
            base_url = endpoint_info.get('base_url', 'https://api.example.com')
            path = endpoint_info.get('path', '/')
            
            code_lines = [
                "const axios = require('axios');",
                "",
                "// API configuration",
                f'const baseURL = "{base_url}";',
                f'const endpoint = "{path}";',
                f'const url = baseURL + endpoint;',
                ""
            ]
            
            # Headers and auth
            headers = endpoint_info.get('headers', {})
            auth_info = endpoint_info.get('auth', {})
            all_headers = {**self.common_headers, **headers}
            
            if auth_info.get('type') == 'bearer':
                all_headers['Authorization'] = 'Bearer YOUR_ACCESS_TOKEN'
            elif auth_info.get('type') == 'api_key':
                if auth_info.get('location') == 'header':
                    all_headers[auth_info.get('name', 'X-API-Key')] = 'YOUR_API_KEY'
            
            # Configuration object
            code_lines.append("// Request configuration")
            code_lines.append("const config = {")
            code_lines.append(f'  method: "{method}",')
            code_lines.append("  url: url,")
            code_lines.append("  headers: {")
            for key, value in all_headers.items():
                code_lines.append(f'    "{key}": "{value}",')
            code_lines.append("  }")
            
            # Query parameters
            parameters = endpoint_info.get('parameters', [])
            query_params = {}
            
            for param in parameters:
                if param.get('in') == 'query':
                    param_name = param.get('name', '')
                    param_example = self._get_param_example(param)
                    query_params[param_name] = param_example
            
            if query_params:
                code_lines.append("  params: {")
                for key, value in query_params.items():
                    if isinstance(value, str):
                        code_lines.append(f'    "{key}": "{value}",')
                    else:
                        code_lines.append(f'    "{key}": {value},')
                code_lines.append("  }")
            
            # Request body
            request_body = endpoint_info.get('request_body')
            if method in ['post', 'put', 'patch'] and request_body:
                code_lines.append("  data: {")
                if isinstance(request_body, dict):
                    for key, value in request_body.items():
                        if isinstance(value, str):
                            code_lines.append(f'    "{key}": "{value}",')
                        else:
                            code_lines.append(f'    "{key}": {json.dumps(value)},')
                code_lines.append("  }")
            
            code_lines.append("};")
            code_lines.append("")
            
            # Make request
            code_lines.extend([
                "// Make request",
                "axios(config)",
                "  .then(response => {",
                "    console.log('Status:', response.status);",
                "    console.log('Data:', JSON.stringify(response.data, null, 2));",
                "  })",
                "  .catch(error => {",
                "    if (error.response) {",
                "      console.error('Error Status:', error.response.status);",
                "      console.error('Error Data:', error.response.data);",
                "    } else {",
                "      console.error('Error Message:', error.message);",
                "    }",
                "  });"
            ])
            
            return {
                'language': 'javascript',
                'title': 'Node.js (axios)',
                'code': '\n'.join(code_lines)
            }
            
        except Exception as e:
            logger.error(f"Error generating Node.js example: {e}")
            return None
    
    def _generate_php_example(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate PHP example using cURL."""
        try:
            method = endpoint_info.get('method', 'GET').upper()
            base_url = endpoint_info.get('base_url', 'https://api.example.com')
            path = endpoint_info.get('path', '/')
            
            code_lines = [
                "<?php",
                "",
                "// API configuration",
                f'$baseUrl = "{base_url}";',
                f'$endpoint = "{path}";',
                f'$url = $baseUrl . $endpoint;',
                "",
                "// Initialize cURL",
                "$ch = curl_init();",
                ""
            ]
            
            # Basic cURL options
            code_lines.extend([
                "// Set cURL options",
                "curl_setopt($ch, CURLOPT_URL, $url);",
                "curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);",
                f'curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "{method}");',
            ])
            
            # Headers
            headers = endpoint_info.get('headers', {})
            auth_info = endpoint_info.get('auth', {})
            all_headers = {**self.common_headers, **headers}
            
            if auth_info.get('type') == 'bearer':
                all_headers['Authorization'] = 'Bearer YOUR_ACCESS_TOKEN'
            elif auth_info.get('type') == 'api_key':
                if auth_info.get('location') == 'header':
                    all_headers[auth_info.get('name', 'X-API-Key')] = 'YOUR_API_KEY'
            
            if all_headers:
                code_lines.append("")
                code_lines.append("// Set headers")
                code_lines.append("$headers = [")
                for key, value in all_headers.items():
                    code_lines.append(f'    "{key}: {value}",')
                code_lines.append("];")
                code_lines.append("curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);")
            
            # Request body
            request_body = endpoint_info.get('request_body')
            if method in ['POST', 'PUT', 'PATCH'] and request_body:
                code_lines.append("")
                code_lines.append("// Request body")
                code_lines.append("$data = [")
                if isinstance(request_body, dict):
                    for key, value in request_body.items():
                        if isinstance(value, str):
                            code_lines.append(f'    "{key}" => "{value}",')
                        else:
                            code_lines.append(f'    "{key}" => {json.dumps(value)},')
                code_lines.append("];")
                code_lines.append("curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));")
            
            # Execute request and handle response
            code_lines.extend([
                "",
                "// Execute request",
                "$response = curl_exec($ch);",
                "$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);",
                "",
                "// Check for errors",
                "if (curl_errno($ch)) {",
                "    echo 'cURL Error: ' . curl_error($ch);",
                "} else {",
                "    echo 'HTTP Status: ' . $httpCode . PHP_EOL;",
                "    echo 'Response: ' . $response . PHP_EOL;",
                "}",
                "",
                "// Close cURL handle",
                "curl_close($ch);",
                "?>"
            ])
            
            return {
                'language': 'php',
                'title': 'PHP (cURL)',
                'code': '\n'.join(code_lines)
            }
            
        except Exception as e:
            logger.error(f"Error generating PHP example: {e}")
            return None
    
    def _get_param_example(self, param: Dict[str, Any]) -> Any:
        """Generate example value for a parameter."""
        param_type = param.get('type', param.get('schema', {}).get('type', 'string'))
        param_name = param.get('name', 'value')
        
        # Use provided example if available
        if 'example' in param:
            return param['example']
        
        # Generate example based on type
        if param_type == 'string':
            if 'email' in param_name.lower():
                return 'user@example.com'
            elif 'id' in param_name.lower():
                return 'abc123'
            elif 'name' in param_name.lower():
                return 'example_name'
            else:
                return 'example_value'
        elif param_type == 'integer':
            return 123
        elif param_type == 'number':
            return 123.45
        elif param_type == 'boolean':
            return True
        elif param_type == 'array':
            return ['item1', 'item2']
        else:
            return 'example_value'
    
    def generate_authentication_examples(self, auth_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate authentication examples for different languages."""
        auth_type = auth_info.get('type', 'api_key')
        examples = []
        
        if auth_type == 'bearer':
            examples.extend([
                {
                    'language': 'curl',
                    'title': 'Bearer Token Authentication',
                    'code': 'curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" https://api.example.com/endpoint'
                },
                {
                    'language': 'python',
                    'title': 'Bearer Token (Python)',
                    'code': '''headers = {
    "Authorization": "Bearer YOUR_ACCESS_TOKEN",
    "Content-Type": "application/json"
}
response = requests.get("https://api.example.com/endpoint", headers=headers)'''
                }
            ])
        
        elif auth_type == 'api_key':
            location = auth_info.get('location', 'header')
            name = auth_info.get('name', 'X-API-Key')
            
            if location == 'header':
                examples.extend([
                    {
                        'language': 'curl',
                        'title': 'API Key Authentication',
                        'code': f'curl -H "{name}: YOUR_API_KEY" https://api.example.com/endpoint'
                    },
                    {
                        'language': 'python', 
                        'title': 'API Key (Python)',
                        'code': f'''headers = {{
    "{name}": "YOUR_API_KEY",
    "Content-Type": "application/json"
}}
response = requests.get("https://api.example.com/endpoint", headers=headers)'''
                    }
                ])
        
        return examples
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return self.supported_languages.copy()

    def generate_example(self, specification: str) -> str:
        """
        Generate code example from a simple string specification.

        This method parses a natural language specification and generates
        appropriate code examples. Used by the LangChain code generator tool.

        Args:
            specification: Natural language description of what code to generate
                Examples:
                - "Python FastAPI POST endpoint"
                - "cURL GET request with authentication"
                - "JavaScript async function"

        Returns:
            Generated code as a formatted string
        """
        try:
            spec_lower = specification.lower()

            # Detect language and endpoint information from specification
            if 'curl' in spec_lower:
                # Generate cURL example
                endpoint_info = {
                    'method': 'POST' if 'post' in spec_lower else 'GET',
                    'base_url': 'https://api.example.com',
                    'path': '/endpoint',
                    'headers': {},
                    'auth': {'type': 'bearer'} if 'auth' in spec_lower else {}
                }
                result = self._generate_curl_example(endpoint_info)
                return f"```bash\n{result['code']}\n```"

            elif 'fastapi' in spec_lower or ('python' in spec_lower and ('api' in spec_lower or 'endpoint' in spec_lower)):
                # Generate FastAPI example
                endpoint_info = {
                    'method': 'POST' if 'post' in spec_lower else 'GET',
                    'base_url': 'https://api.example.com',
                    'path': '/items',
                    'parameters': [],
                    'request_body': {'name': 'example', 'value': 123} if 'post' in spec_lower else None,
                    'headers': {},
                    'auth': {}
                }

                # Generate Python FastAPI code
                if 'post' in spec_lower:
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
                else:
                    return '''```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """Get an item by ID"""
    return {"item_id": item_id, "name": "Sample Item"}
```'''

            elif 'python' in spec_lower and 'async' in spec_lower:
                # Generate Python async example
                return '''```python
import asyncio

async def async_function():
    """Example async function"""
    await asyncio.sleep(1)
    return "result"

async def main():
    """Main async function"""
    result = await async_function()
    print(result)

# Run
asyncio.run(main())
```'''

            elif 'javascript' in spec_lower or 'node' in spec_lower:
                # Generate JavaScript example
                endpoint_info = {
                    'method': 'POST' if 'post' in spec_lower else 'GET',
                    'base_url': 'https://api.example.com',
                    'path': '/endpoint',
                    'parameters': [],
                    'request_body': {'name': 'example'} if 'post' in spec_lower else None,
                    'headers': {},
                    'auth': {}
                }
                result = self._generate_javascript_example(endpoint_info)
                return f"```javascript\n{result['code']}\n```"

            elif 'python' in spec_lower:
                # Generate generic Python example
                endpoint_info = {
                    'method': 'POST' if 'post' in spec_lower else 'GET',
                    'base_url': 'https://api.example.com',
                    'path': '/endpoint',
                    'parameters': [],
                    'request_body': {'key': 'value'} if 'post' in spec_lower else None,
                    'headers': {},
                    'auth': {}
                }
                result = self._generate_python_example(endpoint_info)
                return f"```python\n{result['code']}\n```"

            else:
                # Fallback: generate multi-language examples
                endpoint_info = {
                    'method': 'GET',
                    'base_url': 'https://api.example.com',
                    'path': '/endpoint',
                    'parameters': [],
                    'request_body': None,
                    'headers': {},
                    'auth': {}
                }
                examples = self.generate_multi_language_examples(endpoint_info)
                if examples:
                    # Return the first example (usually cURL)
                    return f"```{examples[0]['language']}\n{examples[0]['code']}\n```"
                else:
                    return "```python\n# Example code\nprint('Hello World')\n```"

        except Exception as e:
            logger.error(f"Error generating example from specification: {e}")
            return f"```python\n# Error generating code: {str(e)}\n# Specification: {specification}\n```"