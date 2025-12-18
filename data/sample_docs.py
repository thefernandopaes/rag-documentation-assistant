"""
Sample documentation data for testing and development
This module contains sample documentation content that would normally be scraped from official sources
"""

SAMPLE_REACT_DOCS = [
    {
        'title': 'React Components and Props',
        'url': 'https://react.dev/learn/passing-props-to-a-component',
        'content': '''# Components and Props

Components let you split the UI into independent, reusable pieces, and think about each piece in isolation.

## Function Components

The simplest way to define a component is to write a JavaScript function:

```javascript
function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}
```

This function is a valid React component because it accepts a single "props" (which stands for properties) object argument with data and returns a React element.

## Class Components

You can also use ES6 classes to define a component:

```javascript
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}</h1>;
  }
}
```

## Using Components

Once you have a component, you can use it in other components:

```javascript
function App() {
  return (
    <div>
      <Welcome name="Sara" />
      <Welcome name="Cahal" />
      <Welcome name="Edite" />
    </div>
  );
}
```

## Props are Read-Only

Whether you declare a component as a function or a class, it must never modify its own props. React is pretty flexible but it has a single strict rule: All React components must act like pure functions with respect to their props.'''
    },
    {
        'title': 'React State and Lifecycle',
        'url': 'https://react.dev/learn/state-a-components-memory',
        'content': '''# State: A Component's Memory

Components often need to change what's on the screen as a result of an interaction. Typing into the form should update the input field, clicking "next" on an image carousel should change which image is displayed, clicking "buy" should put a product in the shopping cart. Components need to "remember" things: the current input value, the current image, the shopping cart. In React, this kind of component-specific memory is called state.

## Adding State to a Component

To add state to a component, use one of these Hooks:
- useState declares a state variable that you can update directly.
- useReducer declares a state variable with the update logic inside a reducer function.

```javascript
import { useState } from 'react';

function MyButton() {
  const [count, setCount] = useState(0);

  function handleClick() {
    setCount(count + 1);
  }

  return (
    <button onClick={handleClick}>
      Clicked {count} times
    </button>
  );
}
```

## State is Isolated and Private

State is local to a component instance on the screen. In other words, if you render the same component twice, each copy will have completely isolated state! Changing one of them will not affect the other.

```javascript
function MyApp() {
  return (
    <div>
      <h1>Counters that update separately</h1>
      <MyButton />
      <MyButton />
    </div>
  );
}
```'''
    },
    {
        'title': 'React Hooks Overview',
        'url': 'https://react.dev/reference/react',
        'content': '''# Hooks

Hooks let you use different React features from your components. You can either use the built-in Hooks or combine them to build your own.

## Built-in Hooks

### State Hooks
State lets a component remember information like user input.

- useState declares a state variable that you can update directly.
- useReducer declares a state variable with the update logic inside a reducer function.

```javascript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('Taylor');
  const [todos, setTodos] = useState(() => createTodos());
  // ...
}
```

### Effect Hooks
Effects let a component connect to and synchronize with external systems.

- useEffect connects a component to an external system.

```javascript
import { useEffect } from 'react';

function ChatRoom({ roomId }) {
  useEffect(() => {
    const connection = createConnection(roomId);
    connection.connect();
    return () => connection.disconnect();
  }, [roomId]);
  // ...
}
```

### Performance Hooks
A common way to optimize re-rendering performance is to skip unnecessary work.

- useMemo lets you cache the result of an expensive calculation.
- useCallback lets you cache a function definition before passing it down to an optimized component.

```javascript
import { useMemo, useCallback } from 'react';

function TodoList({ todos, tab, theme }) {
  const visibleTodos = useMemo(() => filterTodos(todos, tab), [todos, tab]);
  const handleAddTodo = useCallback((text) => {
    const newTodo = { id: nextId++, text };
    setTodos([...todos, newTodo]);
  }, [todos]);
  // ...
}
```'''
    }
]

SAMPLE_PYTHON_DOCS = [
    {
        'title': 'Python Functions',
        'url': 'https://docs.python.org/3/tutorial/controlflow.html#defining-functions',
        'content': '''# Defining Functions

The keyword def introduces a function definition. It must be followed by the function name and the parenthesized list of formal parameters. The statements that form the body of the function start at the next line, and must be indented.

```python
def greet(name):
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"

print(greet("World"))  # Output: Hello, World!
```

## Function Parameters

Functions can have various types of parameters:

### Default Arguments
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))           # Hello, Alice!
print(greet("Bob", "Hi"))       # Hi, Bob!
```

### Keyword Arguments
```python
def describe_pet(name, animal_type="dog"):
    print(f"I have a {animal_type} named {name}")

describe_pet("Buddy")
describe_pet(name="Luna", animal_type="cat")
describe_pet(animal_type="hamster", name="Harry")
```

### Variable-Length Arguments
```python
def make_pizza(*toppings):
    print("Making a pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")

make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')
```

## Lambda Functions

Python supports anonymous functions using the lambda keyword:

```python
# Regular function
def square(x):
    return x ** 2

# Lambda equivalent
square_lambda = lambda x: x ** 2

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]
```'''
    },
    {
        'title': 'Python Data Structures',
        'url': 'https://docs.python.org/3/tutorial/datastructures.html',
        'content': '''# Data Structures

Python has several built-in data structures that are very useful for organizing and storing data.

## Lists

Lists are ordered collections that are mutable (changeable):

```python
# Creating lists
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed = ['hello', 42, 3.14, True]

# List methods
fruits.append('orange')        # Add to end
fruits.insert(1, 'grape')     # Insert at index
fruits.remove('banana')       # Remove first occurrence
last_fruit = fruits.pop()     # Remove and return last item

# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

## Dictionaries

Dictionaries store key-value pairs:

```python
# Creating dictionaries
person = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}

# Accessing and modifying
print(person['name'])          # Alice
person['age'] = 31            # Update value
person['job'] = 'Engineer'    # Add new key-value pair

# Dictionary methods
keys = person.keys()          # Get all keys
values = person.values()      # Get all values
items = person.items()        # Get key-value pairs

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

## Sets

Sets are unordered collections of unique elements:

```python
# Creating sets
numbers = {1, 2, 3, 4, 5}
letters = set('hello')        # {'h', 'e', 'l', 'o'}

# Set operations
numbers.add(6)               # Add element
numbers.discard(1)           # Remove element (no error if not found)

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2          # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2   # {3, 4}
difference = set1 - set2     # {1, 2}
```'''
    },
    {
        'title': 'Python Error Handling',
        'url': 'https://docs.python.org/3/tutorial/errors.html',
        'content': '''# Error Handling

Python uses exceptions to handle errors that occur during program execution. The try/except statement is used to catch and handle exceptions.

## Basic Exception Handling

```python
try:
    x = int(input("Enter a number: "))
    result = 10 / x
    print(f"Result: {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print("No exceptions occurred!")
finally:
    print("This always executes")
```

## Raising Exceptions

You can raise exceptions using the raise statement:

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age

try:
    age = validate_age(-5)
except ValueError as e:
    print(f"Invalid age: {e}")
```

## Custom Exceptions

You can create custom exception classes:

```python
class InsufficientFundsError(Exception):
    """Raised when a bank account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: need ${amount}, but only have ${balance}")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance

# Usage
account = BankAccount(100)
try:
    account.withdraw(150)
except InsufficientFundsError as e:
    print(e)  # Insufficient funds: need $150, but only have $100
```

## Context Managers

The with statement ensures proper resource management:

```python
# File handling with automatic cleanup
with open('data.txt', 'r') as file:
    content = file.read()
    # File is automatically closed, even if an exception occurs

# Custom context manager
class DatabaseConnection:
    def __enter__(self):
        print("Connecting to database")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if exc_type:
            print(f"An exception occurred: {exc_val}")

with DatabaseConnection() as db:
    print("Using database connection")
    # Connection is automatically closed
```'''
    }
]

SAMPLE_FASTAPI_DOCS = [
    {
        'title': 'FastAPI First Steps',
        'url': 'https://fastapi.tiangolo.com/tutorial/first-steps/',
        'content': '''# First Steps

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Installation

```bash
pip install fastapi[all]
```

## Create Your First API

Create a file `main.py` with:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## Run the Server

Run the server with:

```bash
uvicorn main:app --reload
```

You now have an API that:
- Receives HTTP requests in the paths `/` and `/items/{item_id}`
- Both paths take GET operations (also known as HTTP methods)
- The path `/items/{item_id}` has a path parameter `item_id` that should be an int
- The path `/items/{item_id}` has an optional query parameter `q`

## Interactive API Documentation

Now go to http://127.0.0.1:8000/docs to see the automatic interactive API documentation (provided by Swagger UI).

You can also go to http://127.0.0.1:8000/redoc to see the alternative automatic documentation (provided by ReDoc).

## Path Parameters

You can declare path parameters with Python string formatting:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`. If you run this and go to http://127.0.0.1:8000/items/foo, you will see an HTTP error because `foo` is not an integer.'''
    },
    {
        'title': 'FastAPI Request Body and Pydantic Models',
        'url': 'https://fastapi.tiangolo.com/tutorial/body/',
        'content': '''# Request Body

When you need to send data from a client (like a browser) to your API, you send it as a request body.

## Import Pydantic's BaseModel

First, you need to import BaseModel from pydantic:

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

app = FastAPI()

@app.post("/items/")
def create_item(item: Item):
    return item
```

## Declare a Request Body

To declare a request body, you use Pydantic models. A Pydantic model is just a class that inherits from BaseModel.

With just that Python type declaration, FastAPI will:
- Read the body of the request as JSON
- Convert the corresponding types (if needed)
- Validate the data
- Give you the received data in the parameter item
- Generate JSON Schema definitions for your model

## Use the Model

Inside of the function, you can access all the attributes of the model object directly:

```python
@app.post("/items/")
def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Request Body + Path Parameters

You can declare path parameters and request body at the same time:

```python
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}
```

## Request Body + Path + Query Parameters

You can also declare body, path and query parameters, all at the same time:

```python
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, q: str = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result
```

The function parameters will be recognized as follows:
- If the parameter is also declared in the path, it will be a path parameter
- If the parameter is of a singular type (like int, float, str, bool, etc.) it will be interpreted as a query parameter
- If the parameter is declared to be of the type of a Pydantic model, it will be interpreted as a request body'''
    },
    {
        'title': 'FastAPI Authentication and Security',
        'url': 'https://fastapi.tiangolo.com/tutorial/security/',
        'content': '''# Security

There are many ways to handle security, authentication and authorization. And it normally is a complex and "difficult" topic.

FastAPI provides several tools to help you deal with Security easily, rapidly, in a standard way, without having to study and learn all the security specifications.

## OAuth2 with Password and Bearer

Let's first just use OAuth2 with Password (using a Bearer token). We'll do this using the OAuth2PasswordBearer.

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    return {"username": "testuser"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.get("/users/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
```

## JWT Authentication

Here's a more realistic example using JWT tokens:

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

class User(BaseModel):
    username: str
    email: str = None
    full_name: str = None
    disabled: bool = None

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
```'''
    }
]

SAMPLE_DOCKER_DOCS = [
    {
        'title': 'Docker Getting Started',
        'url': 'https://docs.docker.com/get-started/',
        'content': '''# Getting Started with Docker

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly.

## What is Docker?

Docker provides the ability to package and run an application in a loosely isolated environment called a container. Containers are lightweight and contain everything needed to run the application, so you don't need to rely on what's installed on the host.

## Installation

To install Docker, visit https://docs.docker.com/get-docker/ and follow the instructions for your operating system.

## Basic Concepts

### Images
A Docker image is a read-only template with instructions for creating a Docker container. Images are built from a Dockerfile and can be shared through registries like Docker Hub.

### Containers
A container is a runnable instance of an image. You can create, start, stop, move, or delete a container using the Docker CLI or API.

### Dockerfile
A Dockerfile is a text file that contains instructions for building a Docker image. Each instruction in a Dockerfile creates a layer in the image.

## Your First Container

Run your first container:

```bash
docker run hello-world
```

This command:
1. Downloads the hello-world image from Docker Hub (if not already present)
2. Creates a new container from the image
3. Runs the container
4. Displays the output

## Common Docker Commands

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# List images
docker images

# Pull an image from Docker Hub
docker pull nginx

# Run a container
docker run nginx

# Run a container in detached mode
docker run -d nginx

# Run a container with a name
docker run --name my-nginx nginx

# Stop a container
docker stop my-nginx

# Remove a container
docker rm my-nginx

# Remove an image
docker rmi nginx
```'''
    },
    {
        'title': 'Docker Dockerfile Best Practices',
        'url': 'https://docs.docker.com/develop/dev-best-practices/',
        'content': '''# Dockerfile Best Practices

A Dockerfile is a script that contains instructions for building a Docker image. Following best practices ensures your images are efficient, secure, and maintainable.

## Basic Dockerfile Structure

```dockerfile
# Use an official base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Define the command to run the application
CMD ["python", "app.py"]
```

## Best Practices

### 1. Use Official Base Images
Start with official images from trusted sources:

```dockerfile
FROM python:3.11-slim
FROM node:18-alpine
FROM nginx:alpine
```

### 2. Minimize Layers
Each RUN, COPY, and ADD instruction creates a new layer. Combine commands when possible:

```dockerfile
# Bad - creates 3 layers
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git

# Good - creates 1 layer
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
```

### 3. Leverage Build Cache
Order instructions from least to most frequently changing:

```dockerfile
# Dependencies change less frequently
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source code changes more frequently
COPY . .
```

### 4. Use .dockerignore
Create a .dockerignore file to exclude unnecessary files:

```
__pycache__
*.pyc
.git
.env
node_modules
.vscode
```

### 5. Multi-Stage Builds
Use multi-stage builds to reduce final image size:

```dockerfile
# Build stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 6. Don't Run as Root
Create a non-root user for better security:

```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

CMD ["python", "app.py"]
```

## Real-World Example

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```'''
    },
    {
        'title': 'Docker Compose for Multi-Container Applications',
        'url': 'https://docs.docker.com/compose/',
        'content': '''# Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services, networks, and volumes.

## Installation

Docker Compose comes pre-installed with Docker Desktop. For Linux, install separately:

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Basic docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Common Commands

```bash
# Start services in detached mode
docker-compose up -d

# View running services
docker-compose ps

# View logs
docker-compose logs
docker-compose logs -f web

# Stop services
docker-compose stop

# Stop and remove containers, networks
docker-compose down

# Stop and remove containers, networks, volumes
docker-compose down -v

# Rebuild images
docker-compose build

# Rebuild and start
docker-compose up -d --build
```

## Complete Example: Flask App with PostgreSQL and Redis

```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    command: gunicorn --bind 0.0.0.0:8000 app:app
    volumes:
      - ./app:/app
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/flaskapp
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=flaskapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
```

## Environment Variables

Create a `.env` file for sensitive data:

```env
POSTGRES_PASSWORD=supersecret
DATABASE_URL=postgresql://postgres:supersecret@db:5432/myapp
REDIS_URL=redis://redis:6379/0
```

Reference in docker-compose.yml:

```yaml
services:
  web:
    env_file:
      - .env
```

## Health Checks

Add health checks to ensure services are ready:

```yaml
services:
  db:
    image: postgres:15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  web:
    depends_on:
      db:
        condition: service_healthy
```'''
    }
]

SAMPLE_AWS_LAMBDA_DOCS = [
    {
        'title': 'AWS Lambda Getting Started',
        'url': 'https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html',
        'content': '''# Getting Started with AWS Lambda

AWS Lambda is a serverless compute service that runs your code in response to events and automatically manages the underlying compute resources for you.

## What is AWS Lambda?

AWS Lambda is a serverless, event-driven compute service that lets you run code for virtually any type of application or backend service without provisioning or managing servers.

### Key Features
- No servers to manage
- Automatic scaling
- Pay only for compute time
- Built-in fault tolerance
- Supports multiple programming languages

### Supported Runtimes
- Python 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
- Node.js 16.x, 18.x, 20.x
- Java 8, 11, 17, 21
- .NET Core 3.1, 6, 7
- Go 1.x
- Ruby 2.7, 3.2
- Custom runtime (using Runtime API)

## Your First Lambda Function

### Python Example

```python
import json

def lambda_handler(event, context):
    # Get data from event
    name = event.get('name', 'World')

    # Create response
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Hello, {name}!'
        })
    }

    return response
```

### Node.js Example

```javascript
exports.handler = async (event) => {
    const name = event.name || 'World';

    const response = {
        statusCode: 200,
        body: JSON.stringify({
            message: `Hello, ${name}!`
        })
    };

    return response;
};
```

## Lambda Function Components

### Event Object
The event object contains information from the invoking service. Structure varies by event source:

```python
# API Gateway event example
{
    "httpMethod": "POST",
    "path": "/users",
    "headers": {
        "Content-Type": "application/json"
    },
    "body": "{\"name\": \"John\"}"
}
```

### Context Object
The context object provides runtime information:

```python
def lambda_handler(event, context):
    print(f"Request ID: {context.request_id}")
    print(f"Function name: {context.function_name}")
    print(f"Memory limit: {context.memory_limit_in_mb} MB")
    print(f"Time remaining: {context.get_remaining_time_in_millis()} ms")

    return {'statusCode': 200}
```

## Creating a Lambda Function

### Using AWS Console
1. Open AWS Lambda console
2. Click "Create function"
3. Choose "Author from scratch"
4. Configure basic settings:
   - Function name
   - Runtime (e.g., Python 3.11)
   - Architecture (x86_64 or arm64)
5. Click "Create function"
6. Add your code in the code editor
7. Click "Deploy"

### Using AWS CLI

```bash
# Create a deployment package
zip function.zip lambda_function.py

# Create the function
aws lambda create-function \
    --function-name my-function \
    --runtime python3.11 \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://function.zip

# Invoke the function
aws lambda invoke \
    --function-name my-function \
    --payload '{"name": "Alice"}' \
    response.json
```

## Environment Variables

Set environment variables for configuration:

```python
import os

def lambda_handler(event, context):
    db_host = os.environ.get('DB_HOST')
    db_name = os.environ.get('DB_NAME')

    print(f"Connecting to {db_name} at {db_host}")

    return {'statusCode': 200}
```

Configure via AWS Console or CLI:

```bash
aws lambda update-function-configuration \
    --function-name my-function \
    --environment Variables={DB_HOST=mydb.example.com,DB_NAME=production}
```'''
    },
    {
        'title': 'AWS Lambda with API Gateway',
        'url': 'https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html',
        'content': '''# Using AWS Lambda with Amazon API Gateway

You can create a web API with an HTTP endpoint for your Lambda function by using Amazon API Gateway. API Gateway provides features like authentication, request/response transformation, and more.

## REST API Example

### Lambda Function for REST API

```python
import json

def lambda_handler(event, context):
    # Get HTTP method
    http_method = event['httpMethod']

    # Get path parameters
    path_params = event.get('pathParameters', {})

    # Get query parameters
    query_params = event.get('queryStringParameters', {})

    # Get request body
    body = json.loads(event.get('body', '{}'))

    # Route based on HTTP method
    if http_method == 'GET':
        return get_user(path_params.get('id'))
    elif http_method == 'POST':
        return create_user(body)
    elif http_method == 'PUT':
        return update_user(path_params.get('id'), body)
    elif http_method == 'DELETE':
        return delete_user(path_params.get('id'))
    else:
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }

def get_user(user_id):
    # Database lookup logic here
    user = {
        'id': user_id,
        'name': 'John Doe',
        'email': 'john@example.com'
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(user)
    }

def create_user(user_data):
    # Create user in database
    return {
        'statusCode': 201,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'message': 'User created successfully',
            'user': user_data
        })
    }
```

## HTTP API Example (Lambda Proxy Integration)

```python
import json

def lambda_handler(event, context):
    # HTTP API event structure
    route_key = event.get('routeKey')  # e.g., "GET /users/{id}"
    path_params = event.get('pathParameters', {})

    # Get request context
    request_context = event.get('requestContext', {})
    http = request_context.get('http', {})

    method = http.get('method')
    path = http.get('path')

    print(f"Processing {method} request to {path}")

    # Process request
    if route_key == 'GET /users/{id}':
        user_id = path_params.get('id')
        response_body = {
            'id': user_id,
            'name': 'Jane Smith'
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response_body)
        }

    return {
        'statusCode': 404,
        'body': json.dumps({'error': 'Not found'})
    }
```

## CORS Configuration

Enable CORS in your Lambda response:

```python
def lambda_handler(event, context):
    # Process request
    result = {'message': 'Success'}

    # Return response with CORS headers
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization'
        },
        'body': json.dumps(result)
    }
```

## Request/Response Examples

### API Gateway REST API Event

```json
{
    "resource": "/users/{id}",
    "path": "/users/123",
    "httpMethod": "GET",
    "headers": {
        "Accept": "application/json",
        "CloudFront-Viewer-Country": "US"
    },
    "pathParameters": {
        "id": "123"
    },
    "queryStringParameters": {
        "include": "profile"
    },
    "body": null,
    "isBase64Encoded": false
}
```

### Lambda Response Format

```json
{
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "X-Custom-Header": "value"
    },
    "body": "{\"message\": \"Success\"}",
    "isBase64Encoded": false
}
```

## Error Handling

```python
import json
import traceback

def lambda_handler(event, context):
    try:
        # Your business logic
        result = process_request(event)

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except ValueError as e:
        # Client error
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Bad Request',
                'message': str(e)
            })
        }

    except Exception as e:
        # Server error
        print(f"Error: {str(e)}")
        print(traceback.format_exc())

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            })
        }
```'''
    },
    {
        'title': 'AWS Lambda Best Practices',
        'url': 'https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html',
        'content': '''# AWS Lambda Best Practices

Follow these best practices to make the most effective use of AWS Lambda.

## Function Code

### Separate Business Logic from Handler

```python
# Good practice - testable business logic
def calculate_discount(price, customer_tier):
    discounts = {
        'gold': 0.20,
        'silver': 0.10,
        'bronze': 0.05
    }

    discount = discounts.get(customer_tier, 0)
    return price * (1 - discount)

def lambda_handler(event, context):
    # Handler only does I/O and coordination
    price = float(event['price'])
    tier = event['customer_tier']

    final_price = calculate_discount(price, tier)

    return {
        'statusCode': 200,
        'body': json.dumps({'final_price': final_price})
    }
```

### Minimize Cold Start Impact

```python
import boto3
import os

# Initialize outside handler (runs once per container)
s3_client = boto3.client('s3')
db_connection = create_db_connection()

BUCKET_NAME = os.environ.get('BUCKET_NAME')

def lambda_handler(event, context):
    # Use pre-initialized resources
    data = s3_client.get_object(Bucket=BUCKET_NAME, Key='data.json')
    result = db_connection.query('SELECT * FROM users')

    return {'statusCode': 200}
```

## Performance Optimization

### Use Environment Variables

```python
import os

# Load once at module level
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
API_KEY = os.environ.get('API_KEY')

def lambda_handler(event, context):
    # Use pre-loaded variables
    connect_to_database(DB_HOST, DB_NAME, API_KEY)
```

### Right-Size Your Function

- Start with 128 MB memory
- Monitor CloudWatch metrics
- Increase memory if CPU-bound (memory and CPU scale together)
- Use AWS Lambda Power Tuning tool

```python
# Check execution metrics
def lambda_handler(event, context):
    import time
    start_time = time.time()

    # Your code here
    process_data(event)

    duration = time.time() - start_time
    print(f"Execution time: {duration:.2f}s")
    print(f"Memory used: {context.memory_limit_in_mb}MB")
```

### Enable X-Ray Tracing

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch libraries
patch_all()

def lambda_handler(event, context):
    # Create subsegments for tracking
    with xray_recorder.capture('data_processing'):
        result = process_data(event)

    with xray_recorder.capture('database_query'):
        save_to_database(result)

    return {'statusCode': 200}
```

## Error Handling

### Implement Retry Logic

```python
import time
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            result = call_external_api(event)
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
        except ClientError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                # Final attempt failed
                raise
```

### Use Dead Letter Queues

Configure DLQ for failed async invocations:

```bash
aws lambda update-function-configuration \
    --function-name my-function \
    --dead-letter-config TargetArn=arn:aws:sqs:us-east-1:123456789012:my-dlq
```

## Security

### Use IAM Roles with Least Privilege

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::my-bucket/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:GetItem"
            ],
            "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/MyTable"
        }
    ]
}
```

### Store Secrets in AWS Secrets Manager

```python
import boto3
import json

secrets_client = boto3.client('secretsmanager')

def get_secret(secret_name):
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        raise

def lambda_handler(event, context):
    # Retrieve database credentials
    db_credentials = get_secret('prod/db/credentials')

    username = db_credentials['username']
    password = db_credentials['password']

    # Use credentials to connect
    connect_to_database(username, password)
```

## Monitoring and Logging

### Structured Logging

```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Structured log entries
    logger.info(json.dumps({
        'event': 'user_signup',
        'user_id': event.get('user_id'),
        'request_id': context.request_id,
        'timestamp': context.get_remaining_time_in_millis()
    }))

    try:
        result = process_signup(event)
        logger.info(json.dumps({
            'event': 'signup_success',
            'user_id': event.get('user_id')
        }))
        return result
    except Exception as e:
        logger.error(json.dumps({
            'event': 'signup_error',
            'error': str(e),
            'user_id': event.get('user_id')
        }))
        raise
```

## Cost Optimization

### Use Lambda Power Tuning
- Test different memory configurations
- Find optimal price/performance ratio

### Set Appropriate Timeouts
```python
# Don't use default 3 seconds for everything
# Set realistic timeouts based on function needs

# In configuration:
# Quick API calls: 10 seconds
# Data processing: 60-300 seconds
# Maximum: 900 seconds (15 minutes)
```

### Use Reserved Concurrency Carefully
Only set reserved concurrency when necessary to limit costs or prevent downstream overload.'''
    }
]