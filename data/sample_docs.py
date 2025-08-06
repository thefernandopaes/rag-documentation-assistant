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