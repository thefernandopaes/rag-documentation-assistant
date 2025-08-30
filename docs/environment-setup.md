# Environment Setup

This guide covers setting up the development environment for the DocRag project from scratch.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **PostgreSQL**: 12+ (for production) or SQLite (for development)
- **Git**: Latest version
- **OpenAI API**: Valid API key with credits

### Platform-Specific Requirements

#### Windows
```bash
# Install Python from python.org or via Microsoft Store
# Install Git from git-scm.com
# Install PostgreSQL from postgresql.org (optional for development)
```

#### macOS
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 git postgresql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git postgresql postgresql-contrib
```

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-documentation-assistant
```

### 2. Create Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

#### Method 1: From pyproject.toml (Recommended)
```bash
# Extract dependencies from pyproject.toml and install
python - <<'PY'
import tomllib
print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))
PY | pip install -r /dev/stdin
```

#### Method 2: Manual Installation
```bash
pip install alembic>=1.13.2
pip install beautifulsoup4>=4.13.4
pip install chromadb>=1.0.15
pip install email-validator>=2.2.0
pip install flask>=3.1.1
pip install flask-sqlalchemy>=3.1.1
pip install gunicorn>=23.0.0
pip install langchain>=0.3.27
pip install langchain-community>=0.3.27
pip install langchain-openai>=0.3.28
pip install numpy>=2.3.2
pip install openai>=1.99.1
pip install psycopg2-binary>=2.9.10
pip install python-dotenv>=1.1.1
pip install requests>=2.32.4
pip install sqlalchemy>=2.0.42
pip install tiktoken>=0.10.0
pip install trafilatura>=2.0.0
pip install werkzeug>=3.1.3
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# --- Secrets ---
OPENAI_API_KEY=sk-your-openai-api-key-here
SESSION_SECRET=your-64-hex-secret-here
ADMIN_API_KEY=your-64-hex-admin-key-here

# --- Database (Development) ---
DATABASE_URL=sqlite:///docrag.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost:5432/docrag

# --- ChromaDB ---
CHROMA_DB_PATH=./chroma_db

# --- RAG Settings ---
CHUNK_SIZE=800
CHUNK_OVERLAP=150
MAX_RESPONSE_TOKENS=700
TEMPERATURE=0.25

# --- Development Settings ---
DOC_USE_SAMPLE=true
ENV=development
```

#### Generating Secrets

**Session Secret (64 hex characters)**:
```python
# Python method
import secrets
print(secrets.token_hex(32))
```

```bash
# OpenSSL method
openssl rand -hex 32
```

**Admin API Key**:
```python
# Python method
import secrets
print(secrets.token_urlsafe(32))
```

### 5. Database Setup

#### SQLite (Development - Default)
```bash
# Initialize database
alembic upgrade head
```

#### PostgreSQL (Production-like)

**Install PostgreSQL** (if not already installed):

Windows:
```powershell
# Download installer from postgresql.org
# Or use Chocolatey
choco install postgresql
```

macOS:
```bash
brew install postgresql
brew services start postgresql
```

Linux:
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Create Database**:
```bash
# Connect as postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE docrag;
CREATE USER docrag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE docrag TO docrag_user;
\q
```

**Update .env**:
```env
DATABASE_URL=postgresql://docrag_user:your_password@localhost:5432/docrag
DB_SSLMODE=disable
```

**Run Migrations**:
```bash
alembic upgrade head
```

### 6. Verify Installation

#### Check Python Environment
```bash
python --version  # Should show 3.11+
pip list | grep -E "(flask|openai|chromadb)"
```

#### Test Database Connection
```python
python - <<'PY'
from app import create_app
from config import Config

try:
    Config.validate_config()
    app = create_app()
    print("✅ Configuration valid")
    print("✅ Application created successfully")
    print("✅ Database connection works")
except Exception as e:
    print(f"❌ Error: {e}")
PY
```

#### Test OpenAI Connection
```python
python - <<'PY'
from openai import OpenAI
from config import Config
import os

try:
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test"
    )
    print("✅ OpenAI connection successful")
    print(f"✅ Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ OpenAI Error: {e}")
PY
```

## Running the Application

### Development Mode

```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows

# Run the application
python main.py
```

The application will be available at: http://localhost:5000

### Production Mode

```bash
# Set production environment
export ENV=production
export SESSION_SECRET=your-production-secret
export ADMIN_API_KEY=your-production-admin-key

# Use Gunicorn
gunicorn -c gunicorn.conf.py app:app
```

## Data Ingestion

### Sample Data (Development)

With `DOC_USE_SAMPLE=true` in your `.env`:

```bash
python scripts/ingest.py --use-sample
```

This will load predefined sample documentation chunks.

### Real Documentation (Production)

```bash
# Full ingestion (may take time)
DOC_USE_SAMPLE=false python scripts/ingest.py --max-pages 300

# Quick test with fewer pages
DOC_USE_SAMPLE=false python scripts/ingest.py --max-pages 10
```

## Development Tools

### Optional Development Dependencies

```bash
# Code formatting and linting
pip install black isort flake8 mypy

# Testing
pip install pytest pytest-cov

# Development utilities
pip install ipython python-dotenv
```

### IDE Configuration

#### VS Code Settings
Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": false
    }
}
```

#### PyCharm Configuration
1. Open project
2. File → Settings → Project → Python Interpreter
3. Add Interpreter → Existing environment
4. Select `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version

# If using system Python 3.11 explicitly
python3.11 -m venv .venv

# On Windows, try py launcher
py -3.11 -m venv .venv
```

#### Permission Issues (Linux/macOS)
```bash
# If pip install fails with permissions
pip install --user <package>

# Or use sudo (not recommended)
sudo pip install <package>
```

#### ChromaDB Issues
```bash
# If ChromaDB initialization fails
rm -rf ./chroma_db
mkdir chroma_db

# Check disk space and permissions
ls -la chroma_db
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
pg_isready -h localhost -p 5432

# Check if service is running
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # macOS
```

#### OpenAI API Issues
```bash
# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Environment Variables Not Loading

Check `.env` file location and format:
```bash
# File should be in project root
ls -la .env

# Check for syntax errors
cat .env | grep -E "^[A-Z_]+=.*"

# Test loading
python -c "from dotenv import load_dotenv; load_dotenv(); print('OK')"
```

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process if needed
kill <PID>  # macOS/Linux
taskkill /PID <PID>  # Windows
```

## Next Steps

After successful setup:

1. **Read the Architecture**: [System Architecture](./architecture.md)
2. **Configure for Production**: [Configuration Management](./configuration-management.md)
3. **Set up Monitoring**: [Monitoring & Logging](./monitoring-logging.md)
4. **Review Security**: [Security Guidelines](./security-guidelines.md)

## Development Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install any new dependencies
pip install -r <(python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# 4. Run migrations
alembic upgrade head

# 5. Start development server
python main.py
```

---

*This setup guide should get you running quickly. For production deployment, see the [Deployment Guide](./deployment.md).*