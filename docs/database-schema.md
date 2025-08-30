# Database Schema

This document describes the database models, relationships, and migration patterns used in the DocRag application.

## Schema Overview

The DocRag application uses PostgreSQL (production) or SQLite (development) with SQLAlchemy ORM for data persistence. The schema supports:

- **Chat conversations and history**
- **Document metadata and tracking**
- **Rate limiting and abuse prevention**

## Database Models

### Conversation Model

Stores user interactions and AI responses for chat history and analytics.

```python
class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), nullable=False)
    user_query = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)  # JSON string of source URLs
    response_time = db.Column(db.Float)  # Response time in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Integer)  # 1 for thumbs up, -1 for thumbs down
```

**Table Structure**:
```sql
CREATE TABLE conversation (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    user_query TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    sources TEXT,
    response_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feedback INTEGER
);
```

**Indexes**:
```sql
CREATE INDEX idx_conversation_session_id ON conversation(session_id);
CREATE INDEX idx_conversation_created_at ON conversation(created_at);
CREATE INDEX idx_conversation_feedback ON conversation(feedback) WHERE feedback IS NOT NULL;
```

**Relationships**:
- One session can have many conversations
- No foreign key constraints (session management is application-level)

**Usage Patterns**:
```python
# Get conversation history for session
conversations = Conversation.query.filter_by(session_id=session_id)\
    .order_by(Conversation.created_at.desc())\
    .limit(10).all()

# Store new conversation
conversation = Conversation(
    session_id=session_id,
    user_query=clean_query,
    ai_response=response_text,
    sources=json.dumps(source_urls),
    response_time=processing_time
)
db.session.add(conversation)
db.session.commit()
```

### DocumentChunk Model

Tracks processed document chunks and their metadata for deduplication and management.

```python
class DocumentChunk(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_url = db.Column(db.String(500), nullable=False)
    title = db.Column(db.String(200))
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    doc_type = db.Column(db.String(50))  # 'react', 'python', 'fastapi'
    version = db.Column(db.String(20))
    embedding_id = db.Column(db.String(100))  # ChromaDB document ID
    content_hash = db.Column(db.String(64))   # SHA-256 for idempotency
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**Table Structure**:
```sql
CREATE TABLE document_chunk (
    id VARCHAR(36) PRIMARY KEY,
    source_url VARCHAR(500) NOT NULL,
    title VARCHAR(200),
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    doc_type VARCHAR(50),
    version VARCHAR(20),
    embedding_id VARCHAR(100),
    content_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
```sql
CREATE INDEX idx_document_chunk_source_url ON document_chunk(source_url);
CREATE INDEX idx_document_chunk_content_hash ON document_chunk(content_hash);
CREATE INDEX idx_document_chunk_doc_type ON document_chunk(doc_type);
CREATE INDEX idx_document_chunk_embedding_id ON document_chunk(embedding_id);
CREATE UNIQUE INDEX idx_document_chunk_unique_content ON document_chunk(content_hash);
```

**Relationships**:
- Links to ChromaDB documents via `embedding_id`
- Content deduplication via `content_hash`

**Usage Patterns**:
```python
# Check if content already exists (idempotency)
existing_chunk = DocumentChunk.query.filter_by(content_hash=content_hash).first()
if existing_chunk:
    logger.info(f"Skipping duplicate content: {source_url}")
    return

# Store new chunk
chunk = DocumentChunk(
    source_url=url,
    title=title,
    content=content,
    chunk_index=index,
    doc_type=doc_type,
    content_hash=content_hash,
    embedding_id=embedding_id
)
db.session.add(chunk)
db.session.commit()
```

### RateLimit Model

Implements IP-based rate limiting for abuse prevention.

```python
class RateLimit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), nullable=False, unique=True)
    request_count = db.Column(db.Integer, default=0)
    last_request = db.Column(db.DateTime, default=datetime.utcnow)
    reset_time = db.Column(db.DateTime, default=datetime.utcnow)
```

**Table Structure**:
```sql
CREATE TABLE rate_limit (
    id SERIAL PRIMARY KEY,
    ip_address VARCHAR(45) NOT NULL UNIQUE,
    request_count INTEGER DEFAULT 0,
    last_request TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reset_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_rate_limit_ip_address ON rate_limit(ip_address);
CREATE INDEX idx_rate_limit_reset_time ON rate_limit(reset_time);
```

**Usage Patterns**:
```python
# Check and update rate limit
rate_limit = RateLimit.query.filter_by(ip_address=ip_address).first()

if not rate_limit:
    rate_limit = RateLimit(ip_address=ip_address)
    db.session.add(rate_limit)

# Reset counter if window expired
if datetime.utcnow() >= rate_limit.reset_time:
    rate_limit.request_count = 0
    rate_limit.reset_time = datetime.utcnow() + timedelta(minutes=1)

# Check limit
if rate_limit.request_count >= Config.RATE_LIMIT_PER_MINUTE:
    return False  # Rate limit exceeded

rate_limit.request_count += 1
rate_limit.last_request = datetime.utcnow()
db.session.commit()
```

## Database Configuration

### Connection Configuration

**Development (SQLite)**:
```python
SQLALCHEMY_DATABASE_URI = "sqlite:///docrag.db"
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
```

**Production (PostgreSQL)**:
```python
SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://user:pass@host:5432/dbname"
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_size": 10,
    "max_overflow": 20,
    "connect_args": {"sslmode": "require"}
}
```

### Connection Pooling

**Pool Settings**:
- **pool_size**: Number of persistent connections (default: 5)
- **max_overflow**: Additional connections when pool exhausted (default: 10)
- **pool_recycle**: Recreate connections after N seconds (default: 300)
- **pool_pre_ping**: Verify connections before use (default: True)

**Monitoring Connection Pool**:
```python
def get_db_pool_status():
    """Get database connection pool status."""
    engine = db.engine
    pool = engine.pool
    
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalidated": pool.invalidated()
    }
```

## Migration Management

### Alembic Configuration

The project uses Alembic for database schema migrations.

**Configuration** (`alembic.ini`):
```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME
```

### Migration Best Practices

#### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "add feedback column to conversation"

# Create empty migration for data changes
alembic revision -m "migrate legacy data format"
```

#### Migration Structure

```python
"""add feedback column to conversation

Revision ID: 001_add_feedback
Revises: 000_initial
Create Date: 2024-08-10 19:48:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001_add_feedback'
down_revision = '000_initial'
branch_labels = None
depends_on = None

def upgrade():
    """Add feedback column to conversation table."""
    # Add column with default value
    op.add_column('conversation', 
                  sa.Column('feedback', sa.Integer(), nullable=True))

def downgrade():
    """Remove feedback column from conversation table."""
    op.drop_column('conversation', 'feedback')
```

#### Data Migration Example

```python
"""migrate conversation sources to json format

Revision ID: 002_migrate_sources
Revises: 001_add_feedback
Create Date: 2024-08-10 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import json

def upgrade():
    """Convert pipe-separated sources to JSON format."""
    # Get current connection
    connection = op.get_bind()
    
    # Fetch conversations with old format
    result = connection.execute(
        "SELECT id, sources FROM conversation WHERE sources LIKE '%|%'"
    )
    
    for conversation_id, sources in result:
        if sources and '|' in sources:
            # Convert pipe-separated to JSON array
            source_list = [s.strip() for s in sources.split('|') if s.strip()]
            json_sources = json.dumps(source_list)
            
            # Update record
            connection.execute(
                "UPDATE conversation SET sources = %s WHERE id = %s",
                (json_sources, conversation_id)
            )

def downgrade():
    """Convert JSON sources back to pipe-separated format."""
    connection = op.get_bind()
    
    result = connection.execute(
        "SELECT id, sources FROM conversation WHERE sources LIKE '[%'"
    )
    
    for conversation_id, sources in result:
        if sources and sources.startswith('['):
            try:
                source_list = json.loads(sources)
                pipe_sources = ' | '.join(source_list)
                
                connection.execute(
                    "UPDATE conversation SET sources = %s WHERE id = %s",
                    (pipe_sources, conversation_id)
                )
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
```

### Migration Deployment

#### Development
```bash
# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history --verbose
```

#### Production
```bash
# Always backup before migrations
pg_dump $DATABASE_URL > backup_before_migration.sql

# Apply migrations
alembic upgrade head

# Verify migration success
alembic current
psql $DATABASE_URL -c "\dt"  # List tables
```

## Query Optimization

### Indexing Strategy

**Primary Indexes** (automatic):
- All primary key columns
- Unique constraint columns

**Performance Indexes**:
```sql
-- Conversation queries
CREATE INDEX idx_conversation_session_created ON conversation(session_id, created_at DESC);

-- Document chunk queries  
CREATE INDEX idx_document_chunk_type_url ON document_chunk(doc_type, source_url);

-- Rate limiting queries
CREATE INDEX idx_rate_limit_ip_reset ON rate_limit(ip_address, reset_time);
```

**Partial Indexes** (PostgreSQL):
```sql
-- Index only conversations with feedback
CREATE INDEX idx_conversation_feedback_only ON conversation(feedback, created_at) 
WHERE feedback IS NOT NULL;

-- Index only active rate limits
CREATE INDEX idx_rate_limit_active ON rate_limit(ip_address, request_count) 
WHERE reset_time > NOW();
```

### Query Patterns

#### Efficient Conversation Retrieval

```python
# Good: Use indexes effectively
conversations = db.session.query(Conversation)\
    .filter(Conversation.session_id == session_id)\
    .order_by(Conversation.created_at.desc())\
    .limit(10)\
    .all()

# Bad: Inefficient query without proper filtering
# all_conversations = Conversation.query.all()
# user_conversations = [c for c in all_conversations if c.session_id == session_id]
```

#### Batch Operations

```python
# Good: Batch insert for multiple chunks
chunks_to_insert = []
for chunk_data in document_chunks:
    chunk = DocumentChunk(**chunk_data)
    chunks_to_insert.append(chunk)

db.session.bulk_save_objects(chunks_to_insert)
db.session.commit()

# Bad: Individual inserts in loop
# for chunk_data in document_chunks:
#     chunk = DocumentChunk(**chunk_data)
#     db.session.add(chunk)
#     db.session.commit()  # Commits in loop
```

#### Avoiding N+1 Queries

```python
# Good: Use joins to avoid N+1
conversations = db.session.query(Conversation)\
    .options(db.joinedload(Conversation.related_data))\
    .filter(Conversation.session_id == session_id)\
    .all()

# Bad: N+1 query pattern
# conversations = Conversation.query.filter_by(session_id=session_id).all()
# for conv in conversations:
#     related_data = conv.related_data  # Triggers additional query
```

## Database Maintenance

### Cleanup Operations

#### Old Conversation Cleanup

```python
def cleanup_old_conversations(days_to_keep: int = 30):
    """Remove conversations older than specified days."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    deleted_count = db.session.query(Conversation)\
        .filter(Conversation.created_at < cutoff_date)\
        .delete()
    
    db.session.commit()
    logger.info(f"Cleaned up {deleted_count} old conversations")
    
    return deleted_count
```

#### Orphaned Chunk Cleanup

```python
def cleanup_orphaned_chunks():
    """Remove document chunks not referenced by ChromaDB."""
    from rag_engine import RAGEngine
    
    rag = RAGEngine()
    
    # Get all embedding IDs from ChromaDB
    collection_data = rag.collection.get()
    active_embedding_ids = set(collection_data['ids'])
    
    # Find orphaned chunks
    orphaned_chunks = DocumentChunk.query\
        .filter(~DocumentChunk.embedding_id.in_(active_embedding_ids))\
        .all()
    
    # Delete orphaned chunks
    for chunk in orphaned_chunks:
        db.session.delete(chunk)
    
    db.session.commit()
    logger.info(f"Cleaned up {len(orphaned_chunks)} orphaned chunks")
```

### Database Statistics

```python
def get_database_stats():
    """Get database usage statistics."""
    stats = {}
    
    # Table counts
    stats['conversation_count'] = db.session.query(Conversation).count()
    stats['document_chunk_count'] = db.session.query(DocumentChunk).count()
    stats['rate_limit_count'] = db.session.query(RateLimit).count()
    
    # Storage usage (PostgreSQL)
    if 'postgresql' in Config.get_database_uri():
        result = db.session.execute("""
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats 
            WHERE schemaname = 'public'
        """)
        stats['pg_stats'] = [dict(row) for row in result]
    
    return stats
```

## Data Integrity

### Constraints and Validation

#### Database Constraints

```sql
-- Ensure positive response times
ALTER TABLE conversation ADD CONSTRAINT check_positive_response_time 
CHECK (response_time IS NULL OR response_time >= 0);

-- Ensure valid feedback values
ALTER TABLE conversation ADD CONSTRAINT check_valid_feedback 
CHECK (feedback IS NULL OR feedback IN (-1, 1));

-- Ensure non-negative chunk index
ALTER TABLE document_chunk ADD CONSTRAINT check_chunk_index_non_negative 
CHECK (chunk_index >= 0);

-- Ensure valid doc types
ALTER TABLE document_chunk ADD CONSTRAINT check_valid_doc_type 
CHECK (doc_type IN ('react', 'python', 'fastapi'));
```

#### Application-Level Validation

```python
def validate_conversation_data(data: dict) -> bool:
    """Validate conversation data before database insert."""
    required_fields = ['session_id', 'user_query', 'ai_response']
    
    for field in required_fields:
        if field not in data or not data[field]:
            raise ValueError(f"Required field missing: {field}")
    
    # Validate session_id format (UUID)
    try:
        uuid.UUID(data['session_id'])
    except ValueError:
        raise ValueError("Invalid session_id format")
    
    # Validate response_time if provided
    if 'response_time' in data and data['response_time'] is not None:
        if not isinstance(data['response_time'], (int, float)) or data['response_time'] < 0:
            raise ValueError("response_time must be a non-negative number")
    
    return True
```

### Content Hash Generation

```python
import hashlib
import json

def generate_content_hash(content: str, metadata: dict = None) -> str:
    """Generate SHA-256 hash for content deduplication."""
    # Normalize content
    normalized_content = content.strip().lower()
    
    # Include relevant metadata in hash
    hash_data = {
        'content': normalized_content,
        'metadata': metadata or {}
    }
    
    # Create hash
    hash_input = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

# Usage in document processing
content_hash = generate_content_hash(
    content=chunk_content,
    metadata={'source_url': url, 'doc_type': doc_type}
)
```

## Performance Optimization

### Query Performance

#### Explain Query Plans

```python
def analyze_query_performance():
    """Analyze slow queries for optimization."""
    # PostgreSQL query analysis
    if 'postgresql' in Config.get_database_uri():
        slow_queries = db.session.execute("""
            SELECT query, calls, total_time, mean_time
            FROM pg_stat_statements 
            WHERE mean_time > 1000  -- Queries slower than 1s
            ORDER BY mean_time DESC
            LIMIT 10
        """).fetchall()
        
        for query in slow_queries:
            logger.warning(f"Slow query detected: {query.query[:100]}... "
                          f"(avg: {query.mean_time:.2f}ms)")
```

#### Index Usage Monitoring

```sql
-- Check index usage (PostgreSQL)
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0;  -- Unused indexes
```

### Connection Pool Monitoring

```python
def monitor_connection_pool():
    """Monitor database connection pool health."""
    engine = db.engine
    pool = engine.pool
    
    pool_stats = {
        'size': pool.size(),
        'checked_in': pool.checkedin(),
        'checked_out': pool.checkedout(),
        'overflow': pool.overflow(),
        'utilization': pool.checkedout() / (pool.size() + pool.overflow())
    }
    
    # Alert if pool utilization is high
    if pool_stats['utilization'] > 0.8:
        logger.warning(f"High database pool utilization: {pool_stats}")
    
    return pool_stats
```

## Backup and Recovery

### Backup Strategy

#### Automated Backups

```python
def create_database_backup():
    """Create database backup with metadata."""
    import subprocess
    from datetime import datetime
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_file = f"/var/backups/docrag/backup_{timestamp}.sql"
    
    # Create backup
    cmd = f"pg_dump {Config.DATABASE_URL} > {backup_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Compress backup
        subprocess.run(f"gzip {backup_file}", shell=True)
        logger.info(f"Database backup created: {backup_file}.gz")
        
        # Store backup metadata
        backup_info = DocumentChunk(
            source_url=f"backup://{backup_file}.gz",
            title="Database Backup",
            content=f"Backup created at {timestamp}",
            doc_type="backup",
            content_hash=generate_content_hash(timestamp)
        )
        db.session.add(backup_info)
        db.session.commit()
        
        return backup_file + ".gz"
    else:
        logger.error(f"Backup failed: {result.stderr}")
        raise Exception(f"Backup failed: {result.stderr}")
```

#### Point-in-Time Recovery

For PostgreSQL with WAL-E or similar:

```bash
# Enable WAL archiving in postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'

# Create base backup
pg_basebackup -D /var/backups/postgres/base -Ft -z -P

# Restore to specific time
pg_ctl stop -D /var/lib/postgresql/data
rm -rf /var/lib/postgresql/data/*
tar -xzf /var/backups/postgres/base/base.tar.gz -C /var/lib/postgresql/data/
```

### Recovery Testing

```python
def test_backup_recovery():
    """Test backup and recovery procedures."""
    # Create test data
    test_conversation = Conversation(
        session_id=str(uuid.uuid4()),
        user_query="Test query",
        ai_response="Test response"
    )
    db.session.add(test_conversation)
    db.session.commit()
    
    original_id = test_conversation.id
    
    # Create backup
    backup_file = create_database_backup()
    
    # Simulate data loss
    db.session.delete(test_conversation)
    db.session.commit()
    
    # Restore from backup
    restore_database_backup(backup_file)
    
    # Verify recovery
    recovered_conversation = Conversation.query.get(original_id)
    assert recovered_conversation is not None
    assert recovered_conversation.user_query == "Test query"
    
    logger.info("Backup recovery test passed")
```

---

*This database schema documentation provides the foundation for understanding data persistence in DocRag. Keep this updated as the schema evolves with new features and optimizations.*