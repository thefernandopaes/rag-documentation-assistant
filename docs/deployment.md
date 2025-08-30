# Deployment Guide

This guide covers production deployment procedures and best practices for the DocRag application.

## Deployment Overview

DocRag is designed for cloud deployment with support for:
- **Railway** (primary recommendation)
- **Docker containers**
- **Traditional VPS/servers**
- **Kubernetes clusters**

## Railway Deployment (Recommended)

Railway provides the simplest deployment experience with built-in PostgreSQL and volume storage.

### Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Push your code to GitHub
3. **OpenAI API Key**: With sufficient credits

### Deployment Steps

#### 1. Create Railway Project

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and create project
railway login
railway init
```

#### 2. Add PostgreSQL Database

1. Go to Railway dashboard
2. Click "Add Service" → "Database" → "PostgreSQL"
3. Note the database connection details

#### 3. Configure Environment Variables

Set these environment variables in Railway dashboard:

```env
# --- Required Secrets ---
OPENAI_API_KEY=sk-your-openai-api-key
SESSION_SECRET=your-64-hex-production-secret
ADMIN_API_KEY=your-admin-api-key

# --- Database (Auto-configured by Railway) ---
DATABASE_URL=${{Postgres.DATABASE_URL}}
DB_SSLMODE=require

# --- Production Settings ---
ENV=production
CHROMA_DB_PATH=/var/lib/docrag/chroma_db
WEB_CONCURRENCY=2
LOG_LEVEL=info

# --- RAG Configuration ---
CHUNK_SIZE=800
CHUNK_OVERLAP=150
MAX_RESPONSE_TOKENS=700
TEMPERATURE=0.25
DOC_USE_SAMPLE=false
```

#### 4. Configure Volume Storage

1. Add volume in Railway dashboard
2. Mount path: `/var/lib/docrag`
3. Set `CHROMA_DB_PATH=/var/lib/docrag/chroma_db`

#### 5. Deploy Application

Railway auto-deploys from your GitHub repository:

1. Connect GitHub repository
2. Railway detects Python app automatically
3. Uses `Procfile` for process configuration:

```procfile
web: gunicorn -c gunicorn.conf.py app:app
```

#### 6. Run Database Migrations

Create a one-time deployment or worker service:

```bash
# Railway CLI method
railway run alembic upgrade head

# Or create worker service with command:
# alembic upgrade head && python scripts/ingest.py --max-pages 300
```

#### 7. Initialize Document Database

```bash
# Via Railway CLI
railway run python scripts/ingest.py --max-pages 300

# Or via API (replace YOUR_DOMAIN and API_KEY)
curl -X POST https://YOUR_DOMAIN/api/initialize \
  -H "Authorization: Bearer YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sources": ["react", "python", "fastapi"]}'
```

### Railway Configuration Files

**Procfile**:
```
web: gunicorn -c gunicorn.conf.py app:app
worker: python scripts/ingest.py --max-pages 300
```

**railway.toml** (optional):
```toml
[build]
builder = "NIXPACKS"

[deploy]
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /var/lib/docrag/chroma_db
RUN mkdir -p cache

# Set environment variables
ENV ENV=production
ENV CHROMA_DB_PATH=/var/lib/docrag/chroma_db
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app /var/lib/docrag
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

# Start command
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ENV=production
      - DATABASE_URL=postgresql://docrag:password@db:5432/docrag
      - CHROMA_DB_PATH=/var/lib/docrag/chroma_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SESSION_SECRET=${SESSION_SECRET}
      - ADMIN_API_KEY=${ADMIN_API_KEY}
    volumes:
      - chroma_data:/var/lib/docrag
      - cache_data:/app/cache
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=docrag
      - POSTGRES_USER=docrag
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Optional: Run ingestion as separate service
  ingestion:
    build: .
    command: ["python", "scripts/ingest.py", "--max-pages", "300"]
    environment:
      - DATABASE_URL=postgresql://docrag:password@db:5432/docrag
      - CHROMA_DB_PATH=/var/lib/docrag/chroma_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DOC_USE_SAMPLE=false
    volumes:
      - chroma_data:/var/lib/docrag
    depends_on:
      - db
    profiles: ["ingestion"]

volumes:
  postgres_data:
  chroma_data:
  cache_data:
```

### Running with Docker

```bash
# Build and run
docker-compose up -d

# Run database migrations
docker-compose exec app alembic upgrade head

# Run document ingestion
docker-compose --profile ingestion up ingestion

# View logs
docker-compose logs -f app
```

## Traditional Server Deployment

### Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.11+
- PostgreSQL 12+
- Nginx (for reverse proxy)
- SSL certificate (Let's Encrypt recommended)

### Server Setup

#### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql postgresql-contrib nginx certbot python3-certbot-nginx \
    git supervisor redis-server
```

#### 2. Application Setup

```bash
# Create application user
sudo useradd --system --shell /bin/bash --home /opt/docrag docrag

# Clone repository
sudo -u docrag git clone https://github.com/your-org/docrag.git /opt/docrag/app
cd /opt/docrag/app

# Setup virtual environment
sudo -u docrag python3.11 -m venv venv
sudo -u docrag venv/bin/pip install -r requirements.txt
```

#### 3. Database Setup

```bash
# Create database and user
sudo -u postgres createdb docrag
sudo -u postgres createuser docrag
sudo -u postgres psql -c "ALTER USER docrag WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE docrag TO docrag;"
```

#### 4. Environment Configuration

```bash
# Create production environment file
sudo -u docrag tee /opt/docrag/app/.env <<EOF
ENV=production
DATABASE_URL=postgresql://docrag:secure_password@localhost:5432/docrag
CHROMA_DB_PATH=/opt/docrag/data/chroma_db
OPENAI_API_KEY=your-openai-api-key
SESSION_SECRET=your-64-hex-secret
ADMIN_API_KEY=your-admin-api-key
EOF

sudo chmod 600 /opt/docrag/app/.env
```

#### 5. Supervisor Configuration

```ini
# /etc/supervisor/conf.d/docrag.conf
[program:docrag]
command=/opt/docrag/app/venv/bin/gunicorn -c gunicorn.conf.py app:app
directory=/opt/docrag/app
user=docrag
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/docrag/app.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PATH="/opt/docrag/app/venv/bin"
```

#### 6. Nginx Configuration

```nginx
# /etc/nginx/sites-available/docrag
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Proxy to application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files (if serving directly)
    location /static {
        alias /opt/docrag/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### 7. SSL Certificate

```bash
# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Environment-Specific Configuration

### Development
```env
ENV=development
DATABASE_URL=sqlite:///docrag.db
CHROMA_DB_PATH=./chroma_db
DOC_USE_SAMPLE=true
LOG_LEVEL=debug
```

### Staging
```env
ENV=staging
DATABASE_URL=postgresql://user:pass@staging-db:5432/docrag
CHROMA_DB_PATH=/var/lib/docrag/chroma_db
DOC_USE_SAMPLE=false
DOC_MAX_PAGES_PER_SOURCE=50
LOG_LEVEL=info
```

### Production
```env
ENV=production
DATABASE_URL=postgresql://user:pass@prod-db:5432/docrag
DB_SSLMODE=require
CHROMA_DB_PATH=/var/lib/docrag/chroma_db
DOC_USE_SAMPLE=false
DOC_MAX_PAGES_PER_SOURCE=300
WEB_CONCURRENCY=4
LOG_LEVEL=warning
```

## Deployment Checklist

### Pre-Deployment

- [ ] Code reviewed and approved
- [ ] Tests passing in CI
- [ ] Database migrations tested
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Backup procedures in place

### Deployment

- [ ] Database backup created
- [ ] Maintenance mode enabled (if needed)
- [ ] Application deployed
- [ ] Database migrations applied
- [ ] Health checks passing
- [ ] Document ingestion completed
- [ ] Maintenance mode disabled

### Post-Deployment

- [ ] Application responding correctly
- [ ] All endpoints functional
- [ ] Performance metrics normal
- [ ] Error rates acceptable
- [ ] Monitoring alerts configured
- [ ] Team notified of successful deployment

## Rollback Procedures

### Quick Rollback

```bash
# Railway: Redeploy previous version
railway deploy --service=web --environment=production <previous-commit-hash>

# Docker: Rollback to previous image
docker-compose down
docker-compose up -d --scale app=0
docker tag docrag:previous docrag:latest
docker-compose up -d
```

### Database Rollback

```bash
# If migrations need to be rolled back
alembic downgrade -1

# Or to specific revision
alembic downgrade <revision_id>
```

### Emergency Procedures

1. **Immediate Response**
   - Take application offline if critical issue
   - Enable maintenance page
   - Alert team via communication channels

2. **Diagnosis**
   - Check application logs
   - Verify database connectivity
   - Monitor resource usage
   - Review recent changes

3. **Resolution**
   - Apply hotfix if quick fix available
   - Rollback to previous version if needed
   - Communicate status to users

## Monitoring and Alerting

### Health Monitoring

Set up monitoring for:
- **Application health**: `/healthz` endpoint
- **Response times**: API endpoint performance
- **Error rates**: 4xx/5xx response monitoring
- **Resource usage**: CPU, memory, disk usage
- **Database performance**: Connection pool, query times

### Log Aggregation

```bash
# Example: Centralized logging with syslog
# /etc/rsyslog.d/50-docrag.conf
:programname, isequal, "docrag" /var/log/docrag/app.log
& stop
```

### Alerting Rules

Configure alerts for:
- **High error rates**: >5% 5xx responses
- **Slow responses**: >5s average response time
- **High memory usage**: >80% memory utilization
- **Database issues**: Connection failures, slow queries
- **Rate limit abuse**: Excessive rate limit violations

## Performance Optimization

### Database Optimization

```sql
-- Recommended indexes for production
CREATE INDEX CONCURRENTLY idx_conversation_session_id ON conversation(session_id);
CREATE INDEX CONCURRENTLY idx_conversation_created_at ON conversation(created_at);
CREATE INDEX CONCURRENTLY idx_document_chunk_source_url ON document_chunk(source_url);
CREATE INDEX CONCURRENTLY idx_document_chunk_content_hash ON document_chunk(content_hash);
CREATE INDEX CONCURRENTLY idx_rate_limit_ip_address ON rate_limit(ip_address);
```

### Application Optimization

**Gunicorn Configuration** (`gunicorn.conf.py`):
```python
import os

# Worker configuration
bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
workers = int(os.getenv('WEB_CONCURRENCY', 2))
threads = int(os.getenv('WEB_THREADS', 1))
worker_class = "sync"

# Performance tuning
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 120
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
```

### CDN Configuration

For static assets (if serving separately):

```nginx
# Cache static assets
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header X-Content-Type-Options nosniff;
}
```

## Security Hardening

### SSL/TLS Configuration

```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;
```

### Firewall Configuration

```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Security Headers

Configured in `app.py`:
```python
@app.after_request
def add_security_headers(response):
    if Config._is_production():
        response.headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
        })
    return response
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR="/var/backups/docrag"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/docrag_backup_$TIMESTAMP.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### Vector Database Backup

```bash
#!/bin/bash
# backup-chroma.sh

BACKUP_DIR="/var/backups/docrag"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHROMA_BACKUP="$BACKUP_DIR/chroma_backup_$TIMESTAMP.tar.gz"

# Create backup
tar -czf $CHROMA_BACKUP -C /var/lib/docrag chroma_db/

# Keep only last 3 backups (large files)
find $BACKUP_DIR -name "chroma_backup_*.tar.gz" | sort | head -n -3 | xargs rm -f

echo "ChromaDB backup completed: $CHROMA_BACKUP"
```

### Automated Backup Schedule

```bash
# Cron jobs for automated backups
# Edit with: sudo crontab -e

# Daily database backup at 2 AM
0 2 * * * /opt/docrag/scripts/backup-db.sh

# Weekly ChromaDB backup on Sundays at 3 AM
0 3 * * 0 /opt/docrag/scripts/backup-chroma.sh
```

### Recovery Procedures

**Database Recovery**:
```bash
# Stop application
sudo supervisorctl stop docrag

# Restore database
gunzip -c /var/backups/docrag/docrag_backup_20240810_020000.sql.gz | psql $DATABASE_URL

# Restart application
sudo supervisorctl start docrag
```

**Vector Database Recovery**:
```bash
# Stop application
sudo supervisorctl stop docrag

# Restore ChromaDB
cd /var/lib/docrag
sudo rm -rf chroma_db/
sudo tar -xzf /var/backups/docrag/chroma_backup_20240810_030000.tar.gz

# Fix permissions
sudo chown -R docrag:docrag chroma_db/

# Restart application
sudo supervisorctl start docrag
```

## Troubleshooting

### Common Deployment Issues

#### 1. Port Binding Issues

```bash
# Check if port is in use
sudo netstat -tlnp | grep :5000

# Kill process using port
sudo kill <PID>

# Or change port in configuration
export PORT=8000
```

#### 2. Database Connection Issues

```bash
# Test database connection
psql $DATABASE_URL -c "SELECT 1;"

# Check PostgreSQL service
sudo systemctl status postgresql

# Review connection pool settings
tail -f /var/log/postgresql/postgresql-*.log
```

#### 3. ChromaDB Permission Issues

```bash
# Fix permissions
sudo chown -R docrag:docrag /var/lib/docrag/chroma_db
sudo chmod -R 755 /var/lib/docrag/chroma_db

# Check disk space
df -h /var/lib/docrag
```

#### 4. SSL Certificate Issues

```bash
# Test SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Renew certificate
sudo certbot renew --dry-run

# Check certificate expiration
sudo certbot certificates
```

### Log Analysis

```bash
# Application logs
tail -f /var/log/docrag/app.log

# Nginx access logs
tail -f /var/log/nginx/access.log

# Error logs
tail -f /var/log/nginx/error.log

# System logs
journalctl -u docrag -f
```

---

*This deployment guide covers the most common deployment scenarios. Adapt the configuration based on your specific infrastructure requirements and security policies.*