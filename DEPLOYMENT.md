# Deployment Guide - FastAPI Migration

Production deployment strategy for Flask → FastAPI migration with zero downtime.

## Table of Contents

1. [Overview](#overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Blue-Green Deployment Strategy](#blue-green-deployment-strategy)
4. [Deployment Steps](#deployment-steps)
5. [Monitoring](#monitoring)
6. [Rollback Procedure](#rollback-procedure)
7. [Post-Deployment](#post-deployment)

---

## Overview

### Architecture

```
┌─────────────┐
│   Nginx     │  Load Balancer / Reverse Proxy
│ (Port 80)   │
└──────┬──────┘
       │
       ├───────────────────┬───────────────────┐
       │                   │                   │
┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
│   Flask     │    │  FastAPI    │    │  FastAPI    │
│ (Port 5000) │    │ (Port 8000) │    │ (Port 8001) │
│   (Old)     │    │   Worker 1  │    │   Worker 2  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Deployment Strategy: Blue-Green

**Blue**: Current production (Flask on port 5000)
**Green**: New version (FastAPI on port 8000)

Traffic is gradually shifted from Blue to Green while monitoring metrics.

---

## Pre-Deployment Checklist

### 1. Code Validation

```bash
# Run all tests
pytest tests/ -v

# Check test coverage
pytest --cov=. --cov-report=html

# Validate configuration
python -c "from config import Config; Config.validate_config()"

# Check dependencies
pip check
```

### 2. Database Preparation

```bash
# Backup database
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migrations (if any)
alembic upgrade head

# Verify database connectivity
python -c "from database_async import check_db_connection; import asyncio; asyncio.run(check_db_connection())"
```

### 3. Environment Variables

Verify all required environment variables are set:

- `OPENAI_API_KEY`
- `DATABASE_URL`
- `SESSION_SECRET`
- `ADMIN_API_KEY`
- `WEB_CONCURRENCY` (number of workers)
- `PORT` (8000 for FastAPI)
- `ENVIRONMENT=production`

### 4. Performance Baseline

Record current metrics before deployment:

```bash
# Response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/api/stats

# Concurrent users test
python tests/test_performance.py
```

---

## Blue-Green Deployment Strategy

### Phase 1: Deploy Green (0% traffic)

```bash
# Start FastAPI on port 8000
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Nginx config remains unchanged** - all traffic still goes to Flask.

**Verification:**
- Health check: `curl http://localhost:8000/health`
- Stats: `curl http://localhost:8000/api/stats`
- Sample query: Test /api/chat endpoint

### Phase 2: Gradual Traffic Shift (10% → 100%)

#### Day 1: 10% Traffic

```nginx
upstream backend {
    server localhost:5000 weight=90;  # Flask (90%)
    server localhost:8000 weight=10;  # FastAPI (10%)
}
```

**Monitor for 4-6 hours:**
- Error rate < 0.1%
- Response time < 2s
- No critical issues

#### Day 2: 25% Traffic

```nginx
upstream backend {
    server localhost:5000 weight=75;
    server localhost:8000 weight=25;
}
```

Monitor for 4-6 hours.

#### Day 3: 50% Traffic

```nginx
upstream backend {
    server localhost:5000 weight=50;
    server localhost:8000 weight=50;
}
```

Monitor for 6-12 hours (peak + off-peak).

#### Day 4: 75% Traffic

```nginx
upstream backend {
    server localhost:5000 weight=25;
    server localhost:8000 weight=75;
}
```

Monitor for 6-12 hours.

#### Day 5: 100% Traffic

```nginx
upstream backend {
    # Flask removed
    server localhost:8000 weight=100;  # FastAPI only
}
```

Monitor for 24-48 hours before decommissioning Flask.

### Phase 3: Decommission Blue (Flask)

After 48 hours of stable 100% FastAPI traffic:

```bash
# Stop Flask/Gunicorn
pkill -f gunicorn

# Remove from Nginx config
# (remove Flask backend)

# Reload Nginx
nginx -t && nginx -s reload

# Archive Flask code
git tag flask-deprecated-$(date +%Y%m%d)
```

---

## Deployment Steps

### Step 1: Prepare Environment

```bash
# SSH into production server
ssh user@production-server

# Navigate to app directory
cd /var/www/rag-documentation-assistant

# Pull latest code
git fetch origin
git checkout main
git pull origin main

# Activate virtualenv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Pre-Flight Checks

```bash
# Run validation script
python test_phase7_testing.py

# Test FastAPI locally
uvicorn fastapi_app:app --port 8001 --workers 1 &
PID=$!

# Test endpoints
curl http://localhost:8001/health
curl http://localhost:8001/api/stats

# Kill test server
kill $PID
```

### Step 3: Start FastAPI (Green)

```bash
# Option A: Using systemd service
sudo systemctl start fastapi

# Option B: Using uvicorn directly
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4 &

# Option C: Using supervisor
supervisorctl start fastapi

# Verify it's running
curl http://localhost:8000/health
```

### Step 4: Update Nginx Configuration

```bash
# Backup current config
sudo cp /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/default.backup

# Update with blue-green config (10% traffic)
sudo nano /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Reload Nginx
sudo nginx -s reload
```

### Step 5: Monitor Metrics

```bash
# Watch logs
tail -f /var/log/nginx/access.log
tail -f /var/log/fastapi/app.log

# Monitor health
watch -n 5 'curl -s http://localhost:8000/health | jq'

# Check stats
watch -n 30 'curl -s http://localhost:8000/api/stats | jq'
```

### Step 6: Gradual Traffic Increase

Follow the [Blue-Green Deployment Strategy](#blue-green-deployment-strategy) timeline.

At each stage:
1. Update Nginx weights
2. Reload Nginx: `sudo nginx -s reload`
3. Monitor for 4-12 hours
4. Verify metrics within acceptable ranges
5. Proceed to next stage or rollback

---

## Monitoring

### Key Metrics to Monitor

#### 1. Response Time

**Target:** < 2s (average)

```bash
# Monitor response time
curl -w "Time: %{time_total}s\n" -o /dev/null -s http://localhost:8000/api/chat \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?"}'
```

#### 2. Error Rate

**Target:** < 0.1%

```bash
# Count 5xx errors in last hour
tail -1000 /var/log/nginx/access.log | \
  grep -c " 5[0-9][0-9] "
```

#### 3. Request Throughput

**Target:** Match or exceed Flask baseline

```bash
# Requests per minute
tail -60 /var/log/nginx/access.log | wc -l
```

#### 4. CPU & Memory Usage

```bash
# Monitor resources
htop
# Or
ps aux | grep -E '(uvicorn|fastapi)'
```

#### 5. Database Connections

```bash
# Check active connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"
```

### Monitoring Dashboard (Optional)

For production, consider:
- **Prometheus + Grafana**: Metrics visualization
- **Sentry**: Error tracking
- **New Relic / Datadog**: APM
- **CloudWatch** (AWS): Log aggregation

### Health Check Endpoint

```bash
# Automated health check (run every 30s)
*/30 * * * * curl -f http://localhost:8000/health || echo "Health check failed"
```

---

## Rollback Procedure

### When to Rollback

Trigger immediate rollback if:
- Error rate > 5%
- Response time > 2x baseline
- Critical functionality broken
- Database connection failures
- Memory leaks detected

### Immediate Rollback (< 5 minutes)

#### Step 1: Redirect Traffic to Flask

```bash
# SSH to server
ssh user@production-server

# Edit Nginx config
sudo nano /etc/nginx/sites-enabled/default

# Set Flask to 100%
upstream backend {
    server localhost:5000 weight=100;  # Flask 100%
    server localhost:8000 weight=0;    # FastAPI 0%
}

# Or remove FastAPI entirely
upstream backend {
    server localhost:5000;
}

# Reload Nginx
sudo nginx -t && sudo nginx -s reload
```

#### Step 2: Stop FastAPI

```bash
# Option A: Systemd
sudo systemctl stop fastapi

# Option B: Kill process
pkill -f uvicorn

# Option C: Supervisor
supervisorctl stop fastapi
```

#### Step 3: Verify Flask is Healthy

```bash
curl http://localhost:5000/api/stats
curl http://localhost:5000/health  # If health endpoint exists
```

#### Step 4: Notify Team

```bash
echo "Rolled back to Flask at $(date)" | \
  mail -s "FastAPI Rollback Executed" team@example.com
```

### Planned Rollback (< 30 minutes)

If issues are non-critical but concerning:

```bash
# 1. Gradually reduce FastAPI traffic
#    (reverse the gradual rollout process)

# 2. Investigate issues
tail -f /var/log/fastapi/error.log

# 3. Fix issues in staging

# 4. Re-deploy when ready
```

---

## Post-Deployment

### After 100% Traffic on FastAPI (48 hours stable)

#### 1. Decommission Flask

```bash
# Stop Flask/Gunicorn
sudo systemctl stop gunicorn
# Or
pkill -f gunicorn

# Remove from Nginx
sudo nano /etc/nginx/sites-enabled/default
# (remove Flask upstream)

# Reload Nginx
sudo nginx -t && sudo nginx -s reload
```

#### 2. Archive Old Code

```bash
# Tag Flask version
git tag flask-legacy-$(date +%Y%m%d)
git push origin --tags

# Optional: Create backup branch
git branch flask-backup
git push origin flask-backup
```

#### 3. Update Documentation

```bash
# Update README.md
# Update API documentation
# Update runbooks
```

#### 4. Performance Review

Document improvements:
- Response time reduction
- Throughput increase
- Error rate change
- Cost savings (if applicable)

#### 5. Celebrate 🎉

Migration complete! Async FastAPI is now in production.

---

## Troubleshooting

### Issue: FastAPI won't start

```bash
# Check logs
tail -f /var/log/fastapi/error.log

# Check port availability
lsof -i :8000

# Check config
python -c "from config import Config; Config.validate_config()"

# Test manually
uvicorn fastapi_app:app --port 8001
```

### Issue: High error rate

```bash
# Check error logs
grep -i error /var/log/fastapi/error.log | tail -100

# Check database connectivity
python -c "from database_async import check_db_connection; import asyncio; asyncio.run(check_db_connection())"

# Check OpenAI API
python -c "from openai import AsyncOpenAI; import asyncio; asyncio.run(AsyncOpenAI().models.list())"
```

### Issue: Slow responses

```bash
# Check resource usage
htop

# Check worker count
ps aux | grep uvicorn | wc -l

# Increase workers
# Edit WEB_CONCURRENCY or restart with more workers
uvicorn fastapi_app:app --workers 8
```

---

## Configuration Files

### Nginx Configuration (`/etc/nginx/sites-enabled/default`)

See `nginx.conf` in repository.

### Systemd Service (`/etc/systemd/system/fastapi.service`)

See `fastapi.service` in repository.

### Environment Variables (`.env`)

See `.env.example` in repository.

---

## Support

For issues during deployment:
- Check logs: `/var/log/fastapi/`
- Check this guide's [Troubleshooting](#troubleshooting) section
- Contact: devops@example.com
- Emergency rollback: Follow [Rollback Procedure](#rollback-procedure)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-18
**Migration:** Flask → FastAPI (Async)
