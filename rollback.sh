#!/bin/bash
# Emergency Rollback Script
#
# Immediately reverts from FastAPI back to Flask
# Use this script when:
# - Error rate > 5%
# - Response time > 2x baseline
# - Critical functionality broken
#
# Usage:
#   ./rollback.sh [reason]
#
# Example:
#   ./rollback.sh "High error rate detected"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
NGINX_CONFIG="/etc/nginx/sites-enabled/default"
NGINX_BACKUP="/etc/nginx/sites-enabled/default.backup"
ROLLBACK_REASON="${1:-Manual rollback triggered}"
LOG_FILE="/var/log/rollback-$(date +%Y%m%d-%H%M%S).log"

echo -e "${RED}=================================="
echo "EMERGENCY ROLLBACK TO FLASK"
echo "=================================="
echo -e "${NC}"
echo "Reason: $ROLLBACK_REASON"
echo "Time: $(date)"
echo "Log: $LOG_FILE"
echo ""
read -p "Continue with rollback? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Log function
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1" | tee -a "$LOG_FILE"
}

log "==================== ROLLBACK START ===================="
log "Reason: $ROLLBACK_REASON"

# Step 1: Redirect traffic to Flask
log "STEP 1: Redirecting traffic to Flask..."

# Backup current Nginx config
if [ -f "$NGINX_CONFIG" ]; then
    sudo cp "$NGINX_CONFIG" "${NGINX_CONFIG}.pre-rollback"
    log "  - Backed up current Nginx config"
fi

# Restore Flask-only config or create new one
if [ -f "$NGINX_BACKUP" ]; then
    log "  - Restoring backup Nginx config (Flask only)"
    sudo cp "$NGINX_BACKUP" "$NGINX_CONFIG"
else
    log "  - Creating new Flask-only config"
    sudo bash -c "cat > $NGINX_CONFIG" <<'EOF'
upstream backend {
    server localhost:5000;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
fi

# Test Nginx config
log "  - Testing Nginx configuration..."
if sudo nginx -t 2>&1 | tee -a "$LOG_FILE"; then
    log "  - Nginx config valid"
else
    log "  - ERROR: Nginx config invalid! Manual intervention required."
    exit 1
fi

# Reload Nginx
log "  - Reloading Nginx..."
if sudo nginx -s reload 2>&1 | tee -a "$LOG_FILE"; then
    log "  - Nginx reloaded successfully"
else
    log "  - ERROR: Nginx reload failed! Manual intervention required."
    exit 1
fi

echo -e "${GREEN}✓ Traffic redirected to Flask${NC}"

# Step 2: Stop FastAPI
log "STEP 2: Stopping FastAPI..."

# Try systemd first
if systemctl is-active --quiet fastapi 2>/dev/null; then
    log "  - Stopping FastAPI via systemd..."
    sudo systemctl stop fastapi
    log "  - FastAPI stopped (systemd)"
# Try supervisor
elif supervisorctl status fastapi &>/dev/null; then
    log "  - Stopping FastAPI via supervisor..."
    supervisorctl stop fastapi
    log "  - FastAPI stopped (supervisor)"
# Kill uvicorn processes
elif pgrep -f uvicorn &>/dev/null; then
    log "  - Killing uvicorn processes..."
    pkill -f uvicorn
    sleep 2
    log "  - uvicorn processes killed"
else
    log "  - No FastAPI processes found (already stopped?)"
fi

echo -e "${GREEN}✓ FastAPI stopped${NC}"

# Step 3: Verify Flask is healthy
log "STEP 3: Verifying Flask is healthy..."

max_attempts=5
attempt=0
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    log "  - Health check attempt $attempt/$max_attempts..."

    if curl -s -f http://localhost:5000/api/stats &>/dev/null; then
        log "  - Flask is responding"
        break
    elif [ $attempt -eq $max_attempts ]; then
        log "  - ERROR: Flask not responding after $max_attempts attempts!"
        log "  - CRITICAL: Manual intervention required!"
        exit 1
    else
        log "  - Flask not responding, waiting 5s..."
        sleep 5
    fi
done

echo -e "${GREEN}✓ Flask is healthy${NC}"

# Step 4: Notify team (if notification command available)
log "STEP 4: Sending notifications..."

if command -v mail &> /dev/null; then
    echo "Rollback to Flask completed at $(date)" | \
        mail -s "ROLLBACK: FastAPI → Flask" team@example.com
    log "  - Email notification sent"
fi

# Slack notification (if webhook configured)
if [ -n "$SLACK_WEBHOOK" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🔴 ROLLBACK: FastAPI → Flask\\nReason: $ROLLBACK_REASON\"}" \
        "$SLACK_WEBHOOK" &>/dev/null
    log "  - Slack notification sent"
fi

echo -e "${GREEN}✓ Notifications sent${NC}"

# Summary
log "==================== ROLLBACK COMPLETE ===================="
log "Status: SUCCESS"
log "Flask URL: http://localhost:5000"
log "Time taken: $SECONDS seconds"

echo ""
echo -e "${GREEN}=================================="
echo "ROLLBACK COMPLETE"
echo "=================================="
echo -e "${NC}"
echo "Flask is now serving 100% of traffic"
echo ""
echo "Next steps:"
echo "  1. Verify application is working: curl http://localhost:5000/api/stats"
echo "  2. Review logs: tail -f /var/log/nginx/error.log"
echo "  3. Investigate issue that caused rollback"
echo "  4. Fix issue in staging environment"
echo "  5. Re-test before attempting deployment again"
echo ""
echo "Rollback log: $LOG_FILE"
echo ""
