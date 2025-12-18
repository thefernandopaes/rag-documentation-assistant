#!/bin/bash
# Deployment Monitoring Script
#
# Monitors key metrics during FastAPI deployment:
# - Response times
# - Error rates
# - Throughput
# - Resource usage
#
# Usage:
#   ./monitor_deployment.sh [duration_seconds]
#
# Example:
#   ./monitor_deployment.sh 3600  # Monitor for 1 hour

set -e

# Configuration
FASTAPI_URL="${FASTAPI_URL:-http://localhost:8000}"
FLASK_URL="${FLASK_URL:-http://localhost:5000}"
DURATION="${1:-3600}"  # Default: 1 hour
INTERVAL=30  # Check every 30 seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Thresholds
MAX_RESPONSE_TIME=2.0  # 2 seconds
MAX_ERROR_RATE=0.1     # 0.1%
MIN_SUCCESS_RATE=99.5  # 99.5%

echo "=================================="
echo "FastAPI Deployment Monitoring"
echo "=================================="
echo "FastAPI URL: $FASTAPI_URL"
echo "Flask URL: $FLASK_URL"
echo "Duration: ${DURATION}s ($(($DURATION / 60)) minutes)"
echo "Check interval: ${INTERVAL}s"
echo "=================================="
echo ""

# Initialize counters
total_checks=0
failed_checks=0
start_time=$(date +%s)
end_time=$((start_time + DURATION))

# Monitoring loop
while [ $(date +%s) -lt $end_time ]; do
    total_checks=$((total_checks + 1))
    current_time=$(date +"%Y-%m-%d %H:%M:%S")

    echo "[$current_time] Check #$total_checks"
    echo "-----------------------------------"

    # 1. Health Check
    echo -n "Health check... "
    if health_response=$(curl -s -f "$FASTAPI_URL/health" 2>&1); then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        failed_checks=$((failed_checks + 1))
    fi

    # 2. Response Time Test
    echo -n "Response time test... "
    response_time=$(curl -o /dev/null -s -w "%{time_total}" \
        -X POST "$FASTAPI_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d '{"query":"What is FastAPI?"}' 2>&1)

    if (( $(echo "$response_time < $MAX_RESPONSE_TIME" | bc -l) )); then
        echo -e "${GREEN}${response_time}s${NC} (target: < ${MAX_RESPONSE_TIME}s)"
    else
        echo -e "${RED}${response_time}s${NC} (SLOW! target: < ${MAX_RESPONSE_TIME}s)"
        failed_checks=$((failed_checks + 1))
    fi

    # 3. Stats Endpoint
    echo -n "Stats endpoint... "
    if stats_response=$(curl -s -f "$FASTAPI_URL/api/stats" 2>&1); then
        echo -e "${GREEN}OK${NC}"

        # Parse stats (if jq available)
        if command -v jq &> /dev/null; then
            doc_count=$(echo "$stats_response" | jq -r '.documents.document_count')
            conv_count=$(echo "$stats_response" | jq -r '.conversations.total')
            echo "  - Documents: $doc_count"
            echo "  - Conversations: $conv_count"
        fi
    else
        echo -e "${RED}FAILED${NC}"
        failed_checks=$((failed_checks + 1))
    fi

    # 4. Resource Usage
    echo "Resource usage:"
    if command -v ps &> /dev/null; then
        # CPU and Memory for uvicorn processes
        ps aux | grep -E 'uvicorn|fastapi' | grep -v grep | \
            awk '{printf "  - PID %s: CPU %.1f%% MEM %.1f%%\n", $2, $3, $4}'
    fi

    # 5. Error Check (if log file exists)
    if [ -f "/var/log/fastapi/error.log" ]; then
        error_count=$(tail -100 /var/log/fastapi/error.log | grep -ci error || true)
        if [ $error_count -gt 0 ]; then
            echo -e "  - Recent errors: ${YELLOW}${error_count}${NC}"
        fi
    fi

    # Summary
    success_rate=$(echo "scale=2; 100 * ($total_checks - $failed_checks) / $total_checks" | bc)
    echo "-----------------------------------"
    echo "Summary: $total_checks checks, $failed_checks failures ($success_rate% success)"

    if (( $(echo "$success_rate < $MIN_SUCCESS_RATE" | bc -l) )); then
        echo -e "${RED}WARNING: Success rate below threshold!${NC}"
        echo "Consider rollback if issues persist."
    fi

    echo ""

    # Wait before next check
    sleep $INTERVAL
done

# Final summary
echo "=================================="
echo "Monitoring Complete"
echo "=================================="
echo "Total checks: $total_checks"
echo "Failed checks: $failed_checks"
echo "Success rate: $success_rate%"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
elif (( $(echo "$success_rate < $MIN_SUCCESS_RATE" | bc -l) )); then
    echo -e "${RED}CRITICAL: Success rate below $MIN_SUCCESS_RATE%${NC}"
    echo "Recommend ROLLBACK to Flask."
    exit 2
else
    echo -e "${YELLOW}Some issues detected. Review logs.${NC}"
    exit 1
fi
