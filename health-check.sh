#!/bin/bash

# Health check script for Azure deployment
# This script tests if the deployed container is responding correctly

set -e

CONTAINER_APP_URL=$1
TIMEOUT=60
RETRY_INTERVAL=5
RETRIES=$((TIMEOUT / RETRY_INTERVAL))

if [ -z "$CONTAINER_APP_URL" ]; then
    echo "Error: Container App URL not provided"
    echo "Usage: $0 <container_app_url>"
    exit 1
fi

echo "========================================" 
echo "Container Health Check"
echo "========================================"
echo "URL: $CONTAINER_APP_URL"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    
    echo "Testing $method $endpoint..."
    
    response=$(curl -s -w "\n%{http_code}" -X $method "$CONTAINER_APP_URL$endpoint" \
        -H "Content-Type: application/json" \
        --connect-timeout 5 \
        --max-time 10)
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    echo "  HTTP Status: $http_code"
    
    if [ "$http_code" -lt 400 ] || [ "$http_code" = "404" ] || [ "$http_code" = "405" ]; then
        echo "  ✓ Endpoint is accessible"
        return 0
    else
        echo "  ✗ Endpoint returned error: $http_code"
        return 1
    fi
}

# Wait for container to be ready
echo "Waiting for container to be ready..."
attempt=1
while [ $attempt -le $RETRIES ]; do
    echo "  Attempt $attempt/$RETRIES..."
    
    if test_endpoint "/" 2>/dev/null; then
        echo ""
        echo "✓ Container is ready!"
        echo ""
        
        # Run additional health checks
        echo "Running health checks..."
        test_endpoint "/predict" "POST" || true
        
        echo ""
        echo "✓ Health check passed!"
        exit 0
    fi
    
    if [ $attempt -lt $RETRIES ]; then
        echo "  Container not ready yet. Retrying in ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    fi
    
    attempt=$((attempt + 1))
done

echo ""
echo "✗ Health check failed! Container did not respond within ${TIMEOUT}s"
exit 1
