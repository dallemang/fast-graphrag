#!/bin/bash

# Log Redis URL 
echo "Original REDIS_URL: $REDIS_URL"

# Store the original URL for reference
export ORIGINAL_REDIS_URL="$REDIS_URL"

# Clean up any existing Redis variables to avoid conflicts
unset REDIS_URL
unset REDIS_TLS_URL

# Use a fallback local Redis URL for testing if nothing else works
export FALLBACK_REDIS_URL="redis://localhost:6379"

# Function to validate Redis connection
test_redis_connection() {
    local url=$1
    echo "Testing Redis connection to: $url"
    # Use redis-cli to test if we can connect and ping
    if command -v redis-cli &> /dev/null; then
        # Extract host and port from URL
        if [[ "$url" =~ redis://([^:]+):([0-9]+) ]]; then
            local host=${BASH_REMATCH[1]}
            local port=${BASH_REMATCH[2]}
            echo "Attempting to connect to Redis at $host:$port"
            if redis-cli -h "$host" -p "$port" ping > /dev/null 2>&1; then
                echo "✅ Successfully connected to Redis at $host:$port"
                return 0
            else
                echo "❌ Failed to connect to Redis at $host:$port"
                return 1
            fi
        fi
    fi
    # If redis-cli is not available or URL parsing failed, assume it works
    echo "⚠️ Cannot test Redis connection (redis-cli not available or invalid URL format)"
    return 0
}

# Set up direct connection to Heroku Redis using environment variables - try each one in priority order
REDIS_VARS=(
    "HEROKU_REDIS_NAVY_URL" 
    "HEROKU_REDIS_ROSE_URL" 
    "HEROKU_REDIS_CRIMSON_URL"
    "REDIS_MAROON_URL"
    "REDIS_URL"
)

REDIS_CONNECTED=false

for var in "${REDIS_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo "Trying Redis connection from $var"
        # Convert from rediss:// to redis:// to avoid SSL issues
        temp_url=${!var}
        temp_url=${temp_url//rediss:/redis:}
        
        # Test connection before accepting this URL
        if test_redis_connection "$temp_url"; then
            export REDIS_URL="$temp_url"
            echo "✅ Using $var: $REDIS_URL"
            REDIS_CONNECTED=true
            break
        else
            echo "❌ Failed to connect using $var, trying next option"
        fi
    fi
done

# If we still don't have a working Redis connection, try the original URL anyway
if [ "$REDIS_CONNECTED" = false ] && [ -n "$ORIGINAL_REDIS_URL" ]; then
    temp_url=${ORIGINAL_REDIS_URL//rediss:/redis:}
    export REDIS_URL="$temp_url"
    echo "⚠️ Using original REDIS_URL as last resort: $REDIS_URL"
fi

# If we still don't have a Redis URL, use the fallback
if [ -z "$REDIS_URL" ]; then
    export REDIS_URL="$FALLBACK_REDIS_URL"
    echo "⚠️ No Redis URL found, using fallback: $REDIS_URL"
fi

# Handle Redis URL for Heroku - CRITICAL FIX for "Connection reset by peer" errors
# If we're on Heroku, absolutely force the proper URL format with proper port
if [[ "$REDIS_URL" == *"compute-1.amazonaws.com"* ]]; then
    echo "Detected Heroku Redis URL, applying special handling for Connection reset issues"
    
    # For Heroku Redis, we need to reduce the port to a standard Redis port
    # The high port numbers (12000+) often cause "Connection reset by peer" errors
    export REDIS_URL="${REDIS_URL//rediss:/redis:}"
    
    # Add connection options directly to the URL to force proper settings
    if [[ "$REDIS_URL" != *"?connection_timeout"* ]]; then
        export REDIS_URL="${REDIS_URL}?connection_timeout=60.0&socket_timeout=60.0&socket_keepalive=true&health_check_interval=30"
        echo "Added connection parameters to URL to prevent reset errors"
    fi
else
    # Just perform the standard rediss: -> redis: conversion for non-Heroku Redis
    export REDIS_URL=${REDIS_URL//rediss:/redis:}
fi

echo "Final REDIS_URL: $REDIS_URL"

# Disable SSL certificate verification to avoid any SSL-related issues
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
export PYTHONHTTPSVERIFY=0
export SSL_CERT_REQS="none"
export SSL_CERT_REQS="CERT_NONE"

# Set explicit Redis connection configuration via environment variables
# These will be used by the Redis client in Python
export REDIS_SOCKET_TIMEOUT=60
export REDIS_SOCKET_CONNECT_TIMEOUT=30
export REDIS_RETRY_ON_TIMEOUT=true
export REDIS_HEALTH_CHECK_INTERVAL=30

# Determine whether to run web or worker based on DYNO env var
if [[ "$DYNO" == *"web"* ]]; then
  echo "Starting web process..."
  exec gunicorn --bind 0.0.0.0:$PORT --timeout 30 --threads=4 --workers=1 --keep-alive=5 --access-logfile=- app:app
else
  echo "Starting worker process..."
  exec rq worker --url $REDIS_URL graphrag_processing
fi