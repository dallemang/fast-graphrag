#!/bin/bash
# fix_redis.sh - Fix Redis connection issues by applying the optimized URL format

# App name
APP=kgc-ddw-entity

echo "==========================================="
echo "REDIS CONNECTION FIXER"
echo "App: $APP"
echo "Time: $(date)"
echo "==========================================="

# Get the current Redis URL
REDIS_URL=$(heroku config:get REDIS_URL -a $APP)

# Check if it's a Heroku Redis URL
if [[ "$REDIS_URL" == *"compute-1.amazonaws.com"* ]]; then
    echo "Detected Heroku Redis URL"
    
    # 1. Remove rediss:// scheme and replace with redis://
    FIXED_URL=${REDIS_URL//rediss:/redis:}
    
    # 2. Add connection parameters directly to the URL to force proper settings
    if [[ "$FIXED_URL" != *"?connection_timeout"* ]]; then
        FIXED_URL="${FIXED_URL}?connection_timeout=60.0&socket_timeout=60.0&socket_keepalive=true&health_check_interval=30"
        echo "Added connection parameters to URL to prevent reset errors"
    fi
    
    echo -e "\nOriginal URL: $REDIS_URL"
    echo "Fixed URL:    $FIXED_URL"
    
    # Apply the fixed URL
    echo -e "\nApplying optimized Redis URL configuration..."
    heroku config:set REDIS_URL="$FIXED_URL" -a $APP
    
    # Set additional environment variables to fix SSL issues
    echo "Setting additional environment variables to fix SSL issues..."
    heroku config:set PYTHONWARNINGS="ignore:Unverified HTTPS request" -a $APP
    heroku config:set PYTHONHTTPSVERIFY=0 -a $APP
    heroku config:set SSL_CERT_REQS=none -a $APP
    heroku config:set REDIS_SSL_CERT_REQS=none -a $APP
    
    # Set connection parameters via environment
    heroku config:set REDIS_SOCKET_TIMEOUT=60 -a $APP
    heroku config:set REDIS_SOCKET_CONNECT_TIMEOUT=30 -a $APP
    heroku config:set REDIS_RETRY_ON_TIMEOUT=true -a $APP
    heroku config:set REDIS_HEALTH_CHECK_INTERVAL=30 -a $APP
    
    # Restart the app to ensure changes take effect
    echo -e "\nRestarting app with new Redis configuration..."
    heroku restart -a $APP
    
    echo -e "\n✅ Redis connection optimized with improved parameters"
    echo "This should help prevent 'Connection reset by peer' errors"
else
    echo "Not using Heroku Redis or URL is already optimized"
    echo "Current REDIS_URL: $REDIS_URL"
    
    # Check if they want to convert anyway
    read -p "Do you want to add the optimized connection parameters anyway? (y/n): " choice
    
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        # Add connection parameters directly to the URL
        if [[ "$REDIS_URL" != *"?connection_timeout"* ]]; then
            FIXED_URL="${REDIS_URL}?connection_timeout=60.0&socket_timeout=60.0&socket_keepalive=true&health_check_interval=30"
            
            # Apply the fixed URL
            echo -e "\nApplying optimized Redis URL configuration..."
            heroku config:set REDIS_URL="$FIXED_URL" -a $APP
            
            # Set additional environment variables
            echo "Setting additional environment variables to fix SSL issues..."
            heroku config:set PYTHONWARNINGS="ignore:Unverified HTTPS request" -a $APP
            heroku config:set PYTHONHTTPSVERIFY=0 -a $APP
            heroku config:set SSL_CERT_REQS=none -a $APP
            heroku config:set REDIS_SSL_CERT_REQS=none -a $APP
            
            # Restart the app to ensure changes take effect
            echo -e "\nRestarting app with new Redis configuration..."
            heroku restart -a $APP
            
            echo -e "\n✅ Redis connection optimized with improved parameters"
        else
            echo "URL already has connection parameters. No changes needed."
        fi
    else
        echo "No changes made to Redis configuration"
    fi
fi

echo -e "\nTo check if the changes fixed the connection issues, run:"
echo "heroku logs --tail -a $APP"
echo "==========================================="