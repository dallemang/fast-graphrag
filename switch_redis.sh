#!/bin/bash
# switch_redis.sh - Switch between Redis providers to fix connection issues

# App name
APP=kgc-ddw-entity

echo "==========================================="
echo "REDIS PROVIDER SWITCHING UTILITY"
echo "App: $APP"
echo "Time: $(date)"
echo "==========================================="

# Check current Redis configuration
echo -e "\n📊 CURRENT REDIS CONFIGURATION:"
heroku config -a $APP | grep "REDIS"

# Get the current Redis URLs
REDIS_URL=$(heroku config:get REDIS_URL -a $APP)
UPSTASH_REDIS_URL=$(heroku config:get UPSTASH_REDIS_URL -a $APP)

echo -e "\n🔄 Available Redis providers:"
echo "1) Heroku Redis (current primary)"
echo "2) Upstash Redis (available alternative)"

# Ask which Redis to use
read -p "Which Redis provider do you want to use as primary? (1/2): " choice

if [[ "$choice" == "1" ]]; then
    echo -e "\n✅ Using Heroku Redis as primary"
    echo "Setting REDIS_URL to Heroku Redis..."
    heroku config:set REDIS_URL="$REDIS_URL" -a $APP
elif [[ "$choice" == "2" ]]; then
    if [[ -z "$UPSTASH_REDIS_URL" ]]; then
        echo -e "\n❌ Error: UPSTASH_REDIS_URL is not configured"
        echo "Please set up Upstash Redis first"
        exit 1
    fi
    
    echo -e "\n✅ Using Upstash Redis as primary"
    echo "Setting REDIS_URL to Upstash Redis..."
    heroku config:set REDIS_URL="$UPSTASH_REDIS_URL" -a $APP
else
    echo -e "\n❌ Invalid choice. No changes made."
    exit 1
fi

# Restart the app to ensure changes take effect
echo -e "\nRestarting app with new Redis configuration..."
heroku restart -a $APP

echo -e "\n📊 NEW REDIS CONFIGURATION:"
heroku config -a $APP | grep "REDIS"

echo -e "\nRedis switch completed. Check the app logs to verify connections."
echo "Run this command to view logs: heroku logs --tail -a $APP"
echo "==========================================="