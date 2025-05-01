#!/bin/bash
# scaleup.sh - Run this 30 minutes before the workshop

# App name
APP=kgc-ddw-entity

echo "Scaling up resources for workshop..."

# Upgrade all dynos to Standard-2X simultaneously
echo "Upgrading all dynos to Standard-2X..."
heroku ps:type standard-2x -a $APP

# Scale workers after upgrading
echo "Scaling to 2 worker dynos..."
heroku ps:scale worker=2 -a $APP

# Check Redis status
echo "Checking current Redis..."
REDIS_INFO=$(heroku addons -a $APP | grep -i "redis")

# If Premium Redis already exists, don't create a new one
if [[ "$REDIS_INFO" == *"premium"* ]]; then
  echo "Premium Redis already exists, no need to create another"
else
  # If we have heroku-redis, but not premium
  if [[ "$REDIS_INFO" == *"heroku-redis"* && "$REDIS_INFO" != *"premium"* ]]; then
    echo "Upgrading existing Heroku Redis to Premium-0..."
    heroku addons:upgrade heroku-redis:premium-0 -a $APP
  elif [[ "$REDIS_INFO" != *"heroku-redis"* ]]; then
    # No heroku-redis found, create new
    echo "Adding Heroku Redis Premium-0..."
    heroku addons:create heroku-redis:premium-0 -a $APP
  fi
  
  # Restart app to apply Redis changes
  echo "Restarting app to use new Redis configuration..."
  heroku restart -a $APP
fi

echo "Scale-up complete! App is ready for workshop load."
echo "Current dyno formation:"
heroku ps -a $APP
echo "Current addons:"
heroku addons -a $APP

# Run resource check to confirm
echo -e "\nRunning resource check..."
./check-resources.sh
