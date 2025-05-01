#!/bin/bash
# scaledown.sh - Run this immediately after the workshop

# App name
APP=kgc-ddw-entity

echo "Scaling down resources after workshop..."

# First, scale down worker count to avoid issues
echo "Scaling down to 1 worker dyno..."
heroku ps:scale worker=1 -a $APP

# Downgrade all dynos to Basic
echo "Downgrading dynos to Basic tier..."
heroku ps:type basic -a $APP

# Downgrade Redis
echo "Checking for premium Redis..."
PREMIUM_REDIS=$(heroku addons -a $APP | grep -i "heroku-redis.*premium")

if [[ ! -z "$PREMIUM_REDIS" ]]; then
  echo "Downgrading Redis to mini plan..."
  heroku addons:downgrade heroku-redis:mini -a $APP
fi

# Restart the app to ensure changes take effect
echo "Restarting app with new configuration..."
heroku restart -a $APP

echo "Scale-down complete! Resources returned to normal levels."
echo "Current dyno formation:"
heroku ps -a $APP
echo "Current addons:"
heroku addons -a $APP

# Run resource check to confirm
echo -e "\nRunning resource check to confirm no expensive resources remain..."
./check-resources.sh
