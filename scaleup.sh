#!/bin/bash
  # scale-up.sh - Run this 30 minutes before the workshop

  # App name
  APP=kgc-ddw-entity

  echo "Scaling up resources for workshop..."

  # Upgrade dynos
  echo "Upgrading dynos to Standard-2X..."
  heroku ps:type web=standard-2x -a $APP
  heroku ps:type worker=standard-2x -a $APP

  # Scale workers
  echo "Scaling to 2 worker dynos..."
  heroku ps:scale worker=2 -a $APP

  # Add premium Redis (preserve the current Redis first if needed)
  echo "Checking current Redis..."
  CURRENT_REDIS=$(heroku addons -a $APP | grep redis)

  if [[ $CURRENT_REDIS == *"heroku-redis"* ]]; then
    echo "Upgrading existing Heroku Redis to Premium-0..."
    heroku addons:upgrade heroku-redis:premium-0 -a $APP
  else
    # If using Upstash, add Heroku Redis alongside it
    echo "Adding Heroku Redis Premium-0..."
    heroku addons:create heroku-redis:premium-0 -a $APP

    # Update the app to use the new Redis
    echo "Restarting app to use new Redis..."
    heroku restart -a $APP
  fi

  echo "Scale-up complete! App is ready for workshop load."
  echo "Current dyno formation:"
  heroku ps -a $APP
  echo "Current addons:"
  heroku addons -a $APP
