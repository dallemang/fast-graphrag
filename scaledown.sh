#!/bin/bash
  # scale-down.sh - Run this immediately after the workshop

  # App name
  APP=kgc-ddw-entity

  echo "Scaling down resources after workshop..."

  # Downgrade dynos
  echo "Downgrading dynos to hobby..."
  heroku ps:type web=hobby -a $APP
  heroku ps:type worker=hobby -a $APP

  # Scale down workers
  echo "Scaling down to 1 worker dyno..."
  heroku ps:scale worker=1 -a $APP

  # Downgrade Redis
  echo "Checking Redis..."
  PREMIUM_REDIS=$(heroku addons -a $APP | grep "heroku-redis.*premium")

  if [[ ! -z "$PREMIUM_REDIS" ]]; then
    echo "Downgrading Redis to hobby-dev..."
    heroku addons:downgrade heroku-redis:hobby-dev -a $APP
  fi

  echo "Scale-down complete! Resources returned to normal levels."
  echo "Current dyno formation:"
  heroku ps -a $APP
  echo "Current addons:"
  heroku addons -a $APP
