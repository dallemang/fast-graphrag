#!/bin/bash
# Script to set environment variables to fix Redis SSL issues

# App name
APP=kgc-ddw-entity

echo "Setting environment variables to fix Redis SSL issues..."

# Disable SSL verification for Redis python client
heroku config:set PYTHONWARNINGS="ignore:Unverified HTTPS request" -a $APP
heroku config:set REDIS_SSL_CERT_REQS=none -a $APP

echo "Restarting the app to apply changes..."
heroku restart -a $APP

echo "Done! The Redis SSL issues should be fixed now."