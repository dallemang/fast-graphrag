#!/bin/bash
# Deployment script for both web and worker dynos to Heroku with container stack

APP_NAME=${1:-kgc-ddw-entity}

echo "Deploying to Heroku app: $APP_NAME using the container stack"

# Build and push the container for both web and worker
echo "Building and pushing container..."
heroku container:push web  -a $APP_NAME
heroku container:push worker -a $APP_NAME

# Release the container
echo "Releasing container..."
heroku container:release web worker -a $APP_NAME

echo "Deployment complete!"
echo "Check logs with: heroku logs --tail -a $APP_NAME"

# Make a sound notification when complete
say "Deployment complete!"
afplay /System/Library/Sounds/Glass.aiff