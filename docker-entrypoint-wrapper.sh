#!/bin/bash
# Wrapper for docker-entrypoint.sh that sets SSL environment variables

# Disable SSL certificate verification for Redis
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
export REDIS_SSL_CERT_REQS=none
export PYTHONHTTPSVERIFY=0
export REDISPY_DISABLE_SSL_VERIFICATION=1
export REDIS_VERIFY_SSL=false

# Force Redis default timeout to be higher
export REDIS_SOCKET_CONNECT_TIMEOUT=10
export REDIS_SOCKET_TIMEOUT=30

# Set SSL-specific parameters for Redis
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# If it's a rediss:// URL, convert it to redis:// (this bypasses SSL entirely)
if [[ -n "$REDIS_URL" && "$REDIS_URL" == rediss://* ]]; then
    export REDIS_URL=$(echo "$REDIS_URL" | sed 's/^rediss:/redis:/')
    echo "Converted REDIS_URL from rediss:// to redis://"
fi

# Pass all arguments to the original entrypoint
exec /app/docker-entrypoint.sh "$@"