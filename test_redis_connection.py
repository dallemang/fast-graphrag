#!/usr/bin/env python3
"""
Test script to verify Redis connection.
"""

import os
import sys
import logging
import ssl
from redis import Redis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_redis_connection():
    """
    Establish a Redis connection using the same methodology as app_with_redis.py.
    """
    # First, try to use UPSTASH_REDIS_URL
    upstash_url = os.environ.get('UPSTASH_REDIS_URL')
    if upstash_url:
        logger.info(f"Using Upstash Redis URL: {upstash_url}")
        try:
            # Create a custom SSL context that doesn't verify certificates
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.options |= ssl.OP_NO_SSLv2
            ssl_context.options |= ssl.OP_NO_SSLv3
            
            connection = Redis.from_url(
                upstash_url,
                socket_timeout=60,
                socket_connect_timeout=30,
                ssl_cert_reqs="none",
                decode_responses=False  # Important for binary data
            )
            connection.ping()  # Test the connection
            logger.info("Successfully connected to Upstash Redis using URL")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect using Upstash Redis URL: {e}")
    
    # Fall back to direct connection parameters
    host = os.environ.get('REDIS_HOST', 'verified-stag-21591.upstash.io')
    port = int(os.environ.get('REDIS_PORT', '6379'))
    password = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '')
    
    logger.info(f"Falling back to direct Upstash connection at {host}:{port}")
    try:
        # Create connection with SSL settings
        connection = Redis(
            host=host,
            port=port,
            password=password,
            ssl=True,
            ssl_cert_reqs="none",
            socket_timeout=60,
            socket_connect_timeout=30,
            decode_responses=False  # Important for binary data
        )
        
        # Test the connection
        connection.ping()
        logger.info("Successfully connected to Upstash Redis using direct parameters")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to Upstash Redis: {e}")
        raise

def test_basic_operations():
    """Test basic Redis operations"""
    try:
        # Get a Redis connection
        redis_conn = get_redis_connection()
        
        # Test key-value operations
        test_key = "test_connection_key"
        test_value = "test_value"
        
        # Set a key
        redis_conn.set(test_key, test_value)
        logger.info(f"Successfully set key: {test_key}")
        
        # Get the key
        retrieved_value = redis_conn.get(test_key).decode('utf-8') if redis_conn.get(test_key) else None
        logger.info(f"Retrieved value: {retrieved_value}")
        
        # Check if the retrieved value matches
        if retrieved_value == test_value:
            logger.info("✅ Basic key-value operations successful")
        else:
            logger.error(f"❌ Value mismatch: expected {test_value}, got {retrieved_value}")
            
        # Delete the test key
        redis_conn.delete(test_key)
        logger.info(f"Deleted test key: {test_key}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to perform basic Redis operations: {e}", exc_info=True)
        return False

def main():
    """Main test function"""
    logger.info("Starting Redis connection test")
    
    try:
        # Test basic operations
        if test_basic_operations():
            logger.info("All tests passed! Redis connection is working properly.")
            return 0
        else:
            logger.error("Redis connection test failed")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())