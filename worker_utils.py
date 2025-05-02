#!/usr/bin/env python3
"""
Utility functions for worker progress reporting.
"""

import os
import logging
import ssl
import redis
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
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
            connection = redis.Redis.from_url(
                upstash_url,
                socket_timeout=60,
                socket_connect_timeout=30,
                ssl_cert_reqs=None,
                ssl_check_hostname=False,
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
        connection = redis.Redis(
            host=host,
            port=port,
            password=password,
            ssl=True,
            ssl_cert_reqs=None,
            ssl_check_hostname=False,
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

class ProgressReporter:
    """
    Class to handle progress reporting for worker tasks.
    """
    
    def __init__(self, job_id):
        """Initialize with job ID and connect to Redis."""
        self.job_id = job_id
        try:
            self.redis_conn = get_redis_connection()
            self.enabled = True
        except Exception as e:
            logger.error(f"Could not initialize progress reporter: {e}")
            self.enabled = False
    
    def update_progress(self, percent, stage, message):
        """
        Update job progress in Redis.
        
        Args:
            percent (int): Progress percentage (0-100)
            stage (str): Current processing stage
            message (str): Detailed message about current progress
        """
        if not self.enabled:
            logger.warning(f"Progress update skipped for job {self.job_id}: reporter disabled")
            return False
        
        try:
            # Update progress information
            progress_data = {
                'percent': percent,
                'stage': stage,
                'message': message,
                'updated_at': datetime.datetime.now().isoformat()
            }
            
            # Store progress as a separate object
            self.redis_conn.hmset(f"job_progress:{self.job_id}", progress_data)
            self.redis_conn.expire(f"job_progress:{self.job_id}", 86400)  # 24-hour TTL
            
            # Also store summary in metadata key
            update_data = {
                'progress_percent': percent,
                'progress_stage': stage,
                'progress_message': message,
                'updated_at': datetime.datetime.now().isoformat()
            }
            
            # Get existing metadata
            metadata = self.redis_conn.hgetall(f"job_metadata:{self.job_id}")
            if metadata:
                # Keep status but update progress info
                status = metadata.get(b'status', b'in_progress').decode('utf-8') if isinstance(metadata.get(b'status'), bytes) else metadata.get('status', 'in_progress')
                
                # Update metadata with new progress info
                metadata_update = {
                    'status': status,
                }
                metadata_update.update(update_data)
                
                # Convert all values to strings
                metadata_update = {k: str(v) if v is not None else "" for k, v in metadata_update.items()}
                
                # Store in Redis
                self.redis_conn.hmset(f"job_metadata:{self.job_id}", metadata_update)
                self.redis_conn.expire(f"job_metadata:{self.job_id}", 86400)  # 24-hour TTL
            
            logger.info(f"Updated progress for job {self.job_id}: {percent}% - {stage}")
            return True
        except Exception as e:
            logger.error(f"Error updating job progress: {e}")
            return False
            
    def start_stage(self, stage, message, base_percent=0):
        """
        Start a new processing stage.
        
        Args:
            stage (str): Name of the stage
            message (str): Description of the stage
            base_percent (int): Base percentage for this stage
        """
        return self.update_progress(base_percent, stage, f"Starting: {message}")
        
    def complete_stage(self, stage, message, percent=100):
        """Mark a stage as complete."""
        return self.update_progress(percent, stage, f"Completed: {message}")
        
    def update_stage(self, stage, percent, message):
        """Update progress within a stage."""
        return self.update_progress(percent, stage, message)
        
    def get_job_progress(self, job_id):
        """Get job progress from Redis."""
        if not self.enabled:
            logger.warning(f"Cannot get job progress: reporter disabled")
            return {"stage": "unknown", "percent": 0, "message": "Progress tracking unavailable"}
        
        try:
            progress = self.redis_conn.hgetall(f"job_progress:{job_id}")
            if not progress:
                # If no specific progress, check metadata for basic status
                metadata = self.redis_conn.hgetall(f"job_metadata:{job_id}")
                if metadata:
                    # Basic progress info from metadata
                    return {
                        'percent': metadata.get(b'progress_percent', b'0').decode('utf-8') 
                                  if isinstance(metadata.get(b'progress_percent'), bytes) else metadata.get('progress_percent', '0'),
                        'stage': metadata.get(b'progress_stage', b'unknown').decode('utf-8')
                                if isinstance(metadata.get(b'progress_stage'), bytes) else metadata.get('progress_stage', 'unknown'),
                        'message': metadata.get(b'progress_message', b'No progress details available').decode('utf-8')
                                  if isinstance(metadata.get(b'progress_message'), bytes) else metadata.get('progress_message', 'No progress details available'),
                        'updated_at': metadata.get(b'updated_at', b'').decode('utf-8')
                                    if isinstance(metadata.get(b'updated_at'), bytes) else metadata.get('updated_at', '')
                    }
                return {"stage": "unknown", "percent": 0, "message": "No progress data available"}
                
            # Convert bytes to strings if needed
            return {k.decode('utf-8') if isinstance(k, bytes) else k: 
                    v.decode('utf-8') if isinstance(v, bytes) else v 
                    for k, v in progress.items()}
        except Exception as e:
            logger.error(f"Error retrieving job progress: {e}")
            return {"stage": "error", "percent": 0, "message": f"Error retrieving progress: {str(e)}"}