import os
import logging
import base64
import uuid
import datetime
import traceback
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect, url_for
from redis import Redis
from rq import Queue
from rq.job import Job
import json
import ssl

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Redis connection setup
def get_redis_connection():
    # First, try to use UPSTASH_REDIS_URL
    upstash_url = os.environ.get('UPSTASH_REDIS_URL')
    if upstash_url:
        logger.info(f"Using Upstash Redis URL: {upstash_url}")
        try:
            connection = Redis.from_url(
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
        connection = Redis(
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

try:
    # Create a Redis connection
    redis_conn = get_redis_connection()

    # Create a queue instance - ensure this matches the worker queue name
    queue_name = 'graphrag_processing'
    job_queue = Queue(queue_name, connection=redis_conn)
    logger.info(f"Initialized job queue: {queue_name}")
except Exception as e:
    logger.error(f"Failed to initialize Redis: {e}")
    # Initialize empty objects so the app can at least start
    redis_conn = None
    job_queue = None

# Functions for job metadata management using Redis
def store_job_metadata(job_id, status, created_at, version, extra_data=None):
    """Store job metadata in Redis"""
    if redis_conn is None:
        logger.error("Cannot store job metadata: Redis connection not available")
        return False
    
    try:
        # Basic metadata that all jobs should have
        job_data = {
            "status": status,
            "created_at": created_at,
            "version": version,
            "job_id": job_id
        }
        
        # Add any extra data provided
        if extra_data and isinstance(extra_data, dict):
            job_data.update(extra_data)
        
        # Convert all values to strings for Redis
        job_data = {k: str(v) if v is not None else "" for k, v in job_data.items()}
        
        # Store in Redis with 24-hour expiration
        redis_conn.hmset(f"job_metadata:{job_id}", job_data)
        redis_conn.expire(f"job_metadata:{job_id}", 86400)  # 24-hour TTL
        logger.info(f"Stored metadata for job {job_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing job metadata: {e}")
        return False

def get_job_metadata(job_id):
    """Retrieve job metadata from Redis"""
    if redis_conn is None:
        logger.error("Cannot get job metadata: Redis connection not available")
        return None
    
    try:
        metadata = redis_conn.hgetall(f"job_metadata:{job_id}")
        if not metadata:
            return None
            
        # Convert bytes to strings if needed
        return {k.decode('utf-8') if isinstance(k, bytes) else k: 
                v.decode('utf-8') if isinstance(v, bytes) else v 
                for k, v in metadata.items()}
    except Exception as e:
        logger.error(f"Error retrieving job metadata: {e}")
        return None

def update_job_status(job_id, status, additional_data=None):
    """Update job status in Redis"""
    if redis_conn is None:
        logger.error("Cannot update job status: Redis connection not available")
        return False
    
    try:
        # Get existing metadata
        metadata = get_job_metadata(job_id)
        if not metadata:
            logger.warning(f"Cannot update status for unknown job: {job_id}")
            return False
            
        # Update status
        metadata['status'] = status
        metadata['updated_at'] = datetime.datetime.now().isoformat()
        
        # Add any additional data
        if additional_data and isinstance(additional_data, dict):
            metadata.update(additional_data)
            
        # Store back to Redis
        store_job_metadata(job_id, status, metadata.get('created_at'), 
                           metadata.get('version'), metadata)
        logger.info(f"Updated status for job {job_id} to {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating job status: {e}")
        return False

def update_job_progress(job_id, percent, stage, message):
    """Update job progress in Redis"""
    if redis_conn is None:
        logger.error("Cannot update job progress: Redis connection not available")
        return False
    
    try:
        # Get existing metadata
        metadata = get_job_metadata(job_id)
        if not metadata:
            logger.warning(f"Cannot update progress for unknown job: {job_id}")
            return False
        
        # Update progress information
        progress_data = {
            'percent': percent,
            'stage': stage,
            'message': message,
            'updated_at': datetime.datetime.now().isoformat()
        }
        
        # Store progress as a separate object
        redis_conn.hmset(f"job_progress:{job_id}", progress_data)
        redis_conn.expire(f"job_progress:{job_id}", 86400)  # 24-hour TTL
        
        # Also store summary in metadata
        update_data = {
            'progress_percent': percent,
            'progress_stage': stage,
            'progress_message': message
        }
        update_job_status(job_id, metadata.get('status', 'in_progress'), update_data)
        
        logger.info(f"Updated progress for job {job_id}: {percent}% - {stage}")
        return True
    except Exception as e:
        logger.error(f"Error updating job progress: {e}")
        return False

def get_job_progress(job_id):
    """Get job progress from Redis"""
    if redis_conn is None:
        logger.error("Cannot get job progress: Redis connection not available")
        return None
    
    try:
        progress = redis_conn.hgetall(f"job_progress:{job_id}")
        if not progress:
            # If no specific progress, check metadata for basic status
            metadata = get_job_metadata(job_id)
            if metadata:
                # Basic progress info from metadata
                return {
                    'percent': metadata.get('progress_percent', '0'),
                    'stage': metadata.get('progress_stage', metadata.get('status', 'unknown')),
                    'message': metadata.get('progress_message', 'No progress details available'),
                    'updated_at': metadata.get('updated_at', metadata.get('created_at', ''))
                }
            return None
            
        # Convert bytes to strings if needed
        return {k.decode('utf-8') if isinstance(k, bytes) else k: 
                v.decode('utf-8') if isinstance(v, bytes) else v 
                for k, v in progress.items()}
    except Exception as e:
        logger.error(f"Error retrieving job progress: {e}")
        return None

# Routes
@app.route('/health', methods=['GET'])
def health():
    if redis_conn is None:
        return jsonify({"status": "error", "message": "Redis connection failed"}), 500
    
    # Test Redis connection
    try:
        redis_conn.ping()
        return jsonify({"status": "ok", "redis": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Redis error: {str(e)}"}), 500

@app.route('/redis-test', methods=['GET'])
def redis_test():
    """Test Redis connection and operations in detail"""
    results = {}
    
    # Test if Redis connection exists
    if redis_conn is None:
        return jsonify({
            "status": "error", 
            "message": "Redis connection is None",
            "details": "Redis connection failed during initialization"
        }), 500
    
    # Test 1: PING operation
    try:
        ping_result = redis_conn.ping()
        results["ping"] = {"success": True, "result": ping_result}
    except Exception as e:
        results["ping"] = {"success": False, "error": str(e)}
    
    # Test 2: Basic SET/GET
    try:
        test_key = "web_test_key"
        test_value = f"test_value_{datetime.datetime.now().isoformat()}"
        set_result = redis_conn.set(test_key, test_value)
        get_result = redis_conn.get(test_key)
        get_value = get_result.decode('utf-8') if isinstance(get_result, bytes) else get_result
        results["set_get"] = {
            "success": set_result and get_value == test_value,
            "set_result": set_result,
            "get_result": get_value,
            "expected": test_value
        }
    except Exception as e:
        results["set_get"] = {"success": False, "error": str(e)}
    
    # Test 3: Hash operations
    try:
        hash_key = "web_test_hash"
        hash_data = {
            "field1": "value1",
            "field2": "value2",
            "timestamp": datetime.datetime.now().isoformat()
        }
        hmset_result = redis_conn.hmset(hash_key, hash_data)
        hgetall_result = redis_conn.hgetall(hash_key)
        
        # Convert bytes to strings if needed
        decoded_result = {}
        if hgetall_result:
            for k, v in hgetall_result.items():
                k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                v_str = v.decode('utf-8') if isinstance(v, bytes) else v
                decoded_result[k_str] = v_str
                
        # Check if all fields are present
        all_fields_match = all(
            field in decoded_result and decoded_result[field] == value 
            for field, value in hash_data.items()
        )
        
        results["hash_ops"] = {
            "success": hmset_result and all_fields_match,
            "hmset_result": hmset_result,
            "hgetall_result": decoded_result,
            "expected": hash_data,
            "all_fields_match": all_fields_match
        }
    except Exception as e:
        results["hash_ops"] = {"success": False, "error": str(e)}
    
    # Test 4: Check RQ job queue
    try:
        queue = Queue('graphrag_processing', connection=redis_conn)
        job_count = len(queue.get_job_ids())
        results["job_queue"] = {
            "success": True,
            "queue_name": queue.name,
            "job_count": job_count,
            "registry_count": len(queue.started_job_registry.get_job_ids())
        }
    except Exception as e:
        results["job_queue"] = {"success": False, "error": str(e)}
    
    # Overall status
    all_tests_passed = all(test.get("success", False) for test in results.values())
    
    return jsonify({
        "status": "ok" if all_tests_passed else "error",
        "message": "All Redis tests passed" if all_tests_passed else "Some Redis tests failed",
        "details": results
    })

@app.route('/debug', methods=['GET'])
def debug_page():
    """A simple page to test if JavaScript is working"""
    return """
    <html>
    <head>
        <title>Debug Page</title>
        <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            button { padding: 10px; margin: 10px 0; }
            .indicator { display: none; color: red; font-weight: bold; }
        </style>
        <script>
            function testButton() {
                document.getElementById('testBtn').disabled = true;
                document.getElementById('indicator').style.display = 'block';
                return false;
            }
        </script>
    </head>
    <body>
        <h1>Debug Page</h1>
        <p>This page tests if JavaScript is working properly.</p>
        
        <form onsubmit="return testButton()">
            <button id="testBtn" type="submit">Test Button</button>
        </form>
        
        <div id="indicator" class="indicator">
            JavaScript is working! This text should appear when you click the button.
        </div>
    </body>
    </html>
    """

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    # Check Redis connection
    if redis_conn is None:
        return """
        <html>
        <head>
            <title>Service Unavailable</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Service Unavailable</h1>
            <div class="error">Error: Redis connection is not available</div>
            <p>The background processing service is not working. Please try again later or contact support.</p>
        </body>
        </html>
        """
    
    # For debugging
    logger.info(f"Received {'POST' if request.method == 'POST' else 'GET'} request to /upload")
    
    version = request.args.get('version', '')
    remember = request.args.get('remember', 'false').lower() == 'true'
    logger.info(f"Version from args: {version}, Remember: {remember}")
    
    # Session-based text storage
    session_key = f"last_input_text_{version}"
    
    if request.method == 'POST':
        # Handle form submission
        input_text = request.form.get('input_text', '')
        version = request.form.get('version', '')
        
        # Save the input text for this version in Redis
        if input_text and version:
            try:
                redis_conn.set(session_key, input_text)
                redis_conn.expire(session_key, 86400)  # 24 hour TTL
                logger.info(f"Saved input text for version {version} ({len(input_text)} chars)")
            except Exception as e:
                logger.error(f"Failed to save input text: {e}")
        
        if not input_text:
            return """
            <html>
            <head>
                <title>Upload page for version {}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: red; font-weight: bold; }}
                    textarea {{ width: 100%; height: 75vh; margin-top: 10px; padding: 10px; }}
                    button {{ padding: 10px 20px; background-color: #4CAF50; color: white; 
                             border: none; cursor: pointer; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>Upload page for version {}</h1>
                <div class="error">Error: Please provide input text</div>
                <form method="post">
                    <input type="hidden" name="version" value="{}">
                    <button type="submit">Submit</button>
                    <label for="input_text">Input text:</label>
                    <textarea name="input_text" id="input_text"></textarea>
                </form>
            </body>
            </html>
            """.format(version, version, version)
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Submit the job to the queue
        job_data = {
            'version': version,
            'input_text': input_text,
            'renew': True,
            'job_id': job_id
        }
        
        # Track the job status in Redis
        store_job_metadata(
            job_id=job_id,
            status='queued',
            created_at=datetime.datetime.now().isoformat(),
            version=version
        )
        
        # Enqueue the job
        try:
            # Don't pass timeout as an argument to the function but as a job parameter
            logger.info(f"Enqueueing job to '{job_queue.name}' queue with function path 'worker.process_job'")
            job = job_queue.enqueue(
                'worker.process_job', 
                args=[job_data],  # Pass as positional argument
                job_id=job_id, 
                job_timeout=1800,  # 30 minute timeout as job parameter, not function argument
                result_ttl=86400  # Keep result for 24 hours
            )
            
            # Log queue statistics after enqueueing
            try:
                queue_length = len(job_queue)
                job_ids = job_queue.job_ids
                logger.info(f"Queue '{job_queue.name}' now contains {queue_length} jobs. Job IDs: {job_ids[:5] if job_ids else '[]'}")
            except Exception as e:
                logger.warning(f"Failed to get queue statistics: {e}")
            logger.info(f"Job {job_id} submitted to queue with ID {job.id}")
            
            # Redirect to the job status page
            return redirect(url_for('job_status', job_id=job_id))
        except Exception as e:
            logger.error(f"Error queueing job: {e}")
            return f"""
            <html>
            <head>
                <title>Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: red; }}
                    pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <h1>Error</h1>
                <div class="error">Failed to queue job: {str(e)}</div>
                <pre>{traceback.format_exc()}</pre>
                <a href="/upload?version={version}">Go Back</a>
            </body>
            </html>
            """
    
    # Display the upload form (GET request)
    logger.info("Preparing to render upload form")
    
    # Try to retrieve saved input text if remember is true
    saved_text = ""
    if remember and version:
        session_key = f"last_input_text_{version}"
        try:
            saved_text_bytes = redis_conn.get(session_key)
            if saved_text_bytes:
                saved_text = saved_text_bytes.decode('utf-8') if isinstance(saved_text_bytes, bytes) else saved_text_bytes
                logger.info(f"Retrieved saved text for version {version} ({len(saved_text)} chars)")
        except Exception as e:
            logger.error(f"Failed to retrieve saved text: {e}")
    
    return f"""
    <html>
    <head>
        <title>Upload page for version {version}</title>
        <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            textarea {{ width: 100%; height: 70vh; margin-top: 10px; padding: 10px; }}
            button {{ padding: 10px 20px; background-color: #4CAF50; color: white; 
                     border: none; cursor: pointer; margin-bottom: 10px; }}
            .success-banner {{ 
                background-color: #dff2bf; 
                color: #4F8A10; 
                padding: 10px; 
                margin-bottom: 15px; 
                border-radius: 5px; 
                font-weight: bold;
            }}
            .success-icon {{ font-size: 1.2em; margin-right: 5px; }}
            .error-banner {{ 
                background-color: #ffbaba; 
                color: #d8000c; 
                padding: 10px; 
                margin-bottom: 15px; 
                border-radius: 5px; 
                font-weight: bold;
            }}
            .processing {{ 
                display: none;
                color: #0066cc; 
                font-weight: bold;
                margin-left: 15px;
                vertical-align: middle;
            }}
            .spinner {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0,102,204,.3);
                border-radius: 50%;
                border-top-color: #0066cc;
                animation: spin 1s ease-in-out infinite;
                vertical-align: middle;
                margin-right: 10px;
            }}
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        <script>
            function showProcessing() {{
                // Simple change - just make the button say "Processing..."
                document.getElementById('submitBtn').value = "Processing...";
                
                // Show the processing indicator
                document.getElementById('processingIndicator').style.display = "inline";
                
                return true;
            }}
        </script>
    </head>
    <body>
        <h1>Upload page for version {version}</h1>
        <form method="post" onsubmit="return showProcessing()">
            <input type="hidden" name="version" value="{version}">
            <input id="submitBtn" type="submit" value="Submit" style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">
            <span id="processingIndicator" style="display: none; margin-left: 10px; color: #666;">Processing...</span>
            <label for="input_text">Input text:</label>
            <textarea name="input_text" id="input_text">{saved_text}</textarea>
        </form>
    </body>
    </html>
    """

@app.route('/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Check the status of a job."""
    # Debug logging
    logger.info(f"Checking status for job: {job_id}")
    
    if redis_conn is None:
        logger.error("Redis connection is None when checking job status")
        return """
        <html>
        <head>
            <title>Service Unavailable</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Service Unavailable</h1>
            <div class="error">Error: Redis connection is not available</div>
            <p>The background processing service is not working. Please try again later or contact support.</p>
        </body>
        </html>
        """
    
    # Debug: Log if metadata exists for this job
    metadata = get_job_metadata(job_id)
    if metadata:
        logger.info(f"Found job metadata: {metadata}")
    else:
        logger.warning(f"No metadata found for job: {job_id}")
    
    # Debug: Log if progress exists for this job
    progress_info = get_job_progress(job_id)
    if progress_info:
        logger.info(f"Found job progress: {progress_info}")
    else:
        logger.warning(f"No progress found for job: {job_id}")
        
    try:
        # Attempt to get the job from RQ
        logger.info(f"Attempting to fetch RQ job: {job_id}")
        job = Job.fetch(job_id, connection=redis_conn)
        
        # Get the job status
        if job.is_finished:
            status = "completed"
            result = job.result
            
            # Return success or error page based on job result
            if result and result.get('status') == 'success':
                # Make sure we have the version for the "Process Another Document" link
                version = result.get('version', '')
                logger.info(f"Success page with version: {version}")
                uploaded_message = "Successfully uploaded to data.world" if result.get("uploaded_to_dataworld", False) else "Processing completed successfully"
                
                return f"""
                <html>
                <head>
                    <title>Processing Complete</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .success {{ color: green; font-weight: bold; }}
                        pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        .status-banner {{
                            background-color: #dff2bf;
                            color: green;
                            padding: 15px;
                            margin-bottom: 20px;
                            border-radius: 5px;
                            font-weight: bold;
                        }}
                        .success-icon {{ font-size: 1.2em; margin-right: 5px; }}
                        .button {{
                            display: inline-block;
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 15px;
                            margin: 10px 5px 10px 0;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            text-decoration: none;
                        }}
                        .button:hover {{
                            background-color: #45a049;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Processing Complete</h1>
                    
                    <div class="status-banner">
                        <span class="success-icon">✅</span> {uploaded_message}
                    </div>
                    
                    <p>Your text has been processed and the results have been saved to data.world.</p>
                    <p>Would you like to process another document?</p>
                    
                    <div>
                        <a href="/upload?version={version}&remember=true" class="button">Process Another Document</a>
                    </div>
                </body>
                </html>
                """
            else:
                # Error page
                error_message = result.get('message', 'Unknown error') if result else 'Job failed'
                version = result.get('version', '') if result else ''
                
                return f"""
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .error {{ color: red; }}
                        pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        .status-banner {{
                            background-color: #ffbaba;
                            color: red;
                            padding: 15px;
                            margin-bottom: 20px;
                            border-radius: 5px;
                            font-weight: bold;
                        }}
                        .error-icon {{ font-size: 1.2em; margin-right: 5px; }}
                        .button {{
                            display: inline-block;
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 15px;
                            margin: 10px 5px 10px 0;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            text-decoration: none;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Error</h1>
                    
                    <div class="status-banner">
                        <span class="error-icon">❌</span> {error_message}
                    </div>
                    
                    <a href="/upload?version={version}&remember=true" class="button">Try Again</a>
                </body>
                </html>
                """
        elif job.is_failed:
            status = "failed"
            # Get the exception info
            exc_info = job.exc_info or "Unknown error"
            
            return f"""
            <html>
            <head>
                <title>Job Failed</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: red; }}
                    pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    .status-banner {{
                        background-color: #ffbaba;
                        color: red;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    .error-icon {{ font-size: 1.2em; margin-right: 5px; }}
                    .button {{
                        display: inline-block;
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 15px;
                        margin: 10px 5px 10px 0;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        text-decoration: none;
                    }}
                </style>
            </head>
            <body>
                <h1>Job Failed</h1>
                
                <div class="status-banner">
                    <span class="error-icon">❌</span> The job processing failed.
                </div>
                
                <h2>Error Details:</h2>
                <pre>{exc_info}</pre>
                
                <a href="/upload?version={metadata.get('version', '')}&remember=true" class="button">Try Again</a>
            </body>
            </html>
            """
        else:
            # Job is still in progress
            status = job.get_status()
            
            # Get progress information
            progress_info = get_job_progress(job_id)
            percent = 0
            stage = "Starting..."
            message = "Job is starting..."
            
            if progress_info:
                percent = int(progress_info.get('percent', 0))
                stage = progress_info.get('stage', status)
                message = progress_info.get('message', 'Processing your request...')
            
            return f"""
            <html>
            <head>
                <title>Job Status</title>
                <meta http-equiv="refresh" content="5;URL='/job/{job_id}'" />
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .info {{ color: #0066cc; }}
                    .status-banner {{
                        background-color: #e7f3fe;
                        color: #0066cc;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    .info-icon {{ font-size: 1.2em; margin-right: 5px; }}
                    .progress-section {{
                        margin: 25px 0;
                        padding: 15px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                    }}
                    .progress-title {{
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    .progress-bar-container {{
                        width: 100%;
                        height: 30px;
                        background-color: #f3f3f3;
                        border-radius: 5px;
                        margin-top: 10px;
                        margin-bottom: 10px;
                        overflow: hidden;
                    }}
                    .progress-bar {{
                        height: 100%;
                        background-color: #4CAF50;
                        border-radius: 5px;
                        width: {percent}%;
                        transition: width 1s ease-in-out;
                    }}
                    .progress-status {{
                        margin-top: 10px;
                        color: #666;
                    }}
                    .stage-indicator {{
                        font-weight: bold;
                        color: #0066cc;
                        margin-top: 15px;
                    }}
                    .stage-message {{
                        margin-top: 5px;
                        color: #333;
                        font-style: italic;
                    }}
                </style>
            </head>
            <body>
                <h1>Job Status</h1>
                
                <div class="status-banner">
                    <span class="info-icon">ℹ️</span> Your job is currently {status}. This page will refresh automatically.
                </div>
                
                <p>Job ID: {job_id}</p>
                
                <div class="progress-section">
                    <div class="progress-title">Current Progress:</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar"></div>
                    </div>
                    <div class="progress-status">{percent}% complete</div>
                    
                    <div class="stage-indicator">Current Stage: {stage}</div>
                    <div class="stage-message">{message}</div>
                </div>
                
                <p>This may take several minutes. Please keep this page open.</p>
                
                <script>
                    // Set the actual width based on percentage
                    document.querySelector('.progress-bar').style.width = '{percent}%';
                    
                    // This will refresh the page every 5 seconds to check the job status
                    setTimeout(function() {{
                        window.location.reload();
                    }}, 5000);
                </script>
            </body>
            </html>
            """
    except Exception as e:
        # Job not found or other error
        logger.error(f"Error checking job status: {e}", exc_info=True)
        
        # Create a detailed debug section with all available information
        debug_info = ""
        if metadata:
            debug_info += f"<h3>Job Metadata</h3><pre>{metadata}</pre>"
        if progress_info:
            debug_info += f"<h3>Job Progress</h3><pre>{progress_info}</pre>"
        
        # Check if the error is related to Redis connection
        redis_error = "Unknown error"
        try:
            redis_status = "Connected" if redis_conn.ping() else "Not responding"
        except Exception as redis_e:
            redis_error = str(redis_e)
            redis_status = "Error"
        
        debug_info += f"<h3>Redis Status</h3><p>Status: {redis_status}</p><p>Error: {redis_error}</p>"
        
        return f"""
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .error {{ color: red; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .button {{
                    display: inline-block;
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    margin: 10px 5px 10px 0;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                }}
                .debug-section {{ 
                    margin-top: 30px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                summary {{ cursor: pointer; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Error</h1>
            <div class="error">Job not found or error checking status: {str(e)}</div>
            <a href="/upload?version={metadata.get('version', '') if metadata else ''}&remember=true" class="button">Return to Upload Page</a>
            
            <details class="debug-section">
                <summary>Debug Information</summary>
                <div>
                    <h2>Job Information</h2>
                    <p>Job ID: {job_id}</p>
                    <p>Error: {str(e)}</p>
                    {debug_info}
                </div>
            </details>
        </body>
        </html>
        """

@app.route('/process', methods=['POST'])
def process():
    """API endpoint for submitting jobs programmatically."""
    if redis_conn is None:
        return jsonify({"error": "Redis connection is not available"}), 503
    
    if not request.json:
        return jsonify({"error": "Missing JSON payload"}), 400
    
    version = request.json.get('version')
    renew = request.json.get('renew', False)
    input_text = request.json.get('input_text', None)
    
    logger.info(f"Received API request with version={version}, renew={renew}, input_text_provided={input_text is not None}")
    
    if not input_text:
        return jsonify({"error": "Missing input_text field in request"}), 400
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Submit the job to the queue
    job_data = {
        'version': version,
        'input_text': input_text,
        'renew': renew,
        'job_id': job_id
    }
    
    # Track the job status in Redis
    store_job_metadata(
        job_id=job_id,
        status='queued',
        created_at=datetime.datetime.now().isoformat(),
        version=version
    )
    
    # Enqueue the job
    try:
        # Don't pass timeout as an argument to the function but as a job parameter
        logger.info(f"Enqueueing job to '{job_queue.name}' queue with function path 'worker.process_job'")
        job = job_queue.enqueue(
            'worker.process_job', 
            args=[job_data],  # Pass as positional argument
            job_id=job_id, 
            job_timeout=1800,  # 30 minute timeout as job parameter, not function argument
            result_ttl=86400  # Keep result for 24 hours
        )
        
        # Log queue statistics after enqueueing
        try:
            queue_length = len(job_queue)
            job_ids = job_queue.job_ids
            logger.info(f"Queue '{job_queue.name}' now contains {queue_length} jobs. Job IDs: {job_ids[:5] if job_ids else '[]'}")
        except Exception as e:
            logger.warning(f"Failed to get queue statistics: {e}")
        logger.info(f"API job {job_id} submitted to queue with ID {job.id}")
        
        # Return the job ID for status checking
        return jsonify({
            "status": "queued",
            "job_id": job_id,
            "message": "Job submitted to processing queue",
            "check_status_url": f"/api/job/{job_id}"
        })
    except Exception as e:
        logger.error(f"Error queueing API job: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to queue job: {str(e)}"
        }), 500

@app.route('/direct-process', methods=['POST'])
def direct_process():
    """Test endpoint to directly process a job without using RQ for troubleshooting."""
    if not request.json:
        return jsonify({"error": "Missing JSON payload"}), 400
    
    version = request.json.get('version')
    renew = request.json.get('renew', False)
    input_text = request.json.get('input_text', None)
    
    logger.info(f"Received DIRECT process request with version={version}, renew={renew}, input_text_provided={input_text is not None}")
    
    if not input_text:
        return jsonify({"error": "Missing input_text field in request"}), 400
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Create the job data
    job_data = {
        'version': version,
        'input_text': input_text,
        'renew': renew,
        'job_id': job_id
    }
    
    # Track the job status in Redis
    store_job_metadata(
        job_id=job_id,
        status='processing',
        created_at=datetime.datetime.now().isoformat(),
        version=version
    )
    
    # Import the processing function and run it directly
    try:
        # Import the worker module
        import sys
        import importlib.util
        
        # Load the worker module
        spec = importlib.util.spec_from_file_location("worker", "worker.py")
        worker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker_module)
        
        # Execute the job directly
        logger.info(f"Starting direct job processing for job {job_id}")
        result = worker_module.process_job(job_data)
        logger.info(f"Direct job processing completed with status: {result.get('status', 'unknown')}")
        
        # Return detailed result
        return jsonify({
            "job_id": job_id,
            "direct_result": result,
            "message": "Job processed directly without using Redis queue",
            "status": result.get("status", "unknown")
        })
    except Exception as e:
        logger.error(f"Error in direct job processing: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "job_id": job_id,
            "message": f"Direct job processing failed: {str(e)}",
            "error_details": traceback.format_exc()
        }), 500

@app.route('/api/job/<job_id>', methods=['GET'])
def api_job_status(job_id):
    """API endpoint to check job status."""
    if redis_conn is None:
        return jsonify({"error": "Redis connection is not available"}), 503
    
    try:
        # Attempt to get the job from RQ
        job = Job.fetch(job_id, connection=redis_conn)
        
        # Get the job status
        if job.is_finished:
            # Job is finished, return the result
            result = job.result
            return jsonify(result)
        elif job.is_failed:
            # Job failed
            exc_info = job.exc_info or "Unknown error"
            return jsonify({
                "status": "error",
                "message": "Job failed",
                "error_details": exc_info,
                "job_id": job_id
            }), 500
        else:
            # Job is still in progress
            status = job.get_status()
            
            # Get progress information
            progress_info = get_job_progress(job_id)
            response = {
                "status": status,
                "message": f"Job is {status}",
                "job_id": job_id
            }
            
            # Add progress info if available
            if progress_info:
                response["progress"] = {
                    "percent": progress_info.get('percent', 0),
                    "stage": progress_info.get('stage', status),
                    "message": progress_info.get('message', 'Processing your request...'),
                    "updated_at": progress_info.get('updated_at', '')
                }
                
            return jsonify(response)
    except Exception as e:
        # Job not found or other error
        logger.error(f"Error checking API job status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error checking job status: {str(e)}",
            "job_id": job_id
        }), 404

# Download endpoint removed since it's not needed

@app.route('/workshop', methods=['GET'])
def workshop_page():
    """Display a list of projects for a given organization with optional filtering"""
    org = request.args.get('org', '')
    filter_text = request.args.get('filter', '')
    
    if not org:
        return "Please specify an organization with the 'org' parameter", 400
    
    # Get the API token from environment
    api_token = os.getenv('DW_AUTH_TOKEN')
    if not api_token:
        return "API token not found in environment variables", 500
    
    # Fetch projects from the data.world API
    url = f"https://api.data.world/v0/projects/{org}?limit=100"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}"
    }
    
    try:
        import requests
        response = requests.get(url, headers=headers)
        if not response.ok:
            response.raise_for_status()
        
        projects_data = response.json()
        
        # Build the HTML response
        projects_html = ""
        project_count = 0
        total_projects = len(projects_data.get('records', []))
        
        for project in projects_data.get('records', []):
            project_id = project.get('id')
            project_title = project.get('title', project_id)
            
            # Apply filter if specified
            if filter_text and filter_text.lower() not in project_title.lower():
                continue
                
            project_count += 1
            upload_url = f"https://kgc-ddw-entity-db8350081a81.herokuapp.com/upload?version={org}/{project_id}"
            projects_html += f'<li><a href="{upload_url}" target="_blank">{project_title}</a></li>\n'
        
        # Filter status message
        filter_message = ""
        if filter_text:
            filter_message = f"""
            <div class="filter-status">
                <p>Showing {project_count} out of {total_projects} projects containing "{filter_text}"</p>
                <p><a href="/workshop?org={org}" class="reset-filter">Show all projects</a></p>
            </div>
            """
        
        # Filter form
        filter_form = f"""
        <div class="filter-form">
            <form method="get" action="/workshop">
                <input type="hidden" name="org" value="{org}">
                <label for="filter">Filter projects:</label>
                <input type="text" id="filter" name="filter" value="{filter_text}" placeholder="Enter filter text">
                <button type="submit">Apply Filter</button>
            </form>
        </div>
        """
        
        return f"""
        <html>
        <head>
            <title>Workshop Projects for {org}</title>
            <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #333; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin-bottom: 10px; padding: 8px; border-radius: 4px; }}
                li:hover {{ background-color: #f5f5f5; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .filter-form {{ margin: 20px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                .filter-form input[type="text"] {{ padding: 8px; width: 300px; margin-right: 10px; }}
                .filter-form button {{ padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                .filter-form button:hover {{ background-color: #45a049; }}
                .filter-status {{ margin-bottom: 20px; color: #666; }}
                .reset-filter {{ color: #666; text-decoration: underline; }}
                .no-results {{ padding: 20px; background-color: #f8f8f8; border-radius: 5px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Workshop Projects for {org}</h1>
                <p>Click on a project to open the upload page in a new tab.</p>
            </div>
            
            {filter_form}
            {filter_message}
            
            {('<ul>' + projects_html + '</ul>') if project_count > 0 else '<div class="no-results">No projects match your filter criteria.</div>'}
        </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Error in workshop page: {e}", exc_info=True)
        return f"Error processing workshop data: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)