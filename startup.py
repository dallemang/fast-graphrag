#!/usr/bin/env python3
"""
Startup script to disable SSL certificate verification for Redis connections.
This script should be run before worker.py or app.py.
"""

import os
import sys
import ssl
import urllib3

print("Applying aggressive SSL certificate verification disabling...")

# Disable SSL warnings
urllib3.disable_warnings()

# Create a custom SSL context that doesn't verify certificates
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.options |= ssl.OP_NO_SSLv2
ssl_context.options |= ssl.OP_NO_SSLv3

# Monkey patch the default SSL context
ssl._create_default_https_context = lambda: ssl_context

# Set environment variables to disable SSL verification
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['SSL_CERT_REQS'] = 'none'
os.environ['REDIS_SSL_CERT_REQS'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

print("SSL certificate verification disabled!")
print("Environment variables set:")
print(f"PYTHONHTTPSVERIFY: {os.environ.get('PYTHONHTTPSVERIFY')}")
print(f"SSL_CERT_REQS: {os.environ.get('SSL_CERT_REQS')}")
print(f"REDIS_SSL_CERT_REQS: {os.environ.get('REDIS_SSL_CERT_REQS')}")
print(f"PYTHONWARNINGS: {os.environ.get('PYTHONWARNINGS')}")

print("Executing:", " ".join(sys.argv[1:]))
if len(sys.argv) > 1:
    # Execute the command that was passed after this script
    os.execvp(sys.argv[1], sys.argv[1:])
else:
    print("No command specified to run. SSL settings have been applied to this process only.")