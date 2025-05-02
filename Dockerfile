FROM python:3.10

WORKDIR /app

# Install Poetry and other requirements
RUN pip install --no-cache-dir poetry flask gunicorn rdflib networkx requests redis rq

# First copy all files so README.md is available
COPY . .

# Configure poetry to not use virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies (without dev dependencies)
RUN poetry install --without dev

# Create necessary directories (if they don't exist)
RUN mkdir -p ./print3D_example ./output ./hygiene

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV REDIS_HOST=verified-stag-21591.upstash.io
ENV REDIS_PORT=6379
# Removed REDIS_SSL as we now always use SSL with Upstash

# Expose the port
EXPOSE 5000

# Create entry point script to handle both web and worker processes
RUN echo '#!/bin/bash\n\
# Debug info to understand what process we should run\n\
echo "Current DYNO environment variable: $DYNO"\n\
echo "Current DYNO_TYPE environment variable: $DYNO_TYPE"\n\
\n\
if [[ "$DYNO" == *"worker"* ]] || [[ "$DYNO_TYPE" == "worker" ]]; then\n\
    echo "Starting worker process..."\n\
    python worker.py\n\
else\n\
    echo "Starting web process..."\n\
    exec gunicorn --bind 0.0.0.0:$PORT --timeout 30 --threads=4 --workers=1 --keep-alive=5 --access-logfile=- app_with_redis:app\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Command to run - using exec form with entrypoint script
CMD ["/app/entrypoint.sh"]