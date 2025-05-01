FROM python:3.10

WORKDIR /app

# Install system dependencies including redis-cli for connection testing
RUN apt-get update && \
    apt-get install -y --no-install-recommends redis-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
# Use local Redis for testing if not provided
ENV REDIS_URL=redis://localhost:6379

# Expose the port
EXPOSE 5000

# Create startup scripts
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Command to run
CMD ["/app/entrypoint.sh"]