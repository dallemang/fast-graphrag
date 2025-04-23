FROM python:3.10

WORKDIR /app

# Install Poetry and other requirements
RUN pip install --no-cache-dir poetry flask gunicorn rdflib networkx requests

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

# Expose the port
EXPOSE 5000

# Use a shell script to handle environment variable substitution properly
RUN echo '#!/bin/bash\nexec gunicorn --bind 0.0.0.0:$PORT --timeout 30 --threads=4 --workers=1 --keep-alive=5 --access-logfile=- app:app' > /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

# Command to run - using exec form with entrypoint script
CMD ["/app/docker-entrypoint.sh"]