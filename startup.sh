#!/bin/bash

# Install system-level dependency for PDF processing
echo "Updating and installing poppler-utils..."
apt-get update && apt-get install -y poppler-utils

# Install Python packages from requirements.txt
echo "Installing Python requirements..."
pip install -r requirements.txt

# Start the Gunicorn server
# It will bind to the port Render provides via the $PORT environment variable.
echo "Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:$PORT --timeout 600 app:app
