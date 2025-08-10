#!/bin/bash

# Simple startup script for Azure App Service
echo "Starting Solavix AI Backend..."

# Check if uvicorn is available
if command -v uvicorn &> /dev/null; then
    echo "Using uvicorn directly..."
    uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "Using python -m uvicorn..."
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
fi