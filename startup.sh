#!/bin/bash

# Start the FastAPI application with gunicorn
gunicorn main:app -c gunicorn.conf.py