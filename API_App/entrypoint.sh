#!/bin/bash
set -e

if [ ! -f /app/data/clean_reddit/live_data.csv ]; then
    echo "First run detected — populating initial data..."
    /app/pipeline.sh || true
    echo "Initial data population complete."
fi

echo "Starting cron daemon..."
cron

echo "Starting FastAPI server on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
