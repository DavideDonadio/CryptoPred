#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "--- Starting Reddit scrape: $(date) ---"
xvfb-run --server-args="-screen 0 1920x1080x24" "$PROJECT_DIR"/.venv/bin/python "$SCRIPT_DIR"/fetch_reddit.py
echo "--- Finished scrape, building live data: $(date) ---"
"$PROJECT_DIR"/.venv/bin/python "$SCRIPT_DIR"/live_data_builder.py
echo "--- Finished Automation Cycle: $(date) ---"
