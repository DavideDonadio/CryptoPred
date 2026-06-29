#!/bin/bash
set -e
SCRIPT_DIR="/app"
cd "$SCRIPT_DIR"
sleep 30
echo "--- Starting Reddit scrape: $(date) ---"
python "$SCRIPT_DIR"/fetch_reddit.py 2>&1
echo "--- Finished scrape, building live data: $(date) ---"
python "$SCRIPT_DIR"/live_data_builder.py 2>&1
echo "--- Finished Automation Cycle: $(date) ---"