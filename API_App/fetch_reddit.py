#!/usr/bin/env python3
import os
import re
import json
import time
import subprocess
import undetected_chromedriver as uc

BASE_DIR = "/app"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/clean_reddit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBREDDITS = ["bitcoin", "cryptocurrency", "cryptocurrencies", "crypto_currency_news", "ethereum", "solana"]

print(f"Starting headless Chrome fetch at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def _detect_chrome_major_version(chrome_bin: str | None) -> int | None:
    if not chrome_bin or not os.path.exists(chrome_bin):
        return None
    try:
        out = subprocess.check_output([chrome_bin, "--version"], text=True, timeout=5)
        match = re.search(r"(\d+)\.", out)
        return int(match.group(1)) if match else None
    except Exception:
        return None

options = uc.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

chrome_kwargs = {"options": options}
chrome_bin = os.environ.get("CHROME_BIN")
if chrome_bin and os.path.exists(chrome_bin):
    chrome_kwargs["browser_executable_path"] = chrome_bin
    version_main = _detect_chrome_major_version(chrome_bin)
    if version_main is not None:
        chrome_kwargs["version_main"] = version_main

chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
if chromedriver_path and os.path.exists(chromedriver_path):
    chrome_kwargs["driver_executable_path"] = chromedriver_path

driver = uc.Chrome(**chrome_kwargs)

try:
    print("Priming browser cookies on Reddit homepage...")
    driver.get("https://www.reddit.com")
    time.sleep(1)

    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=25"
        file_path = os.path.join(OUTPUT_DIR, f"{sub}.json")

        print(f"Navigating to r/{sub} feed...")
        driver.get(f"https://www.reddit.com/r/{sub}/new")
        time.sleep(1)

        print(f"Extracting JSON source payload...")
        driver.get(url)
        time.sleep(1)

        raw_text = driver.find_element("xpath", "//html").text

        try:
            json_data = json.loads(raw_text)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
            print(f"Successfully saved {sub}.json")
        except json.JSONDecodeError:
            print(f"Error: Blocked on r/{sub}. Page snippet: {raw_text[:200]}")

        time.sleep(1)

finally:
    driver.quit()
    print("Fetch complete.")