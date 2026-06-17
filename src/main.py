from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import os
import json
import pickle
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timezone, timedelta

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_reddit", "live_data.csv")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "xgb.pkl")
LGB_MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "lgb.pkl")
PREDICTIONS = os.path.join(BASE_DIR, "data", "clean_reddit", "predictions.csv")

def run_prediction():
    
    if not os.path.exists(LGB_MODEL_PATH) or not os.path.exists(XGB_MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model files not found")

    row = load_last_live_data_row()
    to_drop_cols = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target"]
    features = row.drop(columns=to_drop_cols)
    features = features.fillna(0)
    
    lgb = pickle.load(open(LGB_MODEL_PATH, 'rb'))
    xgb = pickle.load(open(XGB_MODEL_PATH, 'rb'))
    
    features = features[lgb.feature_names_in_]
    
    xgb_low, xgb_high = xgb.predict_proba(features)[0]
    lgb_low, lgb_high = lgb.predict_proba(features)[0]

    avg_low = (xgb_low + lgb_low) / 2.0
    avg_high = (xgb_high + lgb_high) / 2.0
    prediction = "low" if avg_low >= avg_high else "high"
    
    pred_row = pd.DataFrame([{
    "timestamp": row.index[0].isoformat(),
    "prediction": prediction,
    "prob_low": avg_low,
    "prob_high": avg_high,
    }])
    
    if not os.path.exists(PREDICTIONS):
        pred_row.to_csv(PREDICTIONS, index = False)
    else:
        
        df = pd.read_csv(PREDICTIONS)
        if pred_row["timestamp"].iloc[0] not in df["timestamp"].values:
            pred_row.to_csv(PREDICTIONS, mode = "a", header = False, index = False)

    return {
        "prediction": prediction,
        "prob_low": avg_low,
        "prob_high": avg_high,
    }

def load_btc_price_series(limit: int) -> pd.DataFrame:
    if not os.path.exists(LIVE_DATA_PATH):
        raise HTTPException(status_code=404, detail="live_data.csv not found")

    df = pd.read_csv(LIVE_DATA_PATH, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    return df.tail(limit)
    
def load_last_live_data_row() -> pd.DataFrame:
    if not os.path.exists(LIVE_DATA_PATH):
        raise HTTPException(status_code=404, detail="live_data.csv not found")

    df = pd.read_csv(LIVE_DATA_PATH, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    return df.tail(1)


@app.get("/api/btc-price")
def btc_price(limit: int = Query(12, ge=1, le=720)):
    df = load_btc_price_series(limit)
    return {
            "timestamps": [ts.isoformat() for ts in df.index],
            "prices": [float(val) for val in df["BTC-USD_close"].tolist()],
    }
        
@app.get("/api/volatility")
def volatility(limit: int = Query(1, ge=1, le=720)):
    return(run_prediction())

async def scheduler():
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(minute = 2, second = 0)
        if next_run <= now:
            next_run += timedelta(hours = 1)
        await asyncio.sleep((next_run - now).total_seconds())
        run_prediction()
        
@app.on_event("startup")
async def startup():
    asyncio.create_task(scheduler())
    
@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
        return HTMLResponse(
                """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BTC Hourly Price</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: "Space Grotesk", system-ui, sans-serif;
            background: #0e1118;
            color: #f4f5f7;
        }
        .page {
            max-width: 1000px;
            margin: 0 auto;
            padding: 28px 20px 40px;
        }
        h1 {
            font-size: 28px;
            margin: 0 0 6px;
        }
        p {
            margin: 0 0 18px;
            color: #b8c0cc;
        }
        .card {
            background: #161b25;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 18px;
        }
        .chart-wrap {
            height: 360px;
        }
        .status {
            margin-top: 10px;
            font-size: 13px;
            color: #9aa3b2;
        }
    </style>
</head>
<body>
    <div class="page">
        <h1>BTC Hourly Price</h1>
        <p>Latest hourly close from the live data feed.</p>
        <div class="card">
            <div class="chart-wrap">
                <canvas id="btcChart"></canvas>
            </div>
            <div class="status" id="status"></div>
        </div>
        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Volatility Prediction</h2>
            <div class="status" id="volatilityStatus">Loading...</div>
            <div class="status" id="volatilityDetails"></div>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById("status");

        // Helper function to format ISO strings into MM-DDTHH:MM (Year removed)
        function formatTimestamp(isoString) {
            const date = new Date(isoString);
            
            const mm = String(date.getMonth() + 1).padStart(2, '0');
            const dd = String(date.getDate()).padStart(2, '0');
            const hh = String(date.getHours()).padStart(2, '0');
            const min = String(date.getMinutes()).padStart(2, '0');
            
            return `${mm}-${dd}T${hh}:${min}`;
        }

        async function loadData() {
            const res = await fetch("/api/btc-price?limit=24");
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || "Failed to load data");
            }
            return res.json();
        }

        async function loadVolatility() {
            const res = await fetch("/api/volatility");
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || "Failed to load volatility");
            }
            return res.json();
        }

        let chartInstance = null;

        function renderChart(payload) {
            const formattedLabels = payload.timestamps.map(formatTimestamp);
            const ctx = document.getElementById("btcChart").getContext("2d");
            
            if (chartInstance) {
                chartInstance.data.labels = formattedLabels;
                chartInstance.data.datasets[0].data = payload.prices;
                chartInstance.update();
                return;
            }
            chartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: formattedLabels,
                    datasets: [
                        {
                            label: "BTC close",
                            data: payload.prices,
                            borderColor: "#58d2a3",
                            backgroundColor: "rgba(88, 210, 163, 0.12)",
                            tension: 0.3,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: { 
                                color: "#b8c0cc", 
                                maxTicksLimit: 24
                                // Removed fixed 90-degree rotations to revert to auto layout
                            },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                        y: {
                            ticks: { color: "#b8c0cc" },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                    },
                    plugins: {
                        legend: { labels: { color: "#f4f5f7" } },
                    },
                },
            });
        }

        async function refreshAll() {
            try {
                const payload = await loadData();
                renderChart(payload);
                statusEl.textContent = `Loaded ${payload.prices.length} hours.`;
            } catch (err) {
                statusEl.textContent = `Error: ${err.message}`;
            }

            try {
                const volatility = await loadVolatility();
                const status = document.getElementById("volatilityStatus");
                const details = document.getElementById("volatilityDetails");
                status.textContent = `Prediction: ${volatility.prediction.toUpperCase()}`;
                details.textContent = `Low: ${(volatility.prob_low * 100).toFixed(1)}% | High: ${(volatility.prob_high * 100).toFixed(1)}%`;
            } catch (err) {
                const status = document.getElementById("volatilityStatus");
                status.textContent = `Error: ${err.message}`;
            }
        }

        function scheduleHourlyRefresh() {
            const now = new Date();
            const next = new Date(now);
            next.setMinutes(2, 0, 0);
            if (next <= now) {
                next.setHours(next.getHours() + 1);
            }
            const delay = next.getTime() - now.getTime();
            setTimeout(() => {
                refreshAll();
                setInterval(refreshAll, 60 * 60 * 1000);
            }, delay);
        }

        refreshAll();
        scheduleHourlyRefresh();
    </script>
</body>
</html>
""")