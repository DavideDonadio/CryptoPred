from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cryptopred")

BASE_DIR = "/app"
LIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_reddit", "live_data.csv")
PREDICTIONS = os.path.join(BASE_DIR, "data", "clean_reddit", "predictions.csv")
THRESHOLDS_JSON = os.path.join(BASE_DIR, "data", "models", "optimal_thresholds.json")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "xgb_volatility.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        backfill_vol_targets()
        logger.info("[startup] initial backfill complete")
    except Exception as e:
        logger.error("[startup] backfill failed: %s", e)
    asyncio.create_task(scheduler())
    yield

app = FastAPI(lifespan=lifespan)

def load_threshold_from_json() -> float:
    default = 0.50
    if not os.path.exists(THRESHOLDS_JSON):
        logger.warning("%s not found. Using default %.2f.", THRESHOLDS_JSON, default)
        return default
    try:
        with open(THRESHOLDS_JSON, "r") as f:
            data = json.load(f)
        return float(data.get("xgb", default))
    except (json.JSONDecodeError, IOError, ValueError):
        logger.warning("Could not parse %s. Using default %.2f.", THRESHOLDS_JSON, default)
        return default

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def write_prediction(pred_row: pd.DataFrame) -> None:
    df = safe_read_csv(PREDICTIONS)

    if df.empty or "timestamp" not in df.columns:
        pred_row.to_csv(PREDICTIONS, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    new_ts = pd.to_datetime(pred_row["timestamp"].iloc[0], utc=True)

    if new_ts not in df["timestamp"].values:
        pred_row.to_csv(PREDICTIONS, mode="a", header=False, index=False, date_format="%Y-%m-%d %H:%M:%S%z")

def compute_prediction() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    row = load_last_live_data_row()
    to_drop_cols = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target"]
    features = row.drop(columns=to_drop_cols, errors="ignore")

    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        logger.warning("Feature columns with NaN values (filling with 0): %s", nan_cols)

    features = features.fillna(0)
    xgb = pickle.load(open(MODEL_PATH, "rb"))

    if hasattr(xgb, "feature_names_in_"):
        features = features[xgb.feature_names_in_]
    elif hasattr(xgb, "feature_name_"):
        features = features[xgb.feature_name_]

    _, xgb_high = xgb.predict_proba(features)[0]
    current_threshold = load_threshold_from_json()
    signal = 1 if xgb_high >= current_threshold else 0

    return {
        "prediction": "high" if signal == 1 else "low",
        "prob_high": xgb_high,
        "signal_triggered": signal,
        "threshold": current_threshold,
        "timestamp": pd.to_datetime(row.index[0], utc=True),
    }

def run_prediction() -> dict:
    result = compute_prediction()
    pred_row = pd.DataFrame([{
        "timestamp": result["timestamp"],
        "prediction": result["prediction"],
        "prob_high": result["prob_high"],
        "signal_triggered": result["signal_triggered"],
        "threshold_used": result["threshold"],
    }])
    write_prediction(pred_row)
    return result

def _load_live_data(limit: int | None = None) -> pd.DataFrame:
    df = safe_read_csv(LIVE_DATA_PATH, index_col=0)
    if df.empty:
        raise FileNotFoundError("live_data.csv not found or is empty")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    if limit is not None:
        return df.tail(limit)
    return df

def backfill_vol_targets() -> None:
    df = safe_read_csv(LIVE_DATA_PATH, index_col=0)
    if df.empty:
        return

    df["BTC_vol_target"] = pd.cut(
        df["BTC-USD_volatility"],
        bins=[-np.inf, 0.00638, 0.01069, np.inf],
        labels=[0, 1, 2]
    ).shift(-1)

    df["ETH_vol_target"] = pd.cut(
        df["ETH-USD_volatility"],
        bins=[-np.inf, 0.00867, 0.01427, np.inf],
        labels=[0, 1, 2]
    ).shift(-1)

    df["SOL_vol_target"] = pd.cut(
        df["SOL-USD_volatility"],
        bins=[-np.inf, 0.01205, 0.01960, np.inf],
        labels=[0, 1, 2]
    ).shift(-1)

    df.to_csv(LIVE_DATA_PATH)
    logger.info("[backfill] vol targets updated")

def load_btc_price_series(limit: int) -> pd.DataFrame:
    return _load_live_data(limit)

def load_last_live_data_row() -> pd.DataFrame:
    return _load_live_data(1)

@app.get("/api/performance")
def get_performance_metrics():
    preds_df = safe_read_csv(PREDICTIONS)
    live_df = safe_read_csv(LIVE_DATA_PATH, index_col=0)

    if preds_df.empty or live_df.empty:
        return {"msg": "No historical tracking data available yet."}

    live_df.index = pd.to_datetime(live_df.index, utc=True, errors="coerce")
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)

    now = datetime.now(timezone.utc)
    live_df = live_df[live_df.index + pd.Timedelta(hours=1) <= now]
    live_df = live_df.dropna(subset=["BTC_vol_target"])

    if live_df.empty:
        return {"msg": "Waiting for ground-truth data to finalize."}

    live_df["true_target"] = (live_df["BTC_vol_target"].astype(float) == 2.0).astype(int)
    merged = pd.merge(preds_df, live_df[["true_target"]], left_on="timestamp", right_index=True, how="inner")

    if merged.empty:
        return {"msg": "Waiting for ground-truth data to finalize."}

    preds = merged["signal_triggered"].astype(int)
    trues = merged["true_target"].astype(int)

    tp = int(((preds == 1) & (trues == 1)).sum())
    fp = int(((preds == 1) & (trues == 0)).sum())
    tn = int(((preds == 0) & (trues == 0)).sum())
    fn = int(((preds == 0) & (trues == 1)).sum())

    total_tracked = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_tracked if total_tracked > 0 else 0.0

    return {
        "status": "success",
        "total_tracked": total_tracked,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }

@app.get("/api/btc-price")
def btc_price(limit: int = Query(12, ge=1, le=720)):
    try:
        df = load_btc_price_series(limit)
        return {
            "timestamps": [ts.isoformat() for ts in df.index],
            "prices": [float(val) for val in df["BTC-USD_close"].tolist()],
            "volatilities": [round(float(val) * 100, 3) for val in df["BTC-USD_volatility"].tolist()],
        }
    except (FileNotFoundError, KeyError):
        return {"timestamps": [], "prices": [], "volatilities": [], "msg": "No data available yet. Waiting for first data collection cycle."}

@app.get("/api/volatility")
def volatility(limit: int = Query(1, ge=1, le=720)):
    try:
        preds_df = safe_read_csv(PREDICTIONS)
        if not preds_df.empty:
            latest = preds_df.iloc[-1]
            return {
                "prediction": str(latest["prediction"]),
                "prob_high": float(latest["prob_high"]),
                "signal_triggered": int(latest["signal_triggered"]),
                "threshold": float(latest["threshold_used"]),
                "status": "success"
            }
        res = run_prediction()
        res["status"] = "success"
        return res
    except Exception as e:
        logger.error("Prediction fallback %s", e)
        return {
            "prediction": None,
            "prob_high": None,
            "signal_triggered": None,
            "threshold": None,
            "status": "no_prediction",
        }

@app.get("/api/live_sentiment")
def live_sentiment():
    try:
        df = load_last_live_data_row()
        return {
            "ticker": "BTC",
            "sentiment_score": float(df["BTC_sentiment_score"].iloc[0]),
            "bullish_ratio": float(df["BTC_bullish_ratio"].iloc[0]),
            "bearish_ratio": float(df["BTC_bearish_ratio"].iloc[0]),
            "weighted_sentiment": float(df["BTC_weighted_sentiment"].iloc[0]),
            "delta_sentiment": float(df["BTC_delta_sentiment"].iloc[0]),
            "total_posts": int(df["total_post_count"].iloc[0]),
            "timestamp": df.index[0].isoformat(),
            "status": "success"
        }
    except (FileNotFoundError, KeyError):
        return {"status": "no_data", "msg": "No live data available yet. Waiting for first data collection cycle."}

async def scheduler():
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(minute=4, second=10, microsecond=0)
        if next_run <= now:
            next_run += timedelta(hours=1)
        await asyncio.sleep((next_run - now).total_seconds())
        try:
            backfill_vol_targets() 
            run_prediction()
        except Exception as e:
            logger.error("[scheduler] failed: %s", e)

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
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 10px;
        }
        .metric-label {
            font-size: 11px;
            color: #9aa3b2;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #f4f5f7;
        }
    </style>
</head>
<body>
    <div class="page">
        <h1>BTC Volatility Monitor</h1>
        <p>Real-time ML pipeline evaluation for high-volatility breakouts.</p>
        <div class="card">
            <div class="chart-wrap">
                <canvas id="btcChart"></canvas>
            </div>
            <div class="status" id="status"></div>
        </div>

        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Current Prediction</h2>
            <div class="status" id="volatilityStatus">Loading...</div>
            <div class="status" id="volatilityDetails"></div>
        </div>

        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Live Model Diagnostics</h2>
            
            <div class="grid-4" style="margin-bottom: 20px;">
                <div>
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value" id="trackF1">—</div>
                </div>
                <div>
                    <div class="metric-label">Precision</div>
                    <div class="metric-value" id="trackPrecision">—</div>
                </div>
                <div>
                    <div class="metric-label">Recall</div>
                    <div class="metric-value" id="trackRecall">—</div>
                </div>
                <div>
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" id="trackAccuracy">—</div>
                </div>
            </div>

            <div class="grid-4">
                <div>
                    <div class="metric-label">True Positives (Spike caught)</div>
                    <div class="metric-value" id="trackTP" style="color: #58d2a3;">0</div>
                </div>
                <div>
                    <div class="metric-label">False Positives (False Alarm)</div>
                    <div class="metric-value" id="trackFP" style="color: #ff6b6b;">0</div>
                </div>
                <div>
                    <div class="metric-label">True Negatives (Quiet Market)</div>
                    <div class="metric-value" id="trackTN" style="color: #58d2a3;">0</div>
                </div>
                <div>
                    <div class="metric-label">False Negatives (Missed Spike)</div>
                    <div class="metric-value" id="trackFN" style="color: #ff6b6b;">0</div>
                </div>
            </div>
            <div class="status" id="accHours" style="margin-top: 16px;">Hours Tracked: 0</div>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById("status");

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
            if (!res.ok) throw new Error(await res.text() || "Failed to load data");
            return res.json();
        }

        async function loadVolatility() {
            const res = await fetch("/api/volatility");
            if (!res.ok) throw new Error(await res.text() || "Failed to load volatility");
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
            const hasVol = payload.volatilities && payload.volatilities.length > 0;
            chartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: formattedLabels,
                    datasets: [{
                        label: "BTC close",
                        data: payload.prices,
                        borderColor: "#58d2a3",
                        backgroundColor: "rgba(88, 210, 163, 0.12)",
                        tension: 0.3,
                        yAxisID: "y",
                        order: 2,
                    },
                    ...(hasVol ? [{
                        label: "Volatility %",
                        data: payload.volatilities,
                        borderColor: "#ff9f43",
                        backgroundColor: "rgba(255, 159, 67, 0.08)",
                        tension: 0.3,
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        yAxisID: "y1",
                        order: 1,
                    }] : [])],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: "index", intersect: false },
                    scales: {
                        x: {
                            ticks: { color: "#b8c0cc", maxTicksLimit: 24 },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                        y: {
                            type: "linear",
                            display: true,
                            position: "left",
                            title: { display: true, text: "BTC / USD", color: "#58d2a3" },
                            ticks: { color: "#58d2a3" },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                        y1: {
                            type: "linear",
                            display: hasVol,
                            position: "right",
                            title: { display: true, text: "Volatility %", color: "#ff9f43" },
                            ticks: { color: "#ff9f43" },
                            grid: { drawOnChartArea: false },
                            ...(hasVol ? { min: 0 } : {}),
                        },
                    },
                    plugins: {
                        legend: { labels: { color: "#f4f5f7" } },
                    },
                },
            });
        }

        async function refreshAll() {
            const [btcResult, volResult, perfResult] = await Promise.allSettled([
                loadData(),
                loadVolatility(),
                (async () => {
                    try {
                        const res = await fetch("/api/performance");
                        if (!res.ok) return null;
                        return res.json();
                    } catch (e) {
                        return null;
                    }
                })(),
            ]);

            if (btcResult.status === "fulfilled") {
                renderChart(btcResult.value);
                statusEl.textContent = `Loaded ${btcResult.value.prices.length} hours.`;
            } else {
                statusEl.textContent = `Error: ${btcResult.reason?.message || "Unknown error"}`;
            }

            if (perfResult.status === "fulfilled" && perfResult.value && perfResult.value.status === "success") {
                const data = perfResult.value;
                document.getElementById("trackF1").textContent = data.f1.toFixed(3);
                document.getElementById("trackPrecision").textContent = (data.precision * 100).toFixed(1) + "%";
                document.getElementById("trackRecall").textContent = (data.recall * 100).toFixed(1) + "%";
                document.getElementById("trackAccuracy").textContent = (data.accuracy * 100).toFixed(1) + "%";
                
                document.getElementById("trackTP").textContent = data.tp;
                document.getElementById("trackFP").textContent = data.fp;
                document.getElementById("trackTN").textContent = data.tn;
                document.getElementById("trackFN").textContent = data.fn;
                
                document.getElementById("accHours").textContent = `Hours Tracked: ${data.total_tracked}`;
            }

            const vStatusEl  = document.getElementById("volatilityStatus");
            const vDetailsEl = document.getElementById("volatilityDetails");

            if (volResult.status === "rejected") {
                vStatusEl.textContent = `Error: ${volResult.reason?.message || "Unknown error"}`;
                vDetailsEl.textContent = "";
                return;
            }

            const v = volResult.value;

            if (!v || v.prediction === null || v.status === "no_prediction") {
                vStatusEl.textContent = "No prediction available yet";
                vDetailsEl.textContent = "Waiting for next hourly data...";
                return;
            }

            const predText = v.prediction.toUpperCase();
            vStatusEl.textContent = `Target Prediction: ${predText}`;
            vStatusEl.style.color = predText === "HIGH" ? "#ff6b6b" : "#58d2a3";
            const confidence = predText === "HIGH" ? v.prob_high : (1 - v.prob_high);
            vDetailsEl.textContent =
                `Probability of Spike: ${(v.prob_high * 100).toFixed(1)}% | ` +
                `Calibrated Threshold: ${v.threshold.toFixed(3)}`;
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
"""
        )