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
        if not os.path.exists(PREDICTIONS) or os.path.getsize(PREDICTIONS) == 0:
            backfill_predictions()
        logger.info("[startup] initial backfill complete")
    except Exception as e:
        logger.error("[startup] backfill failed: %s", e)
    asyncio.create_task(scheduler())
    yield

def backfill_predictions() -> None:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LIVE_DATA_PATH):
        logger.warning("[backfill] missing model or data, skipping")
        return
    try:
        df = pd.read_csv(LIVE_DATA_PATH, index_col=0)
        if df.empty:
            return
        df.index = pd.to_datetime(df.index, utc=True)
        model = pickle.load(open(MODEL_PATH, "rb"))
        threshold = load_threshold_from_json()
        to_drop = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target"]
        features = df.drop(columns=to_drop, errors="ignore").fillna(0)
        if hasattr(model, "feature_names_in_"):
            features = features[model.feature_names_in_]
        probs = model.predict_proba(features)
        rows = []
        for i, idx in enumerate(df.index):
            high = probs[i][1]
            pred = "high" if high >= threshold else "low"
            signal = 1 if high >= threshold else 0
            rows.append({
                "timestamp": idx,
                "prediction": pred,
                "prob_high": round(high, 7),
                "signal_triggered": signal,
                "threshold_used": threshold,
            })
            
        pd.DataFrame(rows).to_csv(PREDICTIONS, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
        logger.info("[backfill] generated %d predictions", len(rows))
    except Exception as e:
        logger.error("[backfill] failed: %s", e)

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
    
    if not (df["timestamp"] == new_ts).any():
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
def btc_price(limit: int = Query(24, ge=1, le=720)):
    try:
        df = load_btc_price_series(limit)
        preds_df = safe_read_csv(PREDICTIONS)
        signal_map = {}
        if not preds_df.empty:
            preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)
            for _, row in preds_df.iterrows():
                signal_map[row["timestamp"] + pd.Timedelta(hours=1)] = int(row["signal_triggered"])
        return {
            "timestamps": [ts.isoformat() for ts in df.index],
            "prices": [float(val) for val in df["BTC-USD_close"].tolist()],
            "volatilities": [round(float(val) * 100, 3) for val in df["BTC-USD_volatility"].tolist()],
            "signals": [signal_map.get(ts, None) for ts in df.index],
        }
    except (FileNotFoundError, KeyError):
        return {"timestamps": [], "prices": [], "volatilities": [], "signals": [], "msg": "No data available yet."}

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
def live_sentiment(limit: int = Query(24, ge=1, le=24)):
    try:
        df = _load_live_data(limit)
        history = []
        for idx, row in df.iterrows():
            history.append({
                "timestamp": idx.isoformat(),
                "sentiment_score": round(float(row.get("BTC_sentiment_score", 0)), 4),
                "bullish_ratio": round(float(row.get("BTC_bullish_ratio", 0)), 4),
                "bearish_ratio": round(float(row.get("BTC_bearish_ratio", 0)), 4),
                "delta_sentiment": round(float(row.get("BTC_delta_sentiment", 0)), 4),
                "total_posts": int(row.get("total_post_count", 0)),
                "volatility_pct": round(float(row.get("BTC-USD_volatility", 0)) * 100, 4),
            })
        return {"ticker": "BTC", "status": "success", "history": history}
    except (FileNotFoundError, KeyError):
        return {"status": "no_data", "msg": "No live data available yet.", "history": []}

@app.get("/api/predictions")
def predictions(limit: int = Query(24, ge=1, le=168)):
    try:
        preds_df = safe_read_csv(PREDICTIONS)
        if preds_df.empty:
            return {"status": "no_data", "msg": "No predictions yet.", "history": []}
        preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)
        preds_df = preds_df.sort_values("timestamp").tail(limit)
        history = []
        for _, row in preds_df.iterrows():
            history.append({
                "timestamp": row["timestamp"].isoformat(),
                "prob_high": round(float(row["prob_high"]), 4),
                "prediction": str(row["prediction"]),
                "signal_triggered": int(row["signal_triggered"]),
                "threshold": round(float(row["threshold_used"]), 4),
            })
        return {"status": "success", "history": history}
    except Exception:
        return {"status": "no_data", "msg": "No predictions yet.", "history": []}

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
            <h2 style="margin: 0 0 8px; font-size: 20px;">Volatility Spike Probability</h2>
            <div class="chart-wrap" style="height: 220px;">
                <canvas id="probChart"></canvas>
            </div>
            <div class="status" id="probStatus"></div>
        </div>

        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Community Sentiment (Hourly)</h2>
            <div class="chart-wrap" style="height: 250px;">
                <canvas id="sentimentChart"></canvas>
            </div>
            <div class="status" id="sentimentStatus"></div>
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

        async function loadSentiment() {
            const r = await fetch("/api/live_sentiment?limit=24");
            if (!r.ok) return {history:[]};
            return r.json();
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
        let sentimentChartInstance = null;
        let probChartInstance = null;

        async function loadPredictions() {
            const r = await fetch("/api/predictions?limit=24");
            if (!r.ok) return {history:[]};
            return r.json();
        }

        function renderProbChart(data) {
            const probStatus = document.getElementById("probStatus");
            if (!data.history || data.history.length === 0) {
                probStatus.textContent = "No prediction data yet.";
                return;
            }
            const labels = data.history.map(h => formatTimestamp(h.timestamp));
            const probs = data.history.map(h => h.prob_high);
            const threshold = data.history[0]?.threshold || 0.495;
            const ctx = document.getElementById("probChart").getContext("2d");
            if (probChartInstance) {
                probChartInstance.data.labels = labels;
                probChartInstance.data.datasets[0].data = probs;
                probChartInstance.data.datasets[1].data = Array(labels.length).fill(threshold);
                probChartInstance.update();
                return;
            }
            probChartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [
                        { label: "P(High Volatility)", data: probs, borderColor: "#f97316", backgroundColor: "rgba(249,115,22,0.08)", tension: 0.3, fill: true, pointRadius: 2 },
                        { label: `Threshold (${(threshold*100).toFixed(1)}%)`, data: Array(labels.length).fill(threshold), borderColor: "rgba(255,255,255,0.25)", borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
                    ],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: "#f4f5f7" } } },
                    scales: {
                        x: { ticks: { color: "#b8c0cc", maxTicksLimit: 24 }, grid: { color: "rgba(255,255,255,0.05)" } },
                        y: { min: 0, max: 1, ticks: { color: "#f97316", callback: v => (v*100).toFixed(0)+"%" }, grid: { color: "rgba(255,255,255,0.05)" }, title: { display: true, text: "Probability", color: "#f97316" } },
                    },
                },
            });
            probStatus.textContent = "Loaded " + data.history.length + " predictions.";
        }

        function renderChart(payload) {
            const formattedLabels = payload.timestamps.map(formatTimestamp);
            const ctx = document.getElementById("btcChart").getContext("2d");
            
            if (chartInstance) {
                chartInstance.data.labels = formattedLabels;
                chartInstance.data.datasets[0].data = payload.prices;
                if (payload.volatilities && chartInstance.data.datasets[1]) {
                    chartInstance.data.datasets[1].data = payload.volatilities;
                }
                chartInstance.update();
                return;
            }

            const hasVol = payload.volatilities && payload.volatilities.length > 0;
            const hasSignals = payload.signals && payload.signals.length > 0;
            const highPoints = hasSignals ? payload.prices.map((p,i) => payload.signals[i]===1 ? p : null) : [];
            
            const datasets = [
                {
                    label: "BTC close",
                    data: payload.prices,
                    borderColor: "#58d2a3",
                    backgroundColor: "rgba(88, 210, 163, 0.12)",
                    tension: 0.3,
                    yAxisID: "y",
                    order: 2,
                }
            ];

            if (hasVol) {
                datasets.push({
                    label: "Volatility %",
                    data: payload.volatilities,
                    borderColor: "#ff9f43",
                    backgroundColor: "rgba(255, 159, 67, 0.08)",
                    tension: 0.3,
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    yAxisID: "y1",
                    order: 1,
                });
            }

            if (hasSignals) {
                datasets.push({
                    label: "HIGH Prediction", 
                    data: highPoints,
                    borderColor: "#ff6b6b", 
                    backgroundColor: "#ff6b6b",
                    pointRadius: 5, 
                    pointHoverRadius: 7, 
                    pointStyle: "rectRounded",
                    showLine: false, 
                    yAxisID: "y", 
                    order: 0,
                });
            }

            // High Volatility Baseline Target Line
            datasets.push({
                label: "High Vol. Threshold (1.069%)",
                data: Array(formattedLabels.length).fill(1.069),
                borderColor: "rgba(255,107,107,0.5)", 
                borderWidth: 1, 
                borderDash: [6,4],
                pointRadius: 0, 
                fill: false, 
                yAxisID: "y1", 
                order: 3,
            });

            chartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: formattedLabels,
                    datasets: datasets
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
                            min: 0,
                        },
                    },
                    plugins: {
                        legend: { labels: { color: "#f4f5f7" } },
                    },
                },
            });
        }

        function renderSentimentChart(data) {
            const sentimentStatus = document.getElementById("sentimentStatus");
            const sentimentChartCanvas = document.getElementById("sentimentChart");
            if (!sentimentStatus || !sentimentChartCanvas) return; // Guard rails if sentiment DOM card is hidden

            if (!data.history || data.history.length === 0) {
                sentimentStatus.textContent = "No sentiment data yet.";
                return;
            }
            const labels = data.history.map(h => formatTimestamp(h.timestamp));
            const scores = data.history.map(h => h.sentiment_score);
            const vols = data.history.map(h => h.volatility_pct || 0);
            const ctx = sentimentChartCanvas.getContext("2d");
            
            if (sentimentChartInstance) {
                sentimentChartInstance.data.labels = labels;
                sentimentChartInstance.data.datasets[0].data = scores;
                sentimentChartInstance.data.datasets[1].data = vols;
                sentimentChartInstance.update();
                return;
            }
            sentimentChartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [
                        { 
                            label: "BTC Sentiment Score", 
                            data: scores,
                            borderColor: "#a78bfa", 
                            backgroundColor: "rgba(167,139,250,0.12)",
                            tension: 0.3, 
                            fill: true, 
                            pointRadius: 2,
                            yAxisID: "y",
                            order: 2,
                        },
                        {
                            label: "Volatility %",
                            data: vols,
                            borderColor: "#ff9f43",
                            backgroundColor: "rgba(255, 159, 67, 0.05)",
                            tension: 0.3,
                            borderWidth: 1.5,
                            borderDash: [4, 4],
                            pointRadius: 2,
                            yAxisID: "y1",
                            order: 1,
                        }
                    ],
                },
                options: {
                    responsive: true, 
                    maintainAspectRatio: false,
                    interaction: { mode: "index", intersect: false },
                    plugins: { legend: { labels: { color: "#f4f5f7" } } },
                    scales: {
                        x: { ticks: { color: "#b8c0cc", maxTicksLimit: 24 }, grid: { color: "rgba(255,255,255,0.05)" } },
                        y: { 
                            type: "linear",
                            display: true,
                            position: "left",
                            ticks: { color: "#a78bfa" }, 
                            grid: { color: "rgba(255,255,255,0.05)" },
                            title: { display: true, text: "Sentiment", color: "#a78bfa" } 
                        },
                        y1: {
                            type: "linear",
                            display: true,
                            position: "right",
                            ticks: { color: "#ff9f43" },
                            grid: { drawOnChartArea: false },
                            title: { display: true, text: "Volatility %", color: "#ff9f43" },
                            min: 0,
                        },
                    },
                },
            });
            sentimentStatus.textContent = "Loaded " + data.history.length + " hours.";
        }

        async function refreshAll() {
            const [btcResult, volResult, perfResult, sentResult, probResult] = await Promise.allSettled([
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
                loadSentiment(),
                loadPredictions(),
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

            if (sentResult.status === "fulfilled") {
                renderSentimentChart(sentResult.value);
            }

            if (probResult.status === "fulfilled") {
                renderProbChart(probResult.value);
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