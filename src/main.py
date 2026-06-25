from fastapi import FastAPI, HTTPException, Query
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

PROFILES = {
    "Conservative": {"gain": 0.030, "cost": 0.010},
    "Standard":     {"gain": 0.060, "cost": 0.020},
    "Aggressive":   {"gain": 0.105, "cost": 0.035},
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_reddit", "live_data.csv")
PREDICTIONS = os.path.join(BASE_DIR, "data", "clean_reddit", "predictions.csv")
THRESHOLDS_JSON = os.path.join(BASE_DIR, "data", "models", "optimal_thresholds.json")

LEVERAGE = 3
FEES = 0.0005
SLIPPAGE = 0.0005
RT_COST = 2 * (FEES + SLIPPAGE) * LEVERAGE
TRADE_CAPITAL = float(os.environ.get("CRYPTOPRED_CAPITAL", "10000"))
MIN_CONFIDENCE = 0.70
MIN_POSITION_PCT = 0.50
MAX_POSITION_PCT = 1.00

ACTIVE_PROFILE = os.environ.get("CRYPTOPRED_PROFILE", "Standard")
if ACTIVE_PROFILE not in PROFILES:
    logger.warning("Unknown profile '%s' — falling back to Standard", ACTIVE_PROFILE)
    ACTIVE_PROFILE = "Standard"

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

def model_path(model_type: str) -> str:
    return os.path.join(BASE_DIR, "data", "models", f"{model_type}_{ACTIVE_PROFILE.lower()}.pkl")

def load_threshold_from_json() -> dict[str, float]:
    default = {"lgb": 0.50}
    if not os.path.exists(THRESHOLDS_JSON):
        logger.warning("%s not found. Using default 0.50.", THRESHOLDS_JSON)
        return default
    try:
        with open(THRESHOLDS_JSON, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Could not parse %s. Using default 0.50.", THRESHOLDS_JSON)
        return default

    profile_data = data.get(ACTIVE_PROFILE, {})
    if not profile_data:
        logger.warning("Profile '%s' not found in JSON. Using default 0.50.", ACTIVE_PROFILE)
        return default
    return profile_data

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
        pred_row.to_csv(PREDICTIONS, index=False)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    new_ts = pd.to_datetime(pred_row["timestamp"].iloc[0], utc=True)

    if new_ts not in df["timestamp"].values:
        pred_row.to_csv(PREDICTIONS, mode="a", header=False, index=False)

def compute_prediction() -> dict:
    lgb_path = model_path("lgb")
    if not os.path.exists(lgb_path):
        raise RuntimeError(f"Model file not found: {lgb_path}")

    row = load_last_live_data_row()
    to_drop_cols = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target"]
    features = row.drop(columns=to_drop_cols, errors="ignore")

    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        logger.warning("Feature columns with NaN values (filling with 0): %s", nan_cols)

    features = features.fillna(0)
    lgb = pickle.load(open(lgb_path, "rb"))

    if hasattr(lgb, "feature_names_in_"):
        features = features[lgb.feature_names_in_]
    elif hasattr(lgb, "feature_name_"):
        features = features[lgb.feature_name_]

    _, lgb_high = lgb.predict_proba(features)[0]
    thresholds = load_threshold_from_json()
    current_threshold = thresholds.get("lgb", 0.50)
    signal = 1 if lgb_high > current_threshold else 0

    traded = signal == 1 and lgb_high >= MIN_CONFIDENCE
    if traded:
        fraction = (lgb_high - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)
        position_pct = MIN_POSITION_PCT + fraction * (MAX_POSITION_PCT - MIN_POSITION_PCT)
        capital_at_risk = TRADE_CAPITAL * position_pct
    else:
        position_pct = 0.0
        capital_at_risk = 0.0

    return {
        "prediction": "high" if signal == 1 else "low",
        "prob_high": lgb_high,
        "signal_triggered": signal,
        "traded": traded,
        "position_pct": position_pct,
        "capital_at_risk": capital_at_risk,
        "profile": ACTIVE_PROFILE,
        "threshold": current_threshold,
        "timestamp": pd.to_datetime(row.index[0]).isoformat(),
    }

def run_prediction() -> dict:
    result = compute_prediction()
    pred_row = pd.DataFrame([{
        "timestamp": result["timestamp"],
        "prediction": result["prediction"],
        "prob_high": result["prob_high"],
        "signal_triggered": result["signal_triggered"],
        "traded": result["traded"],
        "position_pct": result["position_pct"],
        "capital_at_risk": result["capital_at_risk"],
        "profile_used": ACTIVE_PROFILE,
        "threshold_used": result["threshold"],
    }])
    write_prediction(pred_row)
    return result

def prediction() -> dict:
    lgb_path = model_path("lgb")
    if not os.path.exists(lgb_path):
        raise RuntimeError(f"Model file not found: {lgb_path}")

    row = load_last_live_data_row()
    to_drop_cols = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target"]
    features = row.drop(columns=to_drop_cols, errors="ignore")

    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        logger.warning("Feature columns with NaN values (filling with 0): %s", nan_cols)

    features = features.fillna(0)

    lgb = pickle.load(open(lgb_path, "rb"))
    if hasattr(lgb, "feature_names_in_"):
        features = features[lgb.feature_names_in_]
    elif hasattr(lgb, "feature_name_"):
        features = features[lgb.feature_name_]
    _, lgb_high = lgb.predict_proba(features)[0]

    thresholds = load_threshold_from_json()
    current_threshold = thresholds.get("lgb", 0.50)

    signal = 1 if lgb_high > current_threshold else 0
    prediction = "high" if signal == 1 else "low"

    return {
        "prediction": prediction,
        "prob_high": lgb_high,
        "signal_triggered": signal,
        "profile": ACTIVE_PROFILE,
        "threshold": current_threshold,
    }

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

    if not preds_df.empty and "profile_used" in preds_df.columns:
        preds_df = preds_df[preds_df["profile_used"] == ACTIVE_PROFILE]

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

    trades = merged[merged["signal_triggered"] == 1]
    no_trades = merged[merged["signal_triggered"] == 0]
    total_trades = len(trades)

    tp = int(np.sum((trades["true_target"] == 1))) if total_trades > 0 else 0
    fp = int(np.sum((trades["true_target"] == 0))) if total_trades > 0 else 0
    fn = int(np.sum((no_trades["true_target"] == 1)))
    tn = int(np.sum((no_trades["true_target"] == 0)))

    total_hours = tp + fp + fn + tn
    overall_accuracy = (tp + tn) / total_hours if total_hours > 0 else 0.0

    cfg = PROFILES[ACTIVE_PROFILE]
    payoff = tp * cfg["gain"] - fp * cfg["cost"] - (tp + fp) * RT_COST if total_trades > 0 else 0.0
    win_rate = tp / total_trades if total_trades > 0 else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "cumulative_payoff": float(payoff) * TRADE_CAPITAL,
        "trade_capital": TRADE_CAPITAL,
        "sample_size": total_hours,
        "missed_swings": int(fn),
        "correct_low_calls": int(tn),
        "overall_accuracy": float(overall_accuracy),
    }

@app.get("/api/btc-price")
def btc_price(limit: int = Query(12, ge=1, le=720)):
    df = load_btc_price_series(limit)
    return {
            "timestamps": [ts.isoformat() for ts in df.index],
            "prices": [float(val) for val in df["BTC-USD_close"].tolist()],
    }

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
                "profile": str(latest["profile_used"]),
                "threshold": float(latest["threshold_used"]),
                "status": "success"
            }
        res = run_prediction()  # ← was compute_prediction
        res["status"] = "success"
        return res
    except Exception as e:
        logger.error("Prediction fallback %s", e)
        return {
            "prediction": None,
            "prob_high": None,
            "signal_triggered": None,
            "profile": ACTIVE_PROFILE,
            "threshold": None,
            "status": "no_prediction",
        }

async def scheduler():
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(minute=2, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(hours=1)
        await asyncio.sleep((next_run - now).total_seconds())
        try:
            backfill_vol_targets()  # ← backfill first
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
        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Live Strategy Tracker</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 10px;">
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Trades Taken</div>
                    <div id="trackTrades" style="font-size: 20px; font-weight: bold; color: #f4f5f7;">0</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Realized Win Rate</div>
                    <div id="trackWinRate" style="font-size: 20px; font-weight: bold; color: #58d2a3;">0.0%</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Strategy Payoff</div>
                    <div id="trackPayoff" style="font-size: 20px; font-weight: bold; color: #f4f5f7;">+$0.00</div>
                </div>
            </div>
        </div>
        <div class="card" style="margin-top: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 20px;">Model Accuracy Tracker</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px;">
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Overall Accuracy</div>
                    <div id="accOverall" style="font-size: 20px; font-weight: bold; color: #f4f5f7;">—</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Missed Swings</div>
                    <div id="accMissed" style="font-size: 20px; font-weight: bold; color: #58d2a3;">0</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Correct "Low" Calls</div>
                    <div id="accLow" style="font-size: 20px; font-weight: bold; color: #58d2a3;">0</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #9aa3b2; text-transform: uppercase;">Hours Tracked</div>
                    <div id="accHours" style="font-size: 20px; font-weight: bold; color: #f4f5f7;">0</div>
                </div>
            </div>
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

        async function loadPerformance() {
            try {
                const res = await fetch("/api/performance");
                if (!res.ok) return;
                const data = await res.json();

                if (data.total_trades !== undefined) {
                    document.getElementById("trackTrades").textContent = data.total_trades;
                    document.getElementById("trackWinRate").textContent = `${(data.win_rate * 100).toFixed(1)}%`;

                    const payoffEl = document.getElementById("trackPayoff");
                    payoffEl.textContent = `${data.cumulative_payoff >= 0 ? '+' : ''}$${data.cumulative_payoff.toFixed(2)}`;
                    payoffEl.style.color = data.cumulative_payoff >= 0 ? "#58d2a3" : "#ff6b6b";
                }

                if (data.overall_accuracy !== undefined) {
                    const accPct = (data.overall_accuracy * 100).toFixed(1);
                    const accEl = document.getElementById("accOverall");
                    accEl.textContent = `${accPct}%`;
                    if (data.overall_accuracy >= 0.55) {
                        accEl.style.color = "#58d2a3";
                    } else if (data.overall_accuracy >= 0.50) {
                        accEl.style.color = "#f0c040";
                    } else {
                        accEl.style.color = "#ff6b6b";
                    }

                    const missedEl = document.getElementById("accMissed");
                    missedEl.textContent = data.missed_swings;
                    missedEl.style.color = data.missed_swings > 0 ? "#ff6b6b" : "#58d2a3";

                    document.getElementById("accLow").textContent = data.correct_low_calls;
                    document.getElementById("accHours").textContent = data.sample_size;
                }
            } catch (err) {
                console.error("Performance tracking error:", err);
            }
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
                    datasets: [{
                        label: "BTC close",
                        data: payload.prices,
                        borderColor: "#58d2a3",
                        backgroundColor: "rgba(88, 210, 163, 0.12)",
                        tension: 0.3,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: { color: "#b8c0cc", maxTicksLimit: 24 },
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

            // Volatility prediction — single call, no early return
            try {
                const v = await loadVolatility();
                const vStatusEl  = document.getElementById("volatilityStatus");
                const vDetailsEl = document.getElementById("volatilityDetails");

                if (!v || v.prediction === null || v.status === "no_prediction") {
                    vStatusEl.textContent = "No prediction available yet";
                    vDetailsEl.textContent = "Waiting for next hourly data...";
                } else {
                    const predText = v.prediction.toUpperCase();
                    vStatusEl.textContent = `Prediction: ${predText}`;
                    vStatusEl.style.color = predText === "HIGH" ? "#ff6b6b" : "#58d2a3";
                    const confidence = predText === "HIGH" ? v.prob_high : (1 - v.prob_high);
                    vDetailsEl.textContent =
                        `Confidence: ${(confidence * 100).toFixed(1)}% | ` +
                        `Threshold: ${v.threshold} | Profile: ${v.profile}`;
                }
            } catch (err) {
                document.getElementById("volatilityStatus").textContent = `Error: ${err.message}`;
            }
            await loadPerformance();
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