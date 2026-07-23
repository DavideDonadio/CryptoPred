# CryptoPred — Real-Time Crypto Volatility Forecasting

CryptoPred is an automated end-to-end machine learning pipeline and FastAPI application that predicts next-hour cryptocurrency volatility spikes by fusing real-time social media sentiment (Reddit) with live market data (Binance).

> 💡 **Detailed Methodology & Notebook:** For full cross-validation breakdowns, multi-class vs. binary trade-off analyses, hyperparameter tuning, threshold calibration, and regime-dependent limitations, please refer to the [`summary.ipynb`](./summary.ipynb) notebook.

---

## 🚀 Key Highlights

* **Sentiment Engine:** SVM model trained on StockTwits Crypto data using TF-IDF n-grams, outperforming specialized transformers like [CryptoBert](https://personal.eur.nl/frasincar/papers/ACSA2023/acsa2023.pdf) at a fraction of the computational cost.
* **Market Pipeline:** Ingests and aggregates hourly OHLCV data across BTC, ETH, and SOL alongside Reddit activity across 6 subreddits, generating sentiment ratios, momentum deltas, cyclical time encodings, and multi-interval lagged returns / volatility.
* **Predictive Task:** Binary extreme event detection targeting top-quantile volatility spikes using LightGBM and XGBoost models.
* **Final Test Performance (Held-Out Test Set):**

  * **XGBoost:** F1-score: **0.63** | Precision: **0.58** | Recall: **0.70** | PR-AUC: **0.717**
  * **LightGBM:** F1-score: **0.62** | Precision: **0.55** | Recall: **0.70** | PR-AUC: **0.725**

---

## 📦 Quickstart (Docker Deployment)

The pre-built Docker container bundles the FastAPI web application, trained models (LightGBM, XGBoost, and SVM), the Reddit scraper, and an internal hourly cron scheduler. It is fully self-contained—no external dependencies are required.

```bash
# Pull the pre-built image from GitHub Container Registry
docker pull ghcr.io/davidedonadio/cryptopred:latest

# Run the container (Dashboard available at http://localhost:8000)
docker run -d --name cryptopred \
  -p 8000:8000 \
  -v "$(pwd)/API_App/data/clean_reddit:/app/data/clean_reddit:Z" \
  ghcr.io/davidedonadio/cryptopred:latest
```

*To build locally from source:*
```bash
docker build -t ghcr.io/davidedonadio/cryptopred:latest API_App/
```

### Automation & Scheduling
Once running, the container manages continuous hourly inference:
* **`cron`**: Triggers `pipeline.sh` at the top of each hour to scrape new Reddit data and construct updated features.
* **`uvicorn`**: Serves refreshed predictions and rolling performance diagnostics via FastAPI at minute `:04` past each hour.

---

## 📡 API Endpoints Reference

| Endpoint | Description |
|---|---|
| `GET /` | Interactive dashboard with BTC price chart, volatility, sentiment, predictions, and live diagnostics|
| `GET /api/volatility` | Current prediction (high/low), probability, and decision threshold |
| `GET /api/btc-price?limit=N` | Recent BTC closing prices, hourly volatility %, and prediction signal markers |
| `GET /api/performance` | Rolling confusion matrix and classification metrics (F1, Precision, Recall, Accuracy) |
| `GET /api/historical-audit?limit=N` | Audit trail combining past predictions, probabilities, and actual volatility spike evaluations |
| `GET /api/live_sentiment?limit=N` | Historical BTC sentiment scores, bullish/bearish ratios, momentum, post counts, and volatility % |
| `GET /api/predictions?limit=N` | Historical prediction probabilities with signal, threshold, and outcome |