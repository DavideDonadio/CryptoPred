# CryptoPred — Real-Time Crypto Volatility Forecasting with Social Sentiment

CryptoPred is a fully automated machine learning pipeline that predicts next-hour cryptocurrency volatility by fusing real-time social media sentiment with live market data. The system ingests posts from six major crypto-focused subreddits — r/Bitcoin, r/Ethereum, r/Solana, r/CryptoCurrency, r/CryptoCurrencies, and r/Crypto_Currency_News — and processes them through a trained Support Vector Machine sentiment classifier that labels each post as bearish, neutral, or bullish. These sentiment signals are aggregated into hourly windows alongside price data fetched from Binance via the CCXT exchange library, producing a rich feature set that includes sentiment ratios, momentum deltas, cyclical time encodings, and multi-interval lagged returns and volatility. The engineered features feed an optimized XGBoost binary classifier that outputs a probabilistic forecast of whether the upcoming hour will experience a high-volatility spike. This prediction is served through a FastAPI web application complete with an interactive dashboard displaying live BTC price charts, current prediction status, and rolling performance diagnostics including precision, recall, F1-score, and confusion matrix breakdowns. The entire application is containerized via Docker, enabling consistent deployment across any environment and continuous hourly inference through an internal scheduler that fetches new Reddit data, recomputes features, and stores predictions for retrospective evaluation.

## Dataset Description

The project leverages two primary data sources combined into a unified training dataset spanning the full calendar year 2022:

**Social Sentiment Data:** A Kaggle dataset containing all Reddit posts from dozens of cryptocurrency subreddits throughout 2022. From this, six subreddits are selected — three coin-specific (Bitcoin, Ethereum, Solana) and three general-purpose (CryptoCurrency, CryptoCurrencies, Crypto_Currency_News). Each post's title and body are cleaned, concatenated, and passed through the SVM sentiment classifier to produce a labeled sentiment score per post. The sentiment model itself was trained and evaluated on the StockTwits Crypto dataset (ElKulako/stocktwits-crypto on HuggingFace), which contains 1.3 million labeled financial social media posts. After benchmark comparisons against VADER, Random Forest, Gemma 2B, and CryptoBERT, the SVM was selected as the most accurate and computationally efficient solution, achieving 59.71% accuracy and a 0.5916 macro F1-score on the 83k-row test set, outperforming the specialized transformer by over 4 percentage points.

**Market Data:** Hourly OHLCV (Open, High, Low, Close, Volume) data for BTC/USDT, ETH/USDT, and SOL/USDT is fetched from Binance via the CCXT library. This raw data is transformed into percentage returns, volatility estimates (high-low spread divided by open), and log-transformed volume metrics, then joined with the aggregated sentiment features on an hourly index to form the complete feature matrix.

The target variable is a binary volatility classification derived by splitting hourly volatility into quantiles. The bottom two-thirds of volatility observations are labeled as "Normal Market" (Class 0), while the top third constitutes a "Volatility Spike" (Class 1). This binary restructuring was chosen after initial multiclass experiments revealed that the boundary between low and medium volatility was too noisy for reliable classification, causing the neutral class to collapse with F1-scores around 0.30. The binary approach eliminated this boundary noise and sharpened the predictive signal.

## Commands to Run

### API Application (Docker)

The Docker image bundles the FastAPI server, trained models (LightGBM + XGBoost volatility classifiers and SVM sentiment classifier), the Reddit scraper, and an internal cron scheduler.<br>
It is fully self-contained, no external dependencies are needed.

```bash
# Pull the pre-built image from GitHub Container Registry
docker pull ghcr.io/davidedonadio/cryptopred:latest

# Run the container (dashboard available at http://localhost:8000)
docker run -d -p 8000:8000 -v "$(pwd)/API_App/data/clean_reddit:/app/data/clean_reddit" ghcr.io/davidedonadio/cryptopred:latest

# Or build locally from source
docker build -t ghcr.io/davidedonadio/cryptopred:latest API_App/
```

The container starts two processes:
- **cron** triggers `pipeline.sh` at the top of each hour to scrape Reddit and build live features
- **uvicorn** serves the FastAPI app with predictions updated at minute 4 past each hour

### API Endpoints
| Endpoint | Description |
|---|---|
| `GET /` | Interactive dashboard with BTC price chart and live diagnostics |
| `GET /api/volatility`     | Current prediction (high/low), probability, and threshold |
| `GET /api/btc-price`      | Recent BTC closing prices |
| `GET /api/performance`    | Rolling confusion matrix and classification metrics |
| `GET /api/live_sentiment` | Latest hour's aggregated BTC sentiment score, ratios, and momentum |


## Project Results

The project progressed through four distinct analytical phases, each yielding measurable improvements in predictive performance.

### Phase 1 — Sentiment Model Selection

Four sentiment analysis approaches were benchmarked against CryptoBERT on the StockTwits test set. VADER achieved only 37.34% accuracy, confirming that lexicon-based approaches fail on crypto-specific slang. Random Forest and Gemma 2B were eliminated early due to prohibitive computational cost at scale. The SVM trained on TF-IDF n-grams (1-5) reached 59.71% accuracy with a 0.5916 macro F1-score, surpassing CryptoBERT's 55.60% accuracy and 0.5579 F1-score. The SVM was selected for deployment due to its superior accuracy, F1-score and significantly lower inference cost.

### Phase 2 — Multiclass Volatility Forecasting

Three tree-based models (XGBoost, Random Forest, and LightGBM) were trained on the full 3-class volatility target using TimeSeriesSplit cross-validation (`n_splits=5`, gap=24h) to prevent temporal leakage.

Averaging across folds, the multiclass formulation revealed a strong class imbalance in predictive difficulty:

- **Class 0 (Low Volatility): mean F1 ≈ 0.64** → Quiet, range-bound regimes were the most stable and predictable.

- **Class 2 (High Volatility): mean F1 ≈ 0.53** → Extreme volatility events showed measurable predictability, likely driven by spikes in social activity and lagged market volatility features.

- **Class 1 (Medium Volatility): mean F1 ≈ 0.35** → This class consistently underperformed, indicating that the boundary between low and medium regimes is inherently noisy and weakly separable.

Overall accuracy ranged between **0.47 and 0.76** across folds. Random Forest provided the strongest overall performance, achieving the highest average F1-scores for both Class 0 and Class 2, though XGBoost remained highly competitive for extreme events. LightGBM showed comparable average performance but higher variance across folds.

Feature importance analysis consistently identified market-structure variables as dominant predictors. `BTC-USD` volatility and its lagged values (1h, 3h, 12h) were the strongest features across all models, followed by ETH and SOL volatility lags, total post volume, and post spike ratio. The cyclical feature `hour_cos` was also consistently relevant, suggesting an intraday structure in volatility dynamics. Sentiment-derived features ranked lower overall, indicating weaker standalone predictive power.

---

## Phase 3 — Binary Restructuring

To improve signal separability and reduce label ambiguity following multiclass forecasting, the problem was reformulated as an **extreme event detection task**. The target variable was redefined to exclusively identify high-volatility regimes:

* **Class 0 + Class 1** → Normal Market Conditions
* **Class 2** → Volatility Spike

### Impact on Learning Dynamics

Simplifying the decision boundary allowed gradient boosting models to optimize directly for separating normal from extreme regimes. Model configurations were adjusted to handle class imbalances (e.g., `scale_pos_weight` for XGBoost, `class_weight="balanced"` for LightGBM/Random Forest).

This resulted in more stable probability outputs, improved recall for volatility spikes, and reduced sensitivity to label noise.

### Resulting Model Performance

The binary setup yielded clear improvements in separability and stability across folds:

* **XGBoost:** Positive-class F1-scores ranged from **0.36 to 0.73**, with the best folds reaching **0.63–0.73** F1 and overall accuracy peaking at **0.91** in highly separable regimes.
* **Random Forest:** Maintained balanced precision-recall trade-offs, with positive-class F1 peaking at **0.73**.
* **LightGBM:** Showed slightly higher variance but remained highly competitive, posting positive-class F1 values between **0.47 and 0.71** with strong recall in mid-regime splits.

Feature importance remained consistent with Phase 2 with `BTC-USD_volatility` and its lagged variants still dominating predictive power across all models.

### AutoML Benchmark Comparison

An AutoGluon stacking framework (combining LightGBM, CatBoost, Random Forest, and Extra Trees) was introduced to test if ensemble complexity could further improve metrics.

While the AutoML system was highly competitive, it did not consistently outperform individually tuned gradient boosting models:
* **Accuracy** ranged from **0.74 to 0.90** across folds.
* **Macro F1** remained broadly comparable to the best single-model results.
* The best-performing single models (XGBoost and LightGBM) frequently matched or slightly exceeded the ensemble's performance.

This indicates that increasing ensemble complexity offers diminishing returns. The predictive signal is primarily driven by feature representation and temporal structure, and gradient-boosted decision trees already capture the majority of the exploitable signal.

### Overall Interpretation

The binary reformulation successfully shifted the problem from ordinal regime classification to rare event detection. This yielded:
* More stable cross-validation performance with reduced variance.
* Substantially higher spike recall (up to **~0.93** in optimal regimes).
* Consistent separation between normal and extreme market conditions.
* Confirmation that highly tuned base boosting models are sufficient, with no substantial gains required from deep AutoML stacking.

### Phase 4 — Hyperparameter Tuning, Threshold Optimization, and Final Evaluation

Grid search was performed using 5-fold time-series cross-validation, optimizing LightGBM and XGBoost hyperparameters with **average precision (PR-AUC)** as the primary scoring metric.

Following training, decision thresholds were optimized independently for each model using a dedicated **15% calibration set**, by scanning 80 probability thresholds and selecting the value maximizing F1-score. Final evaluation was conducted on a strictly held-out **15% test set**, ensuring that neither model fitting nor threshold selection had access to test information.

### Final Results (Held-Out Test Set)

#### LightGBM
- Test F1-score (positive class): **0.62**
- Precision: **0.55**
- Recall: **0.70**
- Optimal threshold: **0.495**
- Calibration F1-score: **0.490**
- PR-AUC (test): **0.725**
- Signals: 241 (TP = 133, FP = 108)

#### XGBoost
- Test F1-score (positive class): **0.63**
- Precision: **0.58**
- Recall: **0.70**
- Optimal threshold: **0.495**
- Calibration F1-score: **0.481**
- PR-AUC (test): **0.717**
- Signals: 229 (TP = 132, FP = 97)

Both models converge to a similar operating region characterized by high recall (~0.70) and moderate precision (~0.56), indicating effective detection of volatility spikes with a controlled false positive rate. The similarity in optimal thresholds (~0.48–0.50) suggests broadly consistent probability calibration across both frameworks.

---

### Limitations of the Evaluation

While the evaluation protocol is strictly time-aware and avoids data leakage, several limitations must be acknowledged.

First, the final evaluation is conducted on a single contiguous 15% test segment of the 2022 dataset. Due to the non-stationary nature of cryptocurrency markets, model performance may be influenced by the specific volatility regime present in this period. As a result, the reported metrics should be interpreted as conditional on this market regime rather than fully representative of all possible market conditions.

Second, the evaluation does not explicitly quantify robustness across multiple market regimes. A single hold-out window cannot capture variability in performance across different volatility environments (e.g., crisis periods vs. stable accumulation phases).

Third, threshold optimization is performed on a separate calibration set, which ensures out-of-sample selection but introduces sensitivity to the distribution of that specific period. If probability calibration shifts over time, optimal thresholds may also change in production.

Overall, while the methodology is sound for time-series forecasting and avoids leakage, the results reflect regime-dependent performance rather than fully regime-invariant generalization.