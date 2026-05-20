import os
import time
import re
import pickle
import ccxt
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

def clean_reddit_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('&#x200B;', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_market_data(df, ticker):
    df = df.rename(columns={"Close": f"{ticker}_close", "Volume": f"{ticker}_volume"})
    df[f"{ticker}_return"] = df[f"{ticker}_close"].pct_change()
    df[f"{ticker}_volatility"] = (df["High"] - df["Low"]) / df["Open"]
    df[f"{ticker}_vol_log"] = np.log1p(df[f"{ticker}_volume"])
    return df[[f"{ticker}_close", f"{ticker}_return", f"{ticker}_volatility", f"{ticker}_vol_log"]]

def fetch_ccxt(symbol, start_dt, end_dt, timeframe='1h'):
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start = exchange.parse8601(start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    end = exchange.parse8601(end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    all_kline = []
    
    while start < end:
        try:
            klines = exchange.fetch_ohlcv(symbol, timeframe, start, limit=1000)
            if not klines:
                break
            start = klines[-1][0] + 1 
            all_kline.extend(klines)
            if klines[-1][0] >= end:
                break
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_kline, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True)
    df.set_index('Datetime', inplace=True)
    return df

def generate_financial_columns(group):
    post_count = len(group)
    btc_count = group["sentiment_btc"].notna().sum()
    eth_count = group["sentiment_eth"].notna().sum()
    sol_count = group["sentiment_sol"].notna().sum()
    gen_count = group["sentiment_gen"].notna().sum()
    
    bullish_btc = (group["sentiment_btc"] == 2).sum()
    bearish_btc = (group["sentiment_btc"] == 0).sum()
    bullish_eth = (group["sentiment_eth"] == 2).sum()
    bearish_eth = (group["sentiment_eth"] == 0).sum()
    bullish_sol = (group["sentiment_sol"] == 2).sum()
    bearish_sol = (group["sentiment_sol"] == 0).sum()
    bullish_gen = (group["sentiment_gen"] == 2).sum()
    bearish_gen = (group["sentiment_gen"] == 0).sum()

    return pd.Series({
        "total_post_count": post_count,
        "BTC_bullish_ratio": bullish_btc / post_count if post_count else 0,
        "ETH_bullish_ratio": bullish_eth / post_count if post_count else 0,
        "SOL_bullish_ratio": bullish_sol / post_count if post_count else 0,
        "GEN_bullish_ratio": bullish_gen / post_count if post_count else 0,
        "BTC_bearish_ratio": bearish_btc / post_count if post_count else 0,
        "ETH_bearish_ratio": bearish_eth / post_count if post_count else 0,
        "SOL_bearish_ratio": bearish_sol / post_count if post_count else 0,
        "GEN_bearish_ratio": bearish_gen / post_count if post_count else 0,
        "BTC_sentiment_score": (bullish_btc - bearish_btc) / btc_count if btc_count else 0,
        "ETH_sentiment_score": (bullish_eth - bearish_eth) / eth_count if eth_count else 0,
        "SOL_sentiment_score": (bullish_sol - bearish_sol) / sol_count if sol_count else 0,
        "GEN_sentiment_score": (bullish_gen - bearish_gen) / gen_count if gen_count else 0,
        "BTC_weighted_sentiment": ((bullish_btc - bearish_btc) / btc_count) * post_count if btc_count else 0,
        "ETH_weighted_sentiment": ((bullish_eth - bearish_eth) / eth_count) * post_count if eth_count else 0,
        "SOL_weighted_sentiment": ((bullish_sol - bearish_sol) / sol_count) * post_count if sol_count else 0,
        "GEN_weighted_sentiment": ((bullish_gen - bearish_gen) / gen_count) * post_count if gen_count else 0
    })

def get_reddit_content(datasets, svm_model):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VolatilityPipelineBot/2.0"}
    all_dfs = []
    
    for dataset in datasets:
        url = f"https://www.reddit.com/r/{dataset}/new.json?limit=100"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                continue
            data = response.json()
            children = data.get("data", {}).get("children", [])
            records = []
            for post in children:
                p_data = post.get("data", {})
                created_utc = p_data.get("created_utc")
                title = p_data.get("title", "")
                selftext = p_data.get("selftext", "")
                if not created_utc or not title:
                    continue
                clean_text = clean_reddit_text(f"{title} {selftext}")
                records.append({
                    "created": pd.to_datetime(created_utc, unit="s", utc=True),
                    "text": clean_text
                })
            if not records:
                continue
            sub_df = pd.DataFrame(records)
            if dataset == "bitcoin":
                sub_df["sentiment_btc"] = svm_model.predict(sub_df["text"])
            elif dataset == "ethereum":
                sub_df["sentiment_eth"] = svm_model.predict(sub_df["text"])
            elif dataset == "solana":
                sub_df["sentiment_sol"] = svm_model.predict(sub_df["text"])
            else:
                sub_df["sentiment_gen"] = svm_model.predict(sub_df["text"])
            all_dfs.append(sub_df)
            time.sleep(2)
        except Exception as e:
            print(f"Failed parsing streaming data for r/{dataset}: {e}")
            
    if not all_dfs:
        return pd.DataFrame()
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df.set_index("created").sort_index()

def main():

    now = datetime.now(timezone.utc)
    start_history = now - timedelta(days=2) # Pulls last 48 hours to ensure lag features fill

    print(f"Fetching CCXT price feeds from {start_history} to {now}...")
    btc_raw = fetch_ccxt("BTC/USDT", start_history, now)
    eth_raw = fetch_ccxt("ETH/USDT", start_history, now)
    sol_raw = fetch_ccxt("SOL/USDT", start_history, now)

    btc = prepare_market_data(btc_raw, "BTC-USD")
    eth = prepare_market_data(eth_raw, "ETH-USD")
    sol = prepare_market_data(sol_raw, "SOL-USD")

    subreddits = ["bitcoin", "cryptocurrency", "cryptocurrencies", "crypto_currency_news", "ethereum", "solana"]
    prefixes = ['BTC', 'ETH', 'SOL', 'GEN']
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    svm_model = pickle.load(open("data/models/svm.pkl", "rb"))
    
    df_final = get_reddit_content(subreddits, svm_model)
    if df_final.empty:
        print("No new entries, stopping.")
        return

    df_final = df_final.resample("60min").apply(generate_financial_columns)
    df_final["BTC_delta_sentiment"] = df_final["BTC_sentiment_score"].diff().fillna(0)
    df_final["ETH_delta_sentiment"] = df_final["ETH_sentiment_score"].diff().fillna(0)
    df_final["SOL_delta_sentiment"] = df_final["SOL_sentiment_score"].diff().fillna(0)
    df_final["GEN_delta_sentiment"] = df_final["GEN_sentiment_score"].diff().fillna(0)
    
    df_final = df_final.join([btc, eth, sol], how="inner")

    df_final['hour_of_day'] = df_final.index.hour
    df_final['hour_sin'] = np.sin(2 * np.pi * df_final['hour_of_day'] / 24)
    df_final['hour_cos'] = np.cos(2 * np.pi * df_final['hour_of_day'] / 24)     
    df_final['day_of_week'] = df_final.index.dayofweek
    df_final['day_sin'] = np.sin(2 * np.pi * df_final['day_of_week'] / 7)
    df_final['day_cos'] = np.cos(2 * np.pi * df_final['day_of_week'] / 7)   
    df_final.drop(columns=['hour_of_day', 'day_of_week'], inplace=True)
    
    for p in prefixes:
        df_final[f'{p}_polarity'] = df_final[f'{p}_bullish_ratio'] - df_final[f'{p}_bearish_ratio']
        
    features_to_lag = []
    for p in prefixes:
        features_to_lag.append(f'{p}_polarity')
        features_to_lag.append(f'{p}_delta_sentiment')
    for t in tickers:
        features_to_lag.append(f'{t}_return')
        features_to_lag.append(f'{t}_volatility')
        features_to_lag.append(f'{t}_vol_log')
        
    for col in features_to_lag:
        df_final[f'{col}_lag_1'] = df_final[col].shift(1)
        df_final[f'{col}_lag_3'] = df_final[col].shift(3)
        df_final[f'{col}_lag_6'] = df_final[col].shift(6)
        df_final[f'{col}_lag_12'] = df_final[col].shift(12)
        
    df_final.drop(columns=[f'{p}_polarity' for p in prefixes], inplace=True)
    
    df_final['BTC_vol_target'] = pd.cut(df_final['BTC-USD_volatility'], bins=[-np.inf, 0.015, 0.035, np.inf], labels=[0, 1, 2])
    df_final['SOL_vol_target'] = pd.cut(df_final['SOL-USD_volatility'], bins=[-np.inf, 0.015, 0.035, np.inf], labels=[0, 1, 2])
    df_final['ETH_vol_target'] = pd.cut(df_final['ETH-USD_volatility'], bins=[-np.inf, 0.015, 0.035, np.inf], labels=[0, 1, 2])

    df_final['BTC_vol_target'] = df_final['BTC_vol_target'].shift(-1)
    df_final['SOL_vol_target'] = df_final['SOL_vol_target'].shift(-1)
    df_final['ETH_vol_target'] = df_final['ETH_vol_target'].shift(-1)
    
    df_final['post_volume_24h_avg'] = df_final['total_post_count'].rolling(window=24).mean()
    df_final['post_spike_ratio'] = df_final['total_post_count'] / (df_final['post_volume_24h_avg'] + 1)
    
    lag_cols = [f'{col}_lag_{lag}' for col in features_to_lag for lag in [1, 3, 6, 12]]
    df_final = df_final.dropna(subset=['BTC_vol_target', 'post_volume_24h_avg'] + lag_cols)
    
    latest_completed_row = df_final.iloc[[-1]]
    
    csv_path = "data/clean_reddit/live_data.csv"
    
    if not os.path.exists(csv_path):
        latest_completed_row.to_csv(csv_path, index=True)
        print("Created new CSV file")
    else:
        latest_completed_row.to_csv(csv_path, mode='a', header=False, index=True)
        print("Successfully appended new hourly data.")

if __name__ == "__main__":
    main()