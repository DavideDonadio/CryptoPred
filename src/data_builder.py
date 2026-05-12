import pandas as pd
import re
import pickle
import ccxt
import numpy as np

def clean_reddit_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('&#x200B;', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_market_data(df, ticker):
    df = df.rename(columns={
        "Close": f"{ticker}_close",
        "Volume": f"{ticker}_volume"
    })
    
    df[f"{ticker}_return"] = df[f"{ticker}_close"].pct_change()
    df[f"{ticker}_volatility"] = (df["High"] - df["Low"]) / df["Open"]
    df[f"{ticker}_vol_log"] = np.log1p(df[f"{ticker}_volume"])
    
    return df[[f"{ticker}_close", f"{ticker}_return", f"{ticker}_volatility", f"{ticker}_vol_log"]]

def fetch_ccxt(symbol, start_str, end_str, timeframe='1h'):

    exchange = ccxt.binance({'enableRateLimit': True})
    
    start = exchange.parse8601(f"{start_str}T00:00:00Z")
    end = exchange.parse8601(f"{end_str}T23:59:59Z")
    
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
        "GEN_sentiment_score": (bullish_gen - bearish_gen) / gen_count if gen_count else 0
    })
    

def clean_dataset(df_btc, df_eth, df_sol) -> None:
    
    datasets = ["bitcoin", "cryptocurrency", "cryptocurrencies", "crypto_currency_news", "ethereum", "solana"]
    prefixes = ['BTC', 'ETH', 'SOL', 'GEN']
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    all_dfs = []
    svm_model = pickle.load(open("data/models/svm.pkl", "rb"))
    for dataset in datasets:

        df = pd.read_csv(f"data/raw_reddit/{dataset}/submission.csv")

        df = df[(df["removed"] == 0) & (df["deleted"] == 0)]
        df = df.drop(columns=[
        "submission",             
        "author",              
        "retrieved",      
        "edited",    
        "pinned",     
        "archived",     
        "locked",       
        "removed",         
        "deleted",           
        "is_video",                
        "is_original_content",     
        "gilded",                  
        "total_awards_received",  
        "num_crossposts",          
        "thumbnail",               
        "shortlink"                
    ])
        df = df.dropna(subset=["created", "title"])
        df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
        df.drop(columns=["title", "selftext"], inplace=True)
        
        df["text"] = df["text"].apply(clean_reddit_text)
        df["created"] = pd.to_datetime(df["created"], unit="s", utc=True)
        
        if dataset == "bitcoin":
            df["sentiment_btc"] = svm_model.predict(df["text"])
        elif dataset == "ethereum":
            df["sentiment_eth"] = svm_model.predict(df["text"])
        elif dataset == "solana":
            df["sentiment_sol"] = svm_model.predict(df["text"])
        else:
            df["sentiment_gen"] = svm_model.predict(df["text"])

        all_dfs.append(df)

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final = df_final.set_index("created").sort_index()

    df_final = df_final.resample("60min").apply(generate_financial_columns)
    df_final["BTC_delta_sentiment"] = df_final["BTC_sentiment_score"].diff().fillna(0)
    df_final["ETH_delta_sentiment"] = df_final["ETH_sentiment_score"].diff().fillna(0)
    df_final["SOL_delta_sentiment"] = df_final["SOL_sentiment_score"].diff().fillna(0)
    df_final["GEN_delta_sentiment"] = df_final["GEN_sentiment_score"].diff().fillna(0)
    df_final = df_final.join([df_btc, df_eth, df_sol], how="inner")
    
    df_final['hour_of_day'] = df_final.index.hour
    df_final['day_of_week'] = df_final.index.dayofweek
    
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
        
    df_final.drop(columns=[f'{p}_polarity' for p in prefixes], inplace=True)
    
    train_size = int(len(df_final) * 0.7)
    train_df = df_final.iloc[:train_size].copy()

    _, bins = pd.qcut(train_df['BTC-USD_volatility'], q=3, retbins=True)

    bins[0] = -np.inf
    bins[-1] = np.inf

    df_final['BTC_vol_target'] = pd.cut(df_final['BTC-USD_volatility'], bins=bins, labels=[0, 1, 2])
    df_final['SOL_vol_target'] = pd.cut(df_final['SOL-USD_volatility'], bins=bins, labels=[0, 1, 2])
    df_final['ETH_vol_target'] = pd.cut(df_final['ETH-USD_volatility'], bins=bins, labels=[0, 1, 2])

    df_final['BTC_vol_target'] = df_final['BTC_vol_target'].shift(-1)
    df_final['SOL_vol_target'] = df_final['SOL_vol_target'].shift(-1)
    df_final['ETH_vol_target'] = df_final['ETH_vol_target'].shift(-1)
    
    lag_cols = [f'{col}_lag_1' for col in features_to_lag]
    df_final = df_final.dropna(subset=['BTC_vol_target', 'SOL_vol_target', 'ETH_vol_target'] + lag_cols)
    df_final = df_final.reset_index()
      
    df_final.to_csv("data/clean_reddit/final_submission_clean.csv", index=False)
        
if __name__ == "__main__":

    btc_raw = fetch_ccxt("BTC/USDT", "2022-01-01", "2022-12-31")
    eth_raw = fetch_ccxt("ETH/USDT", "2022-01-01", "2022-12-31")
    sol_raw = fetch_ccxt("SOL/USDT", "2022-01-01", "2022-12-31")

    btc = prepare_market_data(btc_raw, "BTC-USD")
    eth = prepare_market_data(eth_raw, "ETH-USD")
    sol = prepare_market_data(sol_raw, "SOL-USD")

    clean_dataset(btc, eth, sol)
    

