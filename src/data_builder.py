import pandas as pd
import re
import pickle
import ccxt

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
    
    return df[[f"{ticker}_close", f"{ticker}_volume", f"{ticker}_return", f"{ticker}_volatility"]]

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
    neutral_btc = (group["sentiment_btc"] == 1).sum()   
    bullish_eth = (group["sentiment_eth"] == 2).sum()
    bearish_eth = (group["sentiment_eth"] == 0).sum()
    neutral_eth = (group["sentiment_eth"] == 1).sum()
    bullish_sol = (group["sentiment_sol"] == 2).sum()
    bearish_sol = (group["sentiment_sol"] == 0).sum()
    neutral_sol = (group["sentiment_sol"] == 1).sum()
    bullish_gen = (group["sentiment_gen"] == 2).sum()
    bearish_gen = (group["sentiment_gen"] == 0).sum()
    neutral_gen = (group["sentiment_gen"] == 1).sum()
    neutral_btc = (group["sentiment_btc"] == 1).sum()   
    neutral_eth = (group["sentiment_eth"] == 1).sum()
    neutral_sol = (group["sentiment_sol"] == 1).sum()
    neutral_gen = (group["sentiment_gen"] == 1).sum()

    return pd.Series({
        "total_post_count": post_count,
        "btc_bullish_count": bullish_btc,
        "eth_bullish_count": bullish_eth,
        "sol_bullish_count": bullish_sol,
        "gen_bullish_count": bullish_gen,
        "btc_bearish_count": bearish_btc,
        "eth_bearish_count": bearish_eth,
        "sol_bearish_count": bearish_sol,
        "gen_bearish_count": bearish_gen,
        "btc_neutral_count": neutral_btc,
        "eth_neutral_count": neutral_eth,
        "sol_neutral_count": neutral_sol,
        "gen_neutral_count": neutral_gen,
        "btc_sentiment_score": (bullish_btc - bearish_btc) / btc_count if btc_count else 0,
        "eth_sentiment_score": (bullish_eth - bearish_eth) / eth_count if eth_count else 0,
        "sol_sentiment_score": (bullish_sol - bearish_sol) / sol_count if sol_count else 0,
        "gen_sentiment_score": (bullish_gen - bearish_gen) / gen_count if gen_count else 0
    })
    

def clean_dataset(df_btc, df_eth, df_sol) -> None:
    
    datasets = ["bitcoin", "cryptocurrency", "cryptocurrencies", "crypto_currency_news", "ethereum", "solana"]
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
    df_final["btc_delta_sentiment"] = df_final["btc_sentiment_score"].diff().fillna(0)
    df_final["eth_delta_sentiment"] = df_final["eth_sentiment_score"].diff().fillna(0)
    df_final["sol_delta_sentiment"] = df_final["sol_sentiment_score"].diff().fillna(0)
    df_final["gen_delta_sentiment"] = df_final["gen_sentiment_score"].diff().fillna(0)
    df_final = df_final.join([df_btc, df_eth, df_sol], how="inner")
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
    
