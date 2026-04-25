import re
import ollama
import pandas as pd
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier

TARGET_NAMES = ["Bearish", "Bullish", "Neutral"]
LABEL_ORDER  = [0, 1, 2]
BATCH_SIZE   = 128

def clean_text(text: str) -> str:

    text = str(text)
    text = re.sub(r'http\S+', '', text) # URLs
    text = re.sub(r'@\w+', '', text) # @mentions
    text = re.sub(r'\$([A-Z]{2,10})', r'CRYPTO_\1', text) # $BTC → CRYPTO_BTC
    text = re.sub(r'#(\w+)', r'\1', text) # #hashtag → hashtag
    text = re.sub(r'\s+', ' ', text).strip()
    return text

vader = SentimentIntensityAnalyzer()

def get_vader_sentiment(text: str) -> int:
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return 1   # Bullish
    elif compound <= -0.05:
        return 0   # Bearish
    else:
        return 2   # Neutral

cryptobert = pipeline("text-classification", model="ElKulako/cryptobert",tokenizer="ElKulako/cryptobert")
BERT_MAP = {"Bearish":  0, "Bullish":  1, "Neutral":  2}

def run_cryptobert(texts: list[str]) -> list[int]:
    results = cryptobert(texts, truncation=False, batch_size=BATCH_SIZE,)
    labels = []
    for res in results:
        raw = res["label"]
        mapped = BERT_MAP.get(raw)
        labels.append(mapped)
    return labels

class SentimentResponse(BaseModel):
    label: int  # 0=Bearish, 1=Bullish, 2=Neutral

FEW_SHOT_PROMPT = """\
Classify the financial sentiment of a crypto social media post.
Respond ONLY with a JSON object in the form: {{"label": <int>}}

Label definitions:
  0 = Bearish  (negative outlook, fear, crash, sell-off, loss)
  1 = Bullish  (positive outlook, gains, ATH, moon, buy signal)
  2 = Neutral  (news, announcements, factual info, no clear direction)

Examples:
Post: "BTC is crashing hard, I'm selling everything, this bear market is brutal"
Output: {{"label": 0}}

Post: "ETH breaking ATH again! This bull run is just starting, buy the dip"
Output: {{"label": 1}}

Post: "Binance has announced support for 5 new trading pairs starting Monday"
Output: {{"label": 2}}

Post: "I panic sold at the bottom and now it's pumping back, classic me"
Output: {{"label": 0}}

Post: "Diamond hands since 2018, up 400% on my BTC position, not selling"
Output: {{"label": 1}}

Now classify this post:
Post: "{text}"
Output:"""

def get_gemma_sentiment(text: str) -> int | None:
    try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[{
                "role": "user",
                "content": FEW_SHOT_PROMPT.format(text=text),
            }],
            format=SentimentResponse.model_json_schema(),
            options={"temperature": 0.0, "num_ctx": 2048, "num_gpu": 100,"think": False,}
        ,)
        output = SentimentResponse.model_validate_json(response.message.content)
        if output.label not in (0, 1, 2):
            print(f"Gemma returned out-of-range label {output.label} — defaulting to Neutral")
            return 2
        return output.label
    except Exception as e:
        print(f"[ERROR] Gemma inference failed: {e}")
        return None

def evaluate_model(model_name: str, y_true, y_pred):
    
    print(f" {model_name}")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=TARGET_NAMES, zero_division=0))


if __name__ == "__main__":

    df_train = pd.read_csv("data/sentiment/sent_train.csv")
    df_test  = pd.read_csv("data/sentiment/sent_valid.csv")

    df_train["text_clean"] = df_train["text"].apply(clean_text)
    df_test["text_clean"] = df_test["text"].apply(clean_text)

    X_train = df_train["text_clean"]
    y_train = df_train["label"]
    X_test = df_test["text_clean"]
    y_test = df_test["label"]

    results_df = pd.DataFrame({"text": X_test, "label": y_test})

    print("Running VADER...")
    results_df["vader_pred"] = results_df["text"].apply(get_vader_sentiment)

    print("Running CryptoBERT with raw text...")
    results_df["cryptobert_pred"] = run_cryptobert(df_test["text"].tolist())
    
    print("Training & running SVM...")
    svm_pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2),stop_words="english",)),("clf", LinearSVC()),])
    
    svm_pipeline.fit(X_train, y_train)
    results_df["svm_pred"] = svm_pipeline.predict(X_test)
    
    print("Training & running Random Forest...")
    rf_pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"[^\s]+",ngram_range=(1, 2),stop_words="english",)),
                            ("clf", RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced", n_jobs=-1,random_state=1,)),])

    rf_pipeline.fit(X_train, y_train)
    results_df["rf_pred"] = rf_pipeline.predict(X_test)
    
    print("Running Gemma 2B (this takes a while)...")
    results_df["gemma_pred"] = results_df["text"].apply(get_gemma_sentiment)
    
    evaluate_model("VADER", results_df["label"], results_df["vader_pred"])
    evaluate_model("CryptoBERT", results_df["label"], results_df["cryptobert_pred"])
    evaluate_model("SVM", results_df["label"], results_df["svm_pred"])
    evaluate_model("Random Forest", results_df["label"], results_df["rf_pred"])
    evaluate_model("Gemma", results_df["label"], results_df["gemma_pred"])

    print(" Inter-model Agreement (Cohen's Kappa)")
    model_cols = ["vader_pred", "cryptobert_pred", "svm_pred", "gemma_pred", "rf_pred"]
    model_labels = ["VADER", "CryptoBERT", "SVM", "Gemma", "Random Forest"]
    for i in range(len(model_cols)):
        for j in range(i + 1, len(model_cols)):
            col_a, col_b = model_cols[i], model_cols[j]
            kappa = cohen_kappa_score(results_df[col_a], results_df[col_b])
            print(f"  {model_labels[i]:12s} vs {model_labels[j]:12s} : κ = {kappa:.4f}")

    results_df.to_csv("data/sentiment/results_comparison.csv", index=False)
    print("\nResults saved to data/sentiment/results_comparison.csv")