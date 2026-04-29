import ollama
import pandas as pd
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import RandomForestClassifier

TARGET_NAMES = ["Bearish", "Neutral", "Bullish"]
LABEL_ORDER  = [0, 1, 2]

vader = SentimentIntensityAnalyzer()

def get_vader_sentiment(text: str) -> int:
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return 2   # Bullish
    elif compound <= -0.05:
        return 0   # Bearish
    else:
        return 1   # Neutral
class SentimentResponse(BaseModel):
    label: int  # 0=Bearish, 1=Neutral, 2=Bullish

FEW_SHOT_PROMPT = """\
Classify the financial sentiment of a crypto social media post.
Respond ONLY with a JSON object in the form: {{"label": <int>}}

Label definitions:
  0 = Bearish  (negative outlook, fear, crash, sell-off, loss)
  1 = Neutral  (positive outlook, gains, ATH, moon, buy signal)
  2 = Bullish  (news, announcements, factual info, no clear direction)

Examples:
Post: "BTC is crashing hard, I'm selling everything, this bear market is brutal"
Output: {{"label": 0}}

Post: "ETH breaking ATH again! This bull run is just starting, buy the dip"
Output: {{"label": 2}}

Post: "Binance has announced support for 5 new trading pairs starting Monday"
Output: {{"label": 1}}

Post: "Diamond hands since 2018, up 400% on my BTC position, not selling"
Output: {{"label": 2}}

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
            return 0
        return output.label
    except Exception as e:
        print(f"[ERROR] Gemma inference failed: {e}")
        return None

def evaluate_model(model_name: str, y_true, y_pred):
    
    print(f" {model_name}")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=TARGET_NAMES, zero_division=0))

if __name__ == "__main__":

    df_train_1 = pd.read_csv("data/sentiment/stockwits_train_1.csv")
    df_train_2 = pd.read_csv("data/sentiment/stockwits_train_2.csv")
    df_test = pd.read_csv("data/sentiment/stockwits_test.csv")

    df_train = pd.concat([df_train_1, df_train_2], ignore_index=True)
    
    x_train, y_train = df_train["text"].tolist(), df_train["label"].tolist()
    x_test, y_test = df_test["text"].tolist(), df_test["label"].tolist()
    
    results_df = pd.DataFrame({"text": x_test, "label": y_test})

    print("Running VADER...")
    results_df["vader_pred"] = results_df["text"].apply(get_vader_sentiment)
    
    print("Training & running SVM...")
    svm_pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 5),  min_df=1, max_df=0.95,)),
                             ("clf", LinearSVC(class_weight="balanced", dual=False, max_iter=2000)),])
    
    svm_pipeline.fit(x_train, y_train)
    results_df["svm_pred"] = svm_pipeline.predict(x_test)
    
    print("Training & running Random Forest...")
    rf_pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 5),  min_df=1, max_df=0.95,)),
                            ("clf", RandomForestClassifier(n_estimators=250, max_depth=None, class_weight="balanced", n_jobs=-1,random_state=1,)), ])

    rf_pipeline.fit(x_train, y_train)
    results_df["rf_pred"] = rf_pipeline.predict(x_test)
    
    print("Running Gemma 2B (this takes a while)...")
    results_df["gemma_pred"] = results_df["text"].apply(get_gemma_sentiment)
    
    evaluate_model("VADER", results_df["label"], results_df["vader_pred"])
    evaluate_model("SVM", results_df["label"], results_df["svm_pred"])
    evaluate_model("Random Forest", results_df["label"], results_df["rf_pred"])
    evaluate_model("Gemma", results_df["label"], results_df["gemma_pred"])

    print(" Inter-model Agreement (Cohen's Kappa)")
    model_cols = ["vader_pred", "svm_pred", "gemma_pred", "rf_pred"]
    model_labels = ["VADER", "SVM", "Gemma", "Random Forest"]
    for i in range(len(model_cols)):
        for j in range(i + 1, len(model_cols)):
            col_a, col_b = model_cols[i], model_cols[j]
            kappa = cohen_kappa_score(results_df[col_a], results_df[col_b])
            print(f"  {model_labels[i]:12s} vs {model_labels[j]:12s} : κ = {kappa:.4f}")