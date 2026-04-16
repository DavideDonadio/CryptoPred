import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

TARGET_NAMES = ["Bearish", "Neutral", "Bullish"]
LABEL_ORDER = [0, 1, 2]

def evaluate_model(model_name, y_true, y_pred):
    print(f"{model_name} Performance")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=TARGET_NAMES, zero_division=0))
    
def get_vader_sentiment(text):
    comp = vader.polarity_scores(text)["compound"]
    if(comp >= 0.05):
        return 2
    elif(comp <= -0.05):
        return 0
    else:
        return 1
 
cryptobert = pipeline("text-classification", model="ElKulako/cryptobert", tokenizer="ElKulako/cryptobert")
vader = SentimentIntensityAnalyzer()

df = pd.read_excel("data/sentiment/st-data-full.xlsx").dropna(subset=["text", "label"])
test_df = df.sample(5000, random_state=42).copy()

test_df["vader_sentiment"] = test_df["text"].apply(get_vader_sentiment)

bert_res = cryptobert(test_df["text"].tolist(), truncation=True)
bert_map = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
test_df["cryptobert_sentiment"] = [bert_map[res["label"]] for res in bert_res]

evaluate_model("CRYPTOBERT Sentiment Analysis", test_df["label"], test_df["cryptobert_sentiment"])
evaluate_model("VADER Sentiment Analysis", test_df["label"], test_df["vader_sentiment"])