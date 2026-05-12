import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

def train_and_evaluate(model, x, y, tss, model_name):
    """
    Train and evaluate a classifier using time-series cross-validation.
    Feature importances are averaged across all folds.
    """
    fold = 1
    all_importances = []

    for train_index, test_index in tss.split(x):
        print(f"--- Fold {fold} ---")
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "XGBoost":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(x_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        print(classification_report(y_test, y_pred, zero_division=0))

        all_importances.append(model.feature_importances_)
        fold += 1

    mean_importances = np.mean(all_importances, axis=0)
    feature_importance_df = (
        pd.DataFrame({"Feature": x.columns, "Importance": mean_importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"][:15], feature_importance_df["Importance"][:15])
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Features — {model_name} (avg. across folds)")
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("data/clean_reddit/final_submission_clean.csv")
    df.set_index("index", inplace=True)
    y = df["BTC_vol_target"].astype(int)

    to_drop_cols = ["BTC_vol_target", "ETH_vol_target", "SOL_vol_target",
        "BTC-USD_close", "ETH-USD_close", "SOL-USD_close"
    ]
    x = df.drop(columns=to_drop_cols)
    
    print(x.isnull().sum()[x.isnull().sum() > 0])

    tss = TimeSeriesSplit(n_splits=5)

    model_xgb = GradientBoostingClassifier(
        max_depth=5,
        random_state=42,
    )
    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    model_lgb = lgb.LGBMClassifier(
        objective="multiclass",
        max_depth=3,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )

    print("Training XGBoost Classifier")
    train_and_evaluate(model_xgb, x, y, tss, "XGBoost")

    print("Training Random Forest Classifier")
    train_and_evaluate(model_rf, x, y, tss, "Random Forest")

    print("Training LightGBM Classifier")
    train_and_evaluate(model_lgb, x, y, tss, "LightGBM")

if __name__ == "__main__":
    main()