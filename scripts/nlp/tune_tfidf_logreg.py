"""
Tune TF-IDF + Logistic Regression for Arabic sentiment analysis.

This script:
1. Loads the cleaned Arabic sentiment dataset.
2. Tests several TF-IDF configurations.
3. Tests several Logistic Regression C values.
4. Selects the best model by Macro F1.
5. Saves tuning results and the best model locally.

The goal is to improve on the baseline Macro F1 of 0.6810.
"""

from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed/arabic_sentiment_clean.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models/nlp"

RESULTS_PATH = REPORTS_DIR / "arabic_sentiment_tuned_logreg_results.csv"
REPORT_PATH = REPORTS_DIR / "arabic_sentiment_tuned_logreg_classification_report.txt"

BEST_MODEL_PATH = MODELS_DIR / "tuned_tfidf_logreg_model.pkl"
BEST_VECTORIZER_PATH = MODELS_DIR / "tuned_tfidf_vectorizer.pkl"


TFIDF_CONFIGS = [
    {
        "name": "char_wb_2_5_50k",
        "analyzer": "char_wb",
        "ngram_range": (2, 5),
        "max_features": 50000,
    },
    {
        "name": "char_wb_3_5_80k",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 80000,
    },
    {
        "name": "char_wb_2_6_100k",
        "analyzer": "char_wb",
        "ngram_range": (2, 6),
        "max_features": 100000,
    },
    {
        "name": "char_3_5_80k",
        "analyzer": "char",
        "ngram_range": (3, 5),
        "max_features": 80000,
    },
    {
        "name": "word_1_2_50k",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 50000,
    },
]

C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


def evaluate(y_true, y_pred):
    """Return common classification metrics."""

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro F1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Macro Precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "Macro Recall": round(recall_score(y_true, y_pred, average="macro"), 4),
    }


def main():
    """Run TF-IDF + Logistic Regression tuning."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {DATA_PATH}. "
            "Run scripts/nlp/arabic_preprocess.py first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned Arabic sentiment dataset from:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)

    X = df["clean_text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    all_results = []

    best_macro_f1 = -1
    best_model = None
    best_vectorizer = None
    best_report = None
    best_description = None

    for tfidf_config in TFIDF_CONFIGS:
        print("\n" + "=" * 80)
        print(f"TF-IDF config: {tfidf_config['name']}")
        print("=" * 80)

        vectorizer = TfidfVectorizer(
            analyzer=tfidf_config["analyzer"],
            ngram_range=tfidf_config["ngram_range"],
            max_features=tfidf_config["max_features"],
            sublinear_tf=True,
            min_df=2,
        )

        start_vector_time = time.time()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        vector_time = round(time.time() - start_vector_time, 2)

        print("Train TF-IDF shape:", X_train_tfidf.shape)
        print("Test TF-IDF shape:", X_test_tfidf.shape)
        print("Vectorization time:", vector_time, "seconds")

        for c_value in C_VALUES:
            print(f"\nTraining Logistic Regression with C={c_value}")

            model = LogisticRegression(
                C=c_value,
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
                solver="liblinear",
            )

            start_train_time = time.time()
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            train_time = round(time.time() - start_train_time, 2)

            metrics = evaluate(y_test, y_pred)

            result = {
                "Model": "TF-IDF + Tuned Logistic Regression",
                "TF-IDF Config": tfidf_config["name"],
                "Analyzer": tfidf_config["analyzer"],
                "Ngram Range": str(tfidf_config["ngram_range"]),
                "Max Features": tfidf_config["max_features"],
                "C": c_value,
                **metrics,
                "Vectorization Time Seconds": vector_time,
                "Train Time Seconds": train_time,
            }

            all_results.append(result)

            print(result)

            if metrics["Macro F1"] > best_macro_f1:
                best_macro_f1 = metrics["Macro F1"]
                best_model = model
                best_vectorizer = vectorizer
                best_description = result
                best_report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["negative", "positive"],
                )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="Macro F1", ascending=False)

    print("\n" + "=" * 80)
    print("FINAL TUNING RESULTS")
    print("=" * 80)
    print(results_df.head(10).to_string(index=False))

    print("\nBest configuration:")
    print(best_description)

    results_df.to_csv(RESULTS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Best TF-IDF + Logistic Regression configuration\n")
        file.write("=" * 80)
        file.write("\n")
        file.write(str(best_description))
        file.write("\n\nClassification report:\n")
        file.write(best_report)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(best_vectorizer, BEST_VECTORIZER_PATH)

    print("\nSaved tuning results to:")
    print(RESULTS_PATH)

    print("\nSaved best classification report to:")
    print(REPORT_PATH)

    print("\nSaved best tuned model locally to:")
    print(BEST_MODEL_PATH)

    print("\nSaved best tuned vectorizer locally to:")
    print(BEST_VECTORIZER_PATH)


if __name__ == "__main__":
    main()
