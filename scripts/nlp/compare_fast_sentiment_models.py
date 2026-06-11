"""
Compare fast Arabic sentiment models.

This script:
1. Loads the cleaned Arabic sentiment dataset.
2. Builds TF-IDF character n-gram features.
3. Trains multiple fast machine learning models.
4. Compares them using Accuracy, Macro F1, Precision, and Recall.
5. Saves comparison results and classification reports.
6. Saves the best model locally.

Why this matters:
This gives us a stronger classical NLP benchmark before using heavier transformer models
like CAMeLBERT.
"""

from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
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

RESULTS_PATH = REPORTS_DIR / "arabic_sentiment_fast_model_comparison.csv"
REPORT_PATH = REPORTS_DIR / "arabic_sentiment_fast_model_classification_reports.txt"

BEST_MODEL_PATH = MODELS_DIR / "best_fast_sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "best_fast_tfidf_vectorizer.pkl"


def evaluate_model(model_name, model, X_train_tfidf, X_test_tfidf, y_train, y_test):
    """Train and evaluate one model."""

    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    start_time = time.time()

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    train_time = round(time.time() - start_time, 2)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    result = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Macro Precision": round(precision, 4),
        "Macro Recall": round(recall, 4),
        "Train Time Seconds": train_time,
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"],
    )

    print("Results:")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(report)

    return result, report, model


def main():
    """Run fast Arabic sentiment model comparison."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean Arabic sentiment dataset not found at {DATA_PATH}. "
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

    print("\nLabel distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Test TF-IDF shape:", X_test_tfidf.shape)

    models = {
        "TF-IDF + Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "TF-IDF + Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
        "TF-IDF + Complement Naive Bayes": ComplementNB(),
    }

    all_results = []
    all_reports = []
    trained_models = {}

    for model_name, model in models.items():
        result, report, trained_model = evaluate_model(
            model_name,
            model,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
        )

        all_results.append(result)
        trained_models[model_name] = trained_model

        all_reports.append(
            f"\n{'=' * 80}\n"
            f"{model_name}\n"
            f"{'=' * 80}\n"
            f"{report}\n"
        )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="Macro F1", ascending=False)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    results_df.to_csv(RESULTS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(all_reports))
        file.write(f"\nBest model by Macro F1: {best_model_name}\n")

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nBest model by Macro F1:")
    print(best_model_name)

    print("\nSaved comparison results to:")
    print(RESULTS_PATH)

    print("\nSaved classification reports to:")
    print(REPORT_PATH)

    print("\nSaved best model locally to:")
    print(BEST_MODEL_PATH)

    print("\nSaved TF-IDF vectorizer locally to:")
    print(VECTORIZER_PATH)


if __name__ == "__main__":
    main()
