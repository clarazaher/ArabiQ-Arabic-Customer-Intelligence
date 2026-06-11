"""
TF-IDF + Logistic Regression baseline for Arabic sentiment analysis.

This script:
1. Loads the cleaned Arabic sentiment dataset.
2. Splits it into train/test sets.
3. Converts Arabic text into TF-IDF character n-gram features.
4. Trains a Logistic Regression classifier.
5. Evaluates the model using accuracy, macro F1, precision, and recall.
6. Saves the baseline results to reports/.
7. Saves the model and vectorizer locally in models/nlp/.

Important:
- Model files are ignored by Git.
- Reports and scripts are safe to commit.
"""

from pathlib import Path

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

RESULTS_PATH = REPORTS_DIR / "arabic_sentiment_baseline_results.csv"
REPORT_PATH = REPORTS_DIR / "arabic_sentiment_baseline_classification_report.txt"

MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


def main() -> None:
    """Train and evaluate the TF-IDF baseline model."""

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

    required_columns = {"clean_text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

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

    # Character n-grams work well for Arabic because Arabic words can have many prefixes/suffixes.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True,
    )

    print("\nVectorizing Arabic text...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Test TF-IDF shape:", X_test_tfidf.shape)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    print("\nTraining Logistic Regression baseline...")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    results = {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Macro Precision": round(precision, 4),
        "Macro Recall": round(recall, 4),
    }

    print("\nBASELINE RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"],
    )

    print("\nClassification report:")
    print(report)

    pd.DataFrame([results]).to_csv(RESULTS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nSaved baseline results to:")
    print(RESULTS_PATH)

    print("\nSaved classification report to:")
    print(REPORT_PATH)

    print("\nSaved model locally to:")
    print(MODEL_PATH)

    print("\nSaved vectorizer locally to:")
    print(VECTORIZER_PATH)


if __name__ == "__main__":
    main()
