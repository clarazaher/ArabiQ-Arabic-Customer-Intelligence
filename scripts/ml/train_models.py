"""
Train and compare five churn prediction models for ArabiQ.

This script:
1. Loads the engineered churn dataset.
2. Builds the preprocessing pipeline.
3. Trains five machine learning models.
4. Evaluates each model using AUC, F1, precision, and recall.
5. Saves trained models locally.
6. Saves the model comparison table to reports/model_comparison.csv.

Important:
- Model files are saved locally in models/ml/.
- The models folder is ignored by Git because trained models can be large.
"""

from pathlib import Path
import time
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from preprocess import build_preprocessing_pipeline


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data/processed/churn_engineered.csv"
MODELS_DIR = PROJECT_ROOT / "models/ml"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_reports.txt"


def safe_model_filename(model_name: str) -> str:
    """Convert a model name into a safe filename."""
    return model_name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    """Train and evaluate all churn prediction models."""

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. "
            "Run scripts/ml/feature_engineering.py first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading engineered dataset from:")
    print(ENGINEERED_DATA_PATH)

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    print("Dataset shape:", df.shape)

    X_train, X_test, y_train, y_test, _ = build_preprocessing_pipeline(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print("\nClass balance in training data:")
    print(f"Non-churn customers: {negative_count}")
    print(f"Churn customers: {positive_count}")
    print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
        ),
    }

    results = []
    report_sections = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        start_time = time.time()

        cv_auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        elapsed = time.time() - start_time

        test_auc = roc_auc_score(y_test, y_prob)
        test_f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "CV AUC Mean": round(cv_auc.mean(), 4),
                "CV AUC Std": round(cv_auc.std(), 4),
                "Test AUC": round(test_auc, 4),
                "Test F1": round(test_f1, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "Train Time Seconds": round(elapsed, 1),
            }
        )

        report = classification_report(y_test, y_pred)
        report_sections.append(f"\n{'=' * 80}\n{name}\n{'=' * 80}\n{report}")

        model_path = MODELS_DIR / f"{safe_model_filename(name)}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")
        print(f"Test AUC: {test_auc:.4f} | F1: {test_f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("Test AUC", ascending=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report_sections))

    print("\nSaved model comparison to:")
    print(MODEL_COMPARISON_PATH)

    print("\nSaved classification reports to:")
    print(CLASSIFICATION_REPORT_PATH)


if __name__ == "__main__":
    main()
